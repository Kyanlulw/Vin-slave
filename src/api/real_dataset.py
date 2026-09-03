"""Dataset browsing, Agent evaluation and built-in annotation editor endpoints."""

import asyncio
import io
import json
import logging
import mimetypes
import os
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Annotated, cast
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from google.api_core.exceptions import NotFound
from PIL import Image
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_current_user, get_db_session, get_real_dataset_service, require_roles
from src.config import IngestionSettings
from src.models.admin_control import FrameTask
from src.models.auth_schemas import AuthenticatedUser
from src.models.inference_schemas import InferenceBatchRequest, InferenceImageReference, InferenceResponse
from src.models.ingestion import QAImage, QAObject
from src.models.real_dataset_schemas import (
    AnnotationDocument,
    AnnotationRestoreRequest,
    AnnotationRevisionList,
    AnnotationSaveRequest,
    RealDatasetBBox,
    RealDatasetBatchEvaluation,
    RealDatasetBatchEvaluationRequest,
    RealDatasetBatchEvaluationResult,
    RealDatasetEvaluation,
    RealDatasetFrameSample,
    RealDatasetFrameSampleList,
    RealDatasetImage,
    RealDatasetImageList,
    RealDatasetLabel,
)
from src.services.annotation_editor_service import AnnotationConflictError, AnnotationEditorService
from src.services.google_cloud import create_gcs_storage_client
from src.services.ingestion.yolo_detection_adapter import YoloDatasetLayoutError
from src.services.real_dataset_qa_service import RealDatasetQaService
from src.services.real_dataset_service import RealDatasetService

logger = logging.getLogger(__name__)

_process_gcs_client = None


@dataclass(frozen=True)
class _EvaluationInput:
    image_id: str
    image: RealDatasetImage
    image_payload: bytes | None
    image_reference: InferenceImageReference | None
    revision: int


def _get_process_gcs_client():
    global _process_gcs_client
    if _process_gcs_client is None:
        _process_gcs_client = create_gcs_storage_client(IngestionSettings())
    return _process_gcs_client

_cache_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def _evict_cache_if_needed():
    def do_evict():
        try:
            cache_root = _gcs_cache_root()
            original_dir = cache_root / "original"
            if original_dir.exists():
                size = sum(f.stat().st_size for f in original_dir.glob("*") if f.is_file())
                if size > 10 * 1024 * 1024 * 1024:
                    files = [(f, f.stat().st_mtime) for f in original_dir.glob("*") if f.is_file()]
                    files.sort(key=lambda x: x[1])
                    for f, _ in files[:len(files)//4]:
                        try:
                            f.unlink(missing_ok=True)
                        except OSError:
                            pass
            thumb_dir = cache_root / "thumbnails"
            if thumb_dir.exists():
                size = sum(f.stat().st_size for f in thumb_dir.glob("*") if f.is_file())
                if size > 2 * 1024 * 1024 * 1024:
                    files = [(f, f.stat().st_mtime) for f in thumb_dir.glob("*") if f.is_file()]
                    files.sort(key=lambda x: x[1])
                    for f, _ in files[:len(files)//4]:
                        try:
                            f.unlink(missing_ok=True)
                        except OSError:
                            pass
        except Exception:
            pass
    await asyncio.to_thread(do_evict)


router = APIRouter(
    prefix="/dataset",
    tags=["Real Dataset QA"],
    dependencies=[Depends(get_current_user)],
)

_CAMERA_ORDER = {
    "CAM_FRONT": 0,
    "CAM_FRONT_RIGHT": 1,
    "CAM_BACK_RIGHT": 2,
    "CAM_BACK": 3,
    "CAM_BACK_LEFT": 4,
    "CAM_FRONT_LEFT": 5,
}
_DEFAULT_GCS_CACHE_ROOT = Path("/app/data")
_DATABASE_LIST_TIMEOUT_SECONDS = 20.0


def _not_found(error: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


def _cache_storage_path(root: Path, storage_filename: str) -> Path:
    path = root / storage_filename
    if path.is_file() or PurePosixPath(storage_filename).parts[:1] == ("frames",):
        return path
    return root / "frames" / storage_filename


def _split_from_filename(filename: str, default_split: str) -> str:
    parts = filename.split("/")
    if len(parts) >= 3 and parts[0] == "images":
        return parts[1]
    return default_split


def _database_image_split(image: QAImage, default_split: str) -> str:
    identity = _frame_sample_identity(image.storage_key)
    return identity[0] if identity else _split_from_filename(image.filename, default_split)


def _dataset_release(service: RealDatasetService, dataset: str | None) -> tuple[str, str]:
    """Resolve the canonical release served for each supported dataset."""
    selected_dataset = (dataset or service.dataset_id).lower()
    # All datasets use 'product' as the canonical release for production data.
    release = service.dataset_version
    return selected_dataset, release


def _image_dataset_release(service: RealDatasetService, image: RealDatasetImage) -> tuple[str, str]:
    """Return the dataset identity that owns an image contract."""
    dataset_id = (image.dataset or service.dataset_id).lower()
    dataset_version = image.release or _dataset_release(service, dataset_id)[1]
    return dataset_id, dataset_version


def _database_dataset_conditions(
    service: RealDatasetService,
    dataset: str | None = None,
) -> tuple[object, ...]:
    """Scope metadata reads to the selected dataset and its release/version."""
    selected_dataset = (dataset or service.dataset_id).lower()
    # All datasets use 'product' as the canonical release for production data.
    return (
        func.lower(QAImage.dataset) == selected_dataset,
        QAImage.release == service.dataset_version,
    )


async def _available_database_datasets(
    session: AsyncSession,
    service: RealDatasetService,
) -> list[str]:
    rows = (
        await session.scalars(
            select(QAImage.dataset)
            .where(QAImage.release == service.dataset_version)
            .distinct()
            .order_by(QAImage.dataset)
        )
    ).all()
    return sorted({name for name in rows if name})


def _escaped_like_segment(value: str) -> str:
    """Escape a user-provided path segment before using it in a LIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _database_image_split_condition(split: str) -> object:
    escaped_split = _escaped_like_segment(split)
    return or_(
        QAImage.storage_key.like(f"%/{escaped_split}/frames/%", escape="\\"),
        QAImage.filename.like(f"images/{escaped_split}/%", escape="\\"),
    )


def _database_frame_split_condition(split: str) -> object:
    escaped_split = _escaped_like_segment(split)
    return QAImage.storage_key.like(f"%/{escaped_split}/frames/%", escape="\\")


def _database_metadata_split(storage_key: str | None, filename: str, default_split: str) -> str:
    identity = _frame_sample_identity(storage_key)
    return identity[0] if identity else _split_from_filename(filename, default_split)


def _content_url(split: str, image_id: str) -> str:
    return f"/api/v1/dataset/images/{split}/{image_id}/content"


def _frame_sample_identity(storage_key: str | None) -> tuple[str, str, str, str] | None:
    """Return split, sequence, sample and camera for a canonical frame object key."""
    if not storage_key:
        return None
    parts = PurePosixPath(storage_key.replace("\\", "/")).parts
    try:
        frames_index = parts.index("frames")
    except ValueError:
        return None
    if frames_index == 0 or len(parts) != frames_index + 4:
        return None
    split, sequence_id, sample_id, camera_file = (
        parts[frames_index - 1],
        parts[frames_index + 1],
        parts[frames_index + 2],
        parts[frames_index + 3],
    )
    camera_channel = PurePosixPath(camera_file).stem
    if not camera_channel:
        return None
    return split, sequence_id, sample_id, camera_channel


def _split_from_storage_key(storage_key: str | None, default_split: str) -> str:
    identity = _frame_sample_identity(storage_key)
    if identity is not None:
        return identity[0]
    return default_split


def _dataset_name(row: QAImage) -> str | None:
    return row.dataset or (PurePosixPath(row.storage_key).parts[2] if row.storage_key and len(PurePosixPath(row.storage_key).parts) > 2 else None)


def _matches_dataset(row: QAImage, dataset: str | None) -> bool:
    if not dataset:
        return True
    return (_dataset_name(row) or "").lower() == dataset.lower()


def _selected_split(requested_split: str | None, dataset: str | None, service: RealDatasetService) -> str:
    """Return the requested split or default to 'product'."""
    if requested_split:
        return requested_split
    return cast(str, service.default_split) or "product"


def _gcs_cache_root() -> Path:
    return Path(os.environ.get("LABEL_GUARDIAN_GCS_CACHE_ROOT", str(_DEFAULT_GCS_CACHE_ROOT)))


def _cached_object_path(key: str) -> Path | None:
    normalized_key = key.lstrip("/")
    if ".." in PurePosixPath(normalized_key).parts:
        return None
    path = _gcs_cache_root() / normalized_key
    return path if path.is_file() else None


def _official_cache_roots(service: RealDatasetService, dataset: str | None, split: str) -> list[tuple[str, str, Path]]:
    """Return the canonical cache path(s) for the given dataset and split.

    All datasets use the 'product' canonical layout:
      kitti   → datasets/official/kitti/product/
      nuscenes → datasets/official/nuscenes/product/
    """
    root = _gcs_cache_root() / "datasets" / "official"
    normalized_dataset = (dataset or service.dataset_id).lower()
    resolved_split = split if split else "product"
    if normalized_dataset == "kitti":
        return [("kitti", resolved_split, root / "kitti" / "product")]
    if normalized_dataset == "nuscenes":
        return [("nuscenes", resolved_split, root / "nuscenes" / "product")]
    # Fallback: try both datasets in product layout
    return [
        ("nuscenes", resolved_split, root / "nuscenes" / "product"),
        ("kitti", resolved_split, root / "kitti" / "product"),
    ]


@lru_cache(maxsize=32)
def _load_official_cache(root: str) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    cache_root = Path(root)
    manifest_path = cache_root / "manifests" / "image_manifest.jsonl"
    objects_path = cache_root / "annotations" / "normalized_objects.jsonl"
    images: list[dict[str, object]] = []
    objects_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    images.append(json.loads(line))
    if objects_path.is_file():
        with objects_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                source_image_id = item.get("source_image_id")
                if isinstance(source_image_id, str):
                    objects_by_image[source_image_id].append(item)
    return images, objects_by_image


def _cache_image_contract(
    *,
    dataset: str,
    split: str,
    root: Path,
    image: dict[str, object],
    objects: Sequence[dict[str, object]],
) -> RealDatasetImage | None:
    source_image_id = image.get("source_image_id")
    storage_filename = image.get("storage_filename")
    width = image.get("width")
    height = image.get("height")
    if not isinstance(source_image_id, str) or not isinstance(storage_filename, str):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    storage_path = _cache_storage_path(root, storage_filename)
    storage_parts = PurePosixPath(storage_path.relative_to(root).as_posix()).parts
    if len(storage_parts) < 4:
        return None
    sequence_id, sample_id, camera_file = storage_parts[-3], storage_parts[-2], storage_parts[-1]
    labels: list[RealDatasetLabel] = []
    for index, item in enumerate(objects):
        bbox = item.get("bbox")
        label = item.get("label")
        if not isinstance(bbox, dict) or not isinstance(label, str):
            continue
        try:
            labels.append(
                RealDatasetLabel(
                    id=str(index),
                    class_name=label,
                    bbox=RealDatasetBBox(
                        x1=float(bbox["xmin"]),
                        y1=float(bbox["ymin"]),
                        x2=float(bbox["xmax"]),
                        y2=float(bbox["ymax"]),
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    relative_prefix = root.relative_to(_gcs_cache_root()).as_posix()
    storage_key = f"{relative_prefix}/{storage_path.relative_to(root).as_posix()}"
    return RealDatasetImage(
        id=source_image_id,
        split=split,
        dataset=dataset,
        release="product",
        filename=storage_key,
        width=width,
        height=height,
        label_count=len(labels),
        labels=labels,
        image_url=_content_url(split, source_image_id),
        frame_sample_id=sample_id,
        sequence_id=sequence_id,
        camera_channel=PurePosixPath(camera_file).stem,
    )


def _cached_image_contract(service: RealDatasetService, split: str, image_id: str) -> RealDatasetImage | None:
    for cache_dataset, cache_split, root in _official_cache_roots(service, None, split):
        images, objects_by_image = _load_official_cache(str(root))
        for image in images:
            if image.get("source_image_id") != image_id:
                continue
            return _cache_image_contract(
                dataset=cache_dataset,
                split=cache_split,
                root=root,
                image=image,
                objects=objects_by_image.get(image_id, []),
            )
    return None


def _list_official_cache_frame_samples(
    service: RealDatasetService,
    *,
    split: str,
    dataset: str | None,
    limit: int,
    offset: int,
    sequence_id: str | None = None,
) -> RealDatasetFrameSampleList | None:
    for cache_dataset, cache_split, root in _official_cache_roots(service, dataset, split):
        images, objects_by_image = _load_official_cache(str(root))
        if not images:
            continue
        grouped: dict[tuple[str, str], list[RealDatasetImage]] = defaultdict(list)
        classes: set[str] = set()
        for image in images:
            source_image_id = image.get("source_image_id")
            if not isinstance(source_image_id, str):
                continue
            contract = _cache_image_contract(
                dataset=cache_dataset,
                split=cache_split,
                root=root,
                image=image,
                objects=objects_by_image.get(source_image_id, []),
            )
            if contract is None or contract.sequence_id is None or contract.frame_sample_id is None:
                continue
            if sequence_id and contract.sequence_id != sequence_id:
                continue
            grouped[(contract.sequence_id, contract.frame_sample_id)].append(contract)
            classes.update(label.class_name for label in contract.labels)
        ordered_groups = sorted(grouped.items(), key=lambda item: item[0])
        page_groups = ordered_groups[offset : offset + limit]
        results: list[RealDatasetFrameSample] = []
        for (sequence_id, sample_id), cameras in page_groups:
            cameras.sort(key=lambda camera: (_CAMERA_ORDER.get(camera.camera_channel or "", 99), camera.camera_channel or ""))
            results.append(
                RealDatasetFrameSample(
                    id=f"{sequence_id}/{sample_id}",
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    split=cache_split,
                    dataset=cache_dataset,
                    camera_count=len(cameras),
                    label_count=sum(camera.label_count for camera in cameras),
                    cameras=cameras,
                )
            )
        return RealDatasetFrameSampleList(
            count=len(ordered_groups),
            image_count=sum(len(cameras) for cameras in grouped.values()),
            results=results,
            split=cache_split,
            dataset=cache_dataset,
            limit=limit,
            offset=offset,
            available_splits=[cache_split, split] if cache_split != split else [cache_split],
            available_datasets=["nuscenes", "kitti"],
            classes=sorted(classes),
        )
    return None


def _cached_image_content(service: RealDatasetService, split: str, image_id: str) -> tuple[Path, str] | None:
    for _cache_dataset, _cache_split, root in _official_cache_roots(service, None, split):
        images, _objects_by_image = _load_official_cache(str(root))
        for image in images:
            if image.get("source_image_id") != image_id:
                continue
            storage_filename = image.get("storage_filename")
            if not isinstance(storage_filename, str):
                continue
            path = _cache_storage_path(root, storage_filename)
            if path.is_file():
                return path, mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return None


def _bucket_and_key(image: QAImage) -> tuple[str, str]:
    settings = IngestionSettings()
    if image.storage_key:
        return settings.bucket_name, image.storage_key.lstrip("/")
    if not image.object_url:
        raise FileNotFoundError(f"Dataset image has no cloud object URL: {image.source_image_id}")
    if image.object_url.startswith("gs://"):
        bucket_and_key = image.object_url.removeprefix("gs://")
        bucket, _, key = bucket_and_key.partition("/")
        if bucket and key:
            return bucket, key
    parsed = urlparse(image.object_url)
    if parsed.netloc == "storage.googleapis.com":
        bucket, _, key = parsed.path.lstrip("/").partition("/")
        if bucket and key:
            return bucket, key
    raise FileNotFoundError(f"Dataset image does not have a supported GCS URL: {image.source_image_id}")


def _gcs_blob(image: QAImage):  # type: ignore[no-untyped-def]
    bucket_name, key = _bucket_and_key(image)
    client = _get_process_gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    return blob


def _download_gcs_image(image: QAImage) -> tuple[bytes, str]:
    blob = _gcs_blob(image)
    try:
        data = blob.download_as_bytes()
        return data, blob.content_type or "application/octet-stream"
    except NotFound as error:
        raise FileNotFoundError(f"Dataset image object does not exist in GCS: gs://{blob.bucket.name}/{blob.name}") from error


def _stream_gcs_image(image: QAImage) -> tuple[Iterator[bytes], str, dict[str, str]]:
    """Stream a private GCS object without buffering the full frame in RAM."""
    blob = _gcs_blob(image)
    blob.chunk_size = 1024 * 1024

    try:
        stream = blob.open("rb")
        first_chunk = stream.read(blob.chunk_size)
    except NotFound as error:
        raise FileNotFoundError(f"Dataset image object does not exist in GCS: gs://{blob.bucket.name}/{blob.name}") from error

    def chunks() -> Iterator[bytes]:
        with stream:
            if first_chunk:
                yield first_chunk
            while chunk := stream.read(blob.chunk_size):
                yield chunk

    headers = {"Cache-Control": "private, max-age=31536000, immutable"}
    if blob.etag:
        headers["ETag"] = blob.etag

    filename = getattr(image, "storage_key", None) or getattr(image, "filename", None) or ""
    content_type = blob.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return chunks(), content_type, headers


def _validated_storage_segment(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise FileNotFoundError(f"Invalid point-cloud {field}: {value}")
    return normalized


def _stream_gcs_pointcloud(
    dataset_id: str,
    dataset_version: str,
    split: str,
    sequence_id: str,
    sample_id: str,
) -> tuple[Iterator[bytes], dict[str, str]]:
    """Stream a point cloud belonging to the API service's configured release."""
    settings = IngestionSettings()
    safe_dataset = _validated_storage_segment(dataset_id, "dataset")
    safe_version = _validated_storage_segment(dataset_version, "release")
    safe_split = _validated_storage_segment(split, "split")
    safe_sequence = _validated_storage_segment(sequence_id, "sequence")
    safe_sample = _validated_storage_segment(sample_id, "sample")
    prefix = (
        f"datasets/official/{safe_dataset}/{safe_version}/{safe_split}/"
        f"pointclouds/{safe_sequence}/{safe_sample}"
    )
    client = create_gcs_storage_client(settings)
    bucket = client.bucket(settings.bucket_name)
    blob = next(
        (
            candidate
            for key in (f"{prefix}/LIDAR_TOP.pcd.bin", f"{prefix}/LIDAR_TOP.bin")
            if (candidate := bucket.blob(key)).exists(client=client)
        ),
        None,
    )
    if blob is None:
        raise FileNotFoundError(
            f"Dataset point cloud does not exist in {safe_dataset}/{safe_version}/{safe_split}: "
            f"{safe_sequence}/{safe_sample}"
        )
    blob.chunk_size = 1024 * 1024

    def chunks() -> Iterator[bytes]:
        with blob.open("rb") as stream:
            while chunk := stream.read(blob.chunk_size):
                yield chunk

    headers = {"Cache-Control": "private, max-age=31536000, immutable"}
    if blob.etag:
        headers["ETag"] = blob.etag
    return chunks(), headers


async def _db_image_to_contract(
    session: AsyncSession,
    image: QAImage,
    *,
    split: str,
    objects: Sequence[QAObject] | None = None,
) -> RealDatasetImage:
    identity = _frame_sample_identity(image.storage_key)
    if objects is None:
        objects = (
            await session.scalars(select(QAObject).where(QAObject.image_id == image.id).order_by(QAObject.id))
        ).all()
    labels = [
        RealDatasetLabel(
            id=str(item.id),
            class_name=item.label,
            bbox=RealDatasetBBox(x1=item.xmin, y1=item.ymin, x2=item.xmax, y2=item.ymax),
        )
        for item in objects
    ]
    image_contract = RealDatasetImage(
        id=image.source_image_id,
        split=split,
        dataset=_dataset_name(image),
        release=image.release,
        filename=image.storage_key or image.filename,
        width=image.width,
        height=image.height,
        label_count=len(labels),
        labels=labels,
        image_url=_content_url(split, image.source_image_id),
        frame_sample_id=identity[2] if identity else None,
        sequence_id=identity[1] if identity else None,
        camera_channel=identity[3] if identity else None,
    )
    return image_contract


async def _apply_latest_revisions(
    session: AsyncSession,
    service: RealDatasetService,
    images: list[RealDatasetImage],
    *,
    split: str,
    dataset: str | None = None,
) -> list[RealDatasetImage]:
    dataset_id, dataset_version = _dataset_release(service, dataset)
    revisions = await AnnotationEditorService.latest_for_images(
        session,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split=split,
        image_ids=[image.id for image in images],
    )
    return [
        AnnotationEditorService.to_document(
            image,
            dataset_id,
            dataset_version,
            revisions.get(image.id),
        ).image
        for image in images
    ]


async def _list_database_frame_samples(
    session: AsyncSession,
    service: RealDatasetService,
    *,
    split: str | None,
    dataset: str | None,
    limit: int,
    offset: int,
    sequence_id: str | None = None,
) -> RealDatasetFrameSampleList:
    selected_split = split or service.default_split
    base_conditions = list(_database_dataset_conditions(service, dataset))
    frame_conditions = [*base_conditions, _database_frame_split_condition(selected_split)]
    if sequence_id:
        frame_conditions.append(QAImage.storage_key.like(f"%/frames/{sequence_id}/%"))
    frame_group_key = func.regexp_replace(QAImage.storage_key, r"/[^/]+$", "")
    group_keys = (
        await session.scalars(
            select(frame_group_key)
            .where(*frame_conditions)
            .group_by(frame_group_key)
            .order_by(frame_group_key)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    rows = (
        (
            await session.scalars(
                select(QAImage)
                .where(*frame_conditions, frame_group_key.in_(group_keys))
                .order_by(QAImage.storage_key, QAImage.id)
            )
        ).all()
        if group_keys
        else []
    )
    # Fetch a small distinct sample of storage_key/filename/dataset rows to
    # compute available_splits and available_datasets. Using DISTINCT + LIMIT
    # avoids a full table scan over 1 000+ rows on a remote Supabase pooler.
    metadata_rows = (
        await session.execute(
            select(QAImage.storage_key, QAImage.filename, QAImage.dataset)
            .where(*_database_dataset_conditions(service, dataset))
            .distinct(QAImage.dataset, QAImage.release)
            .limit(50)
        )
    ).all()
    grouped: dict[tuple[str, str], list[tuple[QAImage, str]]] = {}
    available_splits = {
        _database_metadata_split(storage_key, filename, service.default_split)
        for storage_key, filename, _ in metadata_rows
    }
    available_datasets = await _available_database_datasets(session, service)
    for row in rows:
        identity = _frame_sample_identity(row.storage_key)
        if identity is None:
            continue
        row_split, sequence_id, sample_id, camera_channel = identity
        if row_split == selected_split:
            grouped.setdefault((sequence_id, sample_id), []).append((row, camera_channel))

    ordered_groups = sorted(grouped.items(), key=lambda item: item[0])
    page_groups = ordered_groups
    page_image_ids = [row.id for _, camera_rows in page_groups for row, _ in camera_rows]
    object_rows = (
        (
            await session.scalars(
                select(QAObject).where(QAObject.image_id.in_(page_image_ids)).order_by(QAObject.image_id, QAObject.id)
            )
        ).all()
        if page_image_ids
        else []
    )
    objects_by_image: dict[int, list[QAObject]] = defaultdict(list)
    for object_row in object_rows:
        objects_by_image[object_row.image_id].append(object_row)
    results: list[RealDatasetFrameSample] = []
    # Collect all cameras across all page groups, build frame results without
    # revisions first, then apply revisions in ONE batched query instead of
    # calling _apply_latest_revisions once per frame group (N+1 problem).
    all_cameras_ordered: list[tuple[tuple[str, str], list[tuple[QAImage, str]]]] = []
    for (sequence_id, sample_id), camera_rows in page_groups:
        camera_rows.sort(key=lambda item: (_CAMERA_ORDER.get(item[1], 99), item[1]))
        all_cameras_ordered.append(((sequence_id, sample_id), camera_rows))

    # Build flat camera list for single batched revision lookup
    flat_cameras: list[RealDatasetImage] = []
    for _, camera_rows in all_cameras_ordered:
        for row, _ in camera_rows:
            flat_cameras.append(
                await _db_image_to_contract(
                    session,
                    row,
                    split=selected_split,
                    objects=objects_by_image[row.id],
                )
            )

    # Single batch revision query for ALL cameras on this page
    flat_cameras = await _apply_latest_revisions(
        session,
        service,
        flat_cameras,
        split=selected_split,
        dataset=dataset,
    )

    # Re-group cameras back into frame samples
    camera_index = 0
    for (sequence_id, sample_id), camera_rows in all_cameras_ordered:
        n = len(camera_rows)
        cameras = flat_cameras[camera_index : camera_index + n]
        camera_index += n
        results.append(
            RealDatasetFrameSample(
                id=f"{sequence_id}/{sample_id}",
                sample_id=sample_id,
                sequence_id=sequence_id,
                split=selected_split,
                dataset=dataset,
                camera_count=len(cameras),
                label_count=sum(camera.label_count for camera in cameras),
                cameras=cameras,
            )
        )

    class_rows = (
        await session.scalars(
            select(QAObject.label)
            .join(QAImage, QAImage.id == QAObject.image_id)
            .where(*frame_conditions)
            .distinct()
            .order_by(QAObject.label)
        )
    ).all()
    group_count = int(
        await session.scalar(select(func.count(func.distinct(frame_group_key))).where(*frame_conditions)) or 0
    )
    image_count = int(await session.scalar(select(func.count()).select_from(QAImage).where(*frame_conditions)) or 0)
    return RealDatasetFrameSampleList(
        count=group_count,
        image_count=image_count,
        results=results,
        split=selected_split,
        dataset=dataset,
        limit=limit,
        offset=offset,
        available_splits=sorted(available_splits) or [selected_split],
        available_datasets=available_datasets,
        classes=list(class_rows),
    )



async def _list_filesystem_frame_samples(
    session: AsyncSession,
    service: RealDatasetService,
    *,
    split: str | None,
    limit: int,
    offset: int,
) -> RealDatasetFrameSampleList:
    image_list = service.list_images(split=split, limit=limit, offset=offset)
    images = await _apply_latest_revisions(
        session,
        service,
        image_list.results,
        split=image_list.split,
    )
    samples = [
        RealDatasetFrameSample(
            id=image.id,
            sample_id=image.id,
            sequence_id=image.split,
            split=image.split,
            camera_count=1,
            label_count=image.label_count,
            cameras=[image],
        )
        for image in images
    ]
    return RealDatasetFrameSampleList(
        count=image_list.count,
        image_count=image_list.count,
        results=samples,
        split=image_list.split,
        dataset=None,
        limit=image_list.limit,
        offset=image_list.offset,
        available_splits=image_list.available_splits,
        available_datasets=[],
        classes=image_list.classes,
    )


async def _list_database_images(
    session: AsyncSession,
    service: RealDatasetService,
    *,
    split: str | None,
    dataset: str | None,
    limit: int,
    offset: int,
    allowed_image_ids: set[str] | None = None,
) -> RealDatasetImageList:
    selected_split = split or service.default_split
    base_conditions = list(_database_dataset_conditions(service, dataset))
    split_conditions = [*base_conditions, _database_image_split_condition(selected_split)]
    if allowed_image_ids is not None:
        split_conditions.append(QAImage.source_image_id.in_(allowed_image_ids))
    metadata_rows = (
        await session.execute(
            select(QAImage.storage_key, QAImage.filename, QAImage.dataset)
            .where(*_database_dataset_conditions(service, dataset))
            .distinct(QAImage.dataset, QAImage.release)
            .limit(50)
        )
    ).all()
    available_splits = sorted(
        {
            _database_metadata_split(storage_key, filename, service.default_split)
            for storage_key, filename, _ in metadata_rows
        }
    ) or [selected_split]
    available_datasets = await _available_database_datasets(session, service)
    rows = (
        await session.scalars(
            select(QAImage)
            .where(*split_conditions)
            .order_by(QAImage.filename, QAImage.id)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    class_rows = (
        await session.scalars(
            select(QAObject.label)
            .join(QAImage, QAImage.id == QAObject.image_id)
            .where(*split_conditions)
            .distinct()
            .order_by(QAObject.label)
        )
    ).all()
    image_count = int(await session.scalar(select(func.count()).select_from(QAImage).where(*split_conditions)) or 0)
    images = [await _db_image_to_contract(session, row, split=selected_split) for row in rows]
    images = await _apply_latest_revisions(
        session,
        service,
        images,
        split=selected_split,
        dataset=dataset,
    )
    return RealDatasetImageList(
        count=image_count,
        results=images,
        split=selected_split,
        dataset=dataset,
        limit=limit,
        offset=offset,
        available_splits=available_splits,
        available_datasets=available_datasets,
        classes=list(class_rows),
    )


async def _get_database_image(
    session: AsyncSession,
    service: RealDatasetService,
    image_id: str,
    *,
    split: str,
) -> RealDatasetImage:
    image = await _get_database_image_row(session, service, image_id, split=split)
    base_image = await _db_image_to_contract(session, image, split=split)
    dataset_id, dataset_version = _image_dataset_release(service, base_image)
    return (
        await AnnotationEditorService.document(
            session,
            image=base_image,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
    ).image


async def _get_database_image_row(
    session: AsyncSession,
    service: RealDatasetService,
    image_id: str,
    *,
    split: str,
) -> QAImage:
    # Always use the canonical 'product' release for both datasets — this matches
    # _database_dataset_conditions() which also hardcodes release == 'product'.
    # Using service.dataset_version here would break when the Railway env has a
    # stale DATASET_VERSION (e.g. 'v1.0-mini') while all ingested images use
    # release='product'.
    supported_release = or_(
        and_(func.lower(QAImage.dataset) == "nuscenes", QAImage.release == service.dataset_version),
        and_(func.lower(QAImage.dataset) == "kitti", QAImage.release == service.dataset_version),
    )
    image = await session.scalar(
        select(QAImage).where(
            QAImage.source_image_id == image_id,
            supported_release,
        )
    )
    if image is None or _database_image_split(image, service.default_split) != split:
        raise FileNotFoundError(
            f"Dataset image does not exist in a supported release for split {split}: {image_id}"
        )
    return cast(QAImage, image)


@router.get("/images", response_model=RealDatasetImageList)
async def list_real_dataset_images(
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    split: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> RealDatasetImageList:
    try:
        allowed_image_ids: set[str] | None = None
        if current_user.role == "annotator":
            allowed_image_ids = set((await session.scalars(select(FrameTask.image_id).where(FrameTask.annotator_id == current_user.id))).all())
        if service.dataset_backend == "database":
            return await _list_database_images(
                session,
                service,
                split=split,
                dataset=dataset,
                limit=limit,
                offset=offset,
                allowed_image_ids=allowed_image_ids,
            )
        result = service.list_images(split=split, limit=limit, offset=offset)
        if allowed_image_ids is not None:
            result.results = [item for item in result.results if item.id in allowed_image_ids]
        result.results = await _apply_latest_revisions(
            session,
            service,
            result.results,
            split=result.split,
        )
        return result
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


@router.get("/frame-samples", response_model=RealDatasetFrameSampleList)
async def list_real_dataset_frame_samples(
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    split: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    sequence_id: str | None = Query(default=None),
    limit: int = Query(default=8, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> RealDatasetFrameSampleList:
    """List canonical frame samples, grouping all camera views into one review item."""
    try:
        if service.dataset_backend == "database":
            selected_split = _selected_split(split, dataset, service)
            last_db_result: RealDatasetFrameSampleList | None = None
            try:
                result = await asyncio.wait_for(
                    _list_database_frame_samples(
                        session,
                        service,
                        split=selected_split,
                        dataset=dataset,
                        limit=limit,
                        offset=offset,
                        sequence_id=sequence_id,
                    ),
                    timeout=_DATABASE_LIST_TIMEOUT_SECONDS,
                )
                if result.count > 0:
                    return result
                last_db_result = result
            except TimeoutError:
                logger.warning(
                    "frame-samples DB query timed out for split=%s dataset=%s",
                    selected_split,
                    dataset,
                )
                await session.rollback()
            except Exception:
                logger.exception(
                    "frame-samples DB query failed for split=%s dataset=%s",
                    selected_split,
                    dataset,
                )
                await session.rollback()
            cached = _list_official_cache_frame_samples(
                service,
                split=selected_split,
                dataset=dataset,
                limit=limit,
                offset=offset,
                sequence_id=sequence_id,
            )
            if cached is not None:
                return cached
            # Return cached empty DB result or a safe synthetic empty list so the
            # frontend always gets a valid 200 instead of a crash-induced 500.
            if last_db_result is not None:
                return last_db_result
            return RealDatasetFrameSampleList(
                count=0,
                image_count=0,
                results=[],
                split=selected_split,
                dataset=dataset,
                limit=limit,
                offset=offset,
                available_splits=[selected_split],
                available_datasets=["nuscenes", "kitti"],
                classes=[],
            )
        return await _list_filesystem_frame_samples(session, service, split=split, limit=limit, offset=offset)
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


@router.get("/images/{split}/{image_id}", response_model=RealDatasetImage)
async def get_real_dataset_image(
    split: str,
    image_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db_session),
) -> RealDatasetImage:
    try:
        await _ensure_task_access(session, image_id, current_user)
        if service.dataset_backend == "database":
            try:
                return await _get_database_image(session, service, image_id, split=split)
            except FileNotFoundError:
                cached = _cached_image_contract(service, split, image_id)
                if cached is not None:
                    return cached
                raise
        base_image = service.get_image(split, image_id)
        return (
            await AnnotationEditorService.document(
                session,
                image=base_image,
                dataset_id=service.dataset_id,
                dataset_version=service.dataset_version,
            )
        ).image
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


def _generate_thumbnail(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.thumbnail((240, 135), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="WebP")
        return out.getvalue()


async def _serve_generated_thumbnail(original_path: Path, thumb_path: Path) -> FileResponse:
    def _gen():
        with Image.open(original_path) as img:
            img.thumbnail((240, 135), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="WebP")
            return out.getvalue()

    thumb_bytes = await asyncio.to_thread(_gen)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = thumb_path.with_name(f"{thumb_path.name}.tmp-{uuid.uuid4()}")
    tmp_path.write_bytes(thumb_bytes)
    os.replace(tmp_path, thumb_path)
    return FileResponse(thumb_path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/images/{split}/{image_id}/content", response_class=Response, response_model=None)
async def get_real_dataset_image_content(
    split: str,
    image_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db_session),
    size: str | None = Query(default=None),
) -> FileResponse | StreamingResponse | Response:
    if size and size != "thumbnail":
        raise HTTPException(status_code=400, detail="Invalid size parameter")

    try:
        await _ensure_task_access(session, image_id, current_user)
        is_thumbnail = size == "thumbnail"

        if is_thumbnail:
            thumb_path = _gcs_cache_root() / "thumbnails" / f"{image_id}.webp"
            if thumb_path.exists():
                try:
                    os.utime(thumb_path, None)
                except OSError:
                    pass
                return FileResponse(thumb_path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000, immutable"})

        if service.dataset_backend == "database":
            cached = _cached_image_content(service, split, image_id)
            if cached is not None:
                path, media_type = cached
                if is_thumbnail:
                    return await _serve_generated_thumbnail(path, thumb_path)
                return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})
            image_row = await _get_database_image_row(session, service, image_id, split=split)

            if is_thumbnail:
                async with _cache_locks[f"thumb_{image_id}"]:
                    if thumb_path.exists():
                        try:
                            os.utime(thumb_path, None)
                        except OSError:
                            pass
                        return FileResponse(thumb_path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000, immutable"})
                    data, _ = await asyncio.to_thread(_download_gcs_image, image_row)
                    thumb_bytes = await asyncio.to_thread(_generate_thumbnail, data)
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = thumb_path.with_name(f"{thumb_path.name}.tmp-{uuid.uuid4()}")
                    tmp_path.write_bytes(thumb_bytes)
                    os.replace(tmp_path, thumb_path)
                background_tasks.add_task(_evict_cache_if_needed)
                return FileResponse(thumb_path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000, immutable"})

            import pathlib
            ext = pathlib.Path(image_row.filename).suffix if hasattr(image_row, "filename") and image_row.filename else ".jpg"
            full_path = _gcs_cache_root() / "original" / f"{image_id}{ext}"

            import mimetypes
            content_type = mimetypes.guess_type(image_row.filename)[0] if hasattr(image_row, "filename") and image_row.filename else "image/jpeg"

            async with _cache_locks[f"full_{image_id}"]:
                if full_path.exists():
                    try:
                        os.utime(full_path, None)
                    except OSError:
                        pass
                    return FileResponse(full_path, media_type=content_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})

                data, dl_content_type = await asyncio.to_thread(_download_gcs_image, image_row)
                content_type = dl_content_type or content_type

                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = full_path.with_name(f"{full_path.name}.tmp-{uuid.uuid4()}")
                    tmp_path.write_bytes(data)
                    os.replace(tmp_path, full_path)
                    background_tasks.add_task(_evict_cache_if_needed)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to cache full image to disk: {e}")

                return Response(content=data, media_type=content_type, headers={"Cache-Control": "private, max-age=31536000, immutable"})
        path = service.image_path(split, image_id)
        if is_thumbnail:
            return await _serve_generated_thumbnail(path, thumb_path)
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error
    return FileResponse(path)


@router.get(
    "/pointclouds/{dataset_path:path}/{split}/{sequence_id}/{sample_id}/content",
    response_class=Response,
    response_model=None,
)
async def get_real_dataset_pointcloud_content(
    dataset_path: str,
    split: str,
    sequence_id: str,
    sample_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    try:
        configured_path = f"{service.dataset_id}/{service.dataset_version}"
        if dataset_path.strip("/") != configured_path:
            raise FileNotFoundError(
                f"Dataset release is not served by this API: {dataset_path}"
            )
        if service.dataset_backend != "database":
            raise FileNotFoundError(
                "Local filesystem backend does not support point clouds."
            )
        chunks, headers = await asyncio.to_thread(
            _stream_gcs_pointcloud,
            service.dataset_id,
            service.dataset_version,
            split,
            sequence_id,
            sample_id,
        )
        return StreamingResponse(
            chunks,
            media_type="application/octet-stream",
            headers=headers,
        )
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


async def _prepare_evaluation_input(
    session: AsyncSession,
    service: RealDatasetService,
    split: str,
    image_id: str,
) -> _EvaluationInput:
    image_payload: bytes | None = None
    image_reference: InferenceImageReference | None = None
    if service.dataset_backend == "database":
        image_row = await _get_database_image_row(session, service, image_id, split=split)
        effective_image = await _get_database_image(session, service, image_id, split=split)
        if service.uses_remote_inference:
            bucket, object_key = _bucket_and_key(image_row)
            dataset_id, dataset_version = _image_dataset_release(service, effective_image)
            image_reference = InferenceImageReference(
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                split=split,
                image_id=image_id,
                bucket=bucket,
                object_key=object_key,
            )
        else:
            image_payload, _ = await asyncio.to_thread(_download_gcs_image, image_row)
    else:
        base_image = service.get_image(split, image_id)
        effective_image = (
            await AnnotationEditorService.document(
                session,
                image=base_image,
                dataset_id=service.dataset_id,
                dataset_version=service.dataset_version,
            )
        ).image

    dataset_id, dataset_version = _image_dataset_release(service, effective_image)
    latest = await AnnotationEditorService.latest_revision(
        session,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split=split,
        image_id=image_id,
    )
    return _EvaluationInput(
        image_id=image_id,
        image=effective_image,
        image_payload=image_payload,
        image_reference=image_reference,
        revision=latest.version if latest else 0,
    )


@router.post("/images/{split}/{image_id}/evaluate", response_model=RealDatasetEvaluation)
async def evaluate_real_dataset_image(
    split: str,
    image_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    _operator: Annotated[AuthenticatedUser, Depends(require_roles("reviewer", "admin"))],
    force: bool = Query(default=False),
    persist: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
) -> RealDatasetEvaluation:
    try:
        prepared = await _prepare_evaluation_input(session, service, split, image_id)
        evaluation = await service.evaluate(
            split,
            image_id,
            force=force,
            image_override=prepared.image,
            image_payload=prepared.image_payload,
            image_reference=prepared.image_reference,
            revision=prepared.revision,
        )
        if not persist:
            return evaluation
        case_ids = await RealDatasetQaService().persist(session, evaluation)
        evaluation.persisted = True
        evaluation.created_case_ids = case_ids
        return evaluation
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


@router.post("/images/{split}/evaluate-batch", response_model=RealDatasetBatchEvaluation)
async def evaluate_real_dataset_images_batch(
    split: str,
    request: RealDatasetBatchEvaluationRequest,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    _operator: Annotated[AuthenticatedUser, Depends(require_roles("reviewer", "admin"))],
    session: AsyncSession = Depends(get_db_session),
) -> RealDatasetBatchEvaluation:
    del background_tasks
    qa_service = RealDatasetQaService()
    prepared_inputs: list[_EvaluationInput] = []
    results: list[RealDatasetBatchEvaluationResult] = []
    for image_id in request.image_ids:
        try:
            prepared_inputs.append(await _prepare_evaluation_input(session, service, split, image_id))
        except HTTPException as error:
            results.append(RealDatasetBatchEvaluationResult(image_id=image_id, error=str(error.detail)))
        except (FileNotFoundError, YoloDatasetLayoutError) as error:
            results.append(RealDatasetBatchEvaluationResult(image_id=image_id, error=str(error)))

    inference_responses: dict[str, InferenceResponse] = {}
    inference_errors: dict[str, str] = {}
    inference_batch_used = False
    batch_client = (
        service.inference_client
        if service.inference_client is not None
        and hasattr(service.inference_client, "detect_batch")
        else None
    )
    remote_inputs = [item for item in prepared_inputs if item.image_reference is not None]
    if batch_client is not None and len(remote_inputs) == len(prepared_inputs):
        batch_size = 64
        for start in range(0, len(remote_inputs), batch_size):
            chunk = remote_inputs[start:start + batch_size]
            try:
                batch_response = await batch_client.detect_batch(
                    InferenceBatchRequest(
                        images=[
                            item.image_reference
                            for item in chunk
                            if item.image_reference is not None
                        ]
                    )
                )
            except Exception as error:
                for item in chunk:
                    inference_errors[item.image_id] = f"Batch inference failed: {error}"
                continue
            inference_batch_used = True
            for item in batch_response.results:
                if item.error:
                    inference_errors[item.image.image_id] = item.error
                    continue
                inference_responses[item.image.image_id] = InferenceResponse(
                    model_name=batch_response.model_name,
                    model_version=batch_response.model_version,
                    detections=item.detections,
                    raw_risk_score=item.raw_risk_score,
                    latency_ms=item.latency_ms,
                    metadata={
                        **item.metadata,
                        "batch_latency_ms": batch_response.latency_ms,
                        "batch_metadata": batch_response.metadata,
                    },
                )

    for item in prepared_inputs:
        if item.image_id in inference_errors:
            results.append(
                RealDatasetBatchEvaluationResult(
                    image_id=item.image_id,
                    error=inference_errors[item.image_id],
                )
            )
            continue
        try:
            evaluation = await service.evaluate(
                split,
                item.image_id,
                force=request.force,
                image_override=item.image,
                image_payload=item.image_payload,
                image_reference=item.image_reference,
                inference_response=inference_responses.get(item.image_id),
                revision=item.revision,
            )
            if request.persist:
                case_ids = await qa_service.persist(session, evaluation)
                evaluation.persisted = True
                evaluation.created_case_ids = case_ids
            results.append(
                RealDatasetBatchEvaluationResult(
                    image_id=item.image_id,
                    evaluation=evaluation,
                )
            )
        except Exception as error:
            results.append(
                RealDatasetBatchEvaluationResult(
                    image_id=item.image_id,
                    error=str(error),
                )
            )

    succeeded = sum(1 for result in results if result.evaluation is not None)
    failed = len(results) - succeeded
    return RealDatasetBatchEvaluation(
        count=len(results),
        succeeded=succeeded,
        failed=failed,
        inference_batch_used=inference_batch_used,
        results=results,
    )


async def _base_editor_image(
    session: AsyncSession,
    service: RealDatasetService,
    *,
    split: str,
    image_id: str,
) -> RealDatasetImage:
    if service.dataset_backend == "database":
        try:
            image = await _get_database_image_row(session, service, image_id, split=split)
            return await _db_image_to_contract(session, image, split=split)
        except FileNotFoundError:
            cached = _cached_image_contract(service, split, image_id)
            if cached is not None:
                return cached
            raise
    return service.get_image(split, image_id)


async def _ensure_task_access(session: AsyncSession, image_id: str, user: AuthenticatedUser) -> None:
    """Enforce task-level visibility for the editor without breaking legacy data."""
    if user.role == "admin":
        return
    tasks = list((await session.scalars(select(FrameTask).where(FrameTask.image_id == image_id))).all())
    if not tasks:
        if user.role == "annotator":
            raise HTTPException(status_code=404, detail="This frame is not assigned to you.")
        return
    if user.role == "annotator" and not any(task.annotator_id == user.id for task in tasks):
        raise HTTPException(status_code=404, detail="This frame is not assigned to you.")
    if user.role == "reviewer" and not any(task.reviewer_id == user.id for task in tasks):
        raise HTTPException(status_code=404, detail="This frame is not in your review batch.")


@router.get("/images/{split}/{image_id}/annotations", response_model=AnnotationDocument)
async def get_image_annotations(
    split: str,
    image_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db_session),
) -> AnnotationDocument:
    try:
        image = await _base_editor_image(session, service, split=split, image_id=image_id)
        await _ensure_task_access(session, image_id, current_user)
        dataset_id, dataset_version = _image_dataset_release(service, image)
        return await AnnotationEditorService.document(
            session,
            image=image,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


@router.put("/images/{split}/{image_id}/annotations", response_model=AnnotationDocument)
async def save_image_annotations(
    split: str,
    image_id: str,
    request: AnnotationSaveRequest,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        AuthenticatedUser,
        Depends(require_roles("annotator", "reviewer", "admin")),
    ],
    session: AsyncSession = Depends(get_db_session),
) -> AnnotationDocument:
    try:
        image = await _base_editor_image(session, service, split=split, image_id=image_id)
        await _ensure_task_access(session, image_id, current_user)
        dataset_id, dataset_version = _image_dataset_release(service, image)
        return await AnnotationEditorService.save(
            session,
            image=image,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            expected_revision=request.expected_revision,
            labels=request.labels,
            actor_id=current_user.id,
            change_note=request.change_note,
        )
    except AnnotationConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error


@router.get("/images/{split}/{image_id}/annotations/history", response_model=AnnotationRevisionList)
async def get_image_annotation_history(
    split: str,
    image_id: str,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db_session),
) -> AnnotationRevisionList:
    image = await _base_editor_image(session, service, split=split, image_id=image_id)
    await _ensure_task_access(session, image_id, current_user)
    dataset_id, dataset_version = _image_dataset_release(service, image)
    return await AnnotationEditorService.history(
        session,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        split=split,
        image_id=image_id,
    )


@router.post("/images/{split}/{image_id}/annotations/restore", response_model=AnnotationDocument)
async def restore_image_annotations(
    split: str,
    image_id: str,
    request: AnnotationRestoreRequest,
    service: Annotated[RealDatasetService, Depends(get_real_dataset_service)],
    background_tasks: BackgroundTasks,
    current_user: Annotated[
        AuthenticatedUser,
        Depends(require_roles("annotator", "reviewer", "admin")),
    ],
    session: AsyncSession = Depends(get_db_session),
) -> AnnotationDocument:
    try:
        image = await _base_editor_image(session, service, split=split, image_id=image_id)
        await _ensure_task_access(session, image_id, current_user)
        dataset_id, dataset_version = _image_dataset_release(service, image)
        return await AnnotationEditorService.restore(
            session,
            image=image,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            expected_revision=request.expected_revision,
            target_revision=request.target_revision,
            actor_id=current_user.id,
            change_note=request.change_note,
        )
    except AnnotationConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (FileNotFoundError, YoloDatasetLayoutError) as error:
        raise _not_found(error) from error
