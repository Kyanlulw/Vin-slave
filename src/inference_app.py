"""Standalone GPU inference runtime.

This app owns the detector dependency stack and image loading for model
execution. The main FastAPI app should call it with a GCS object reference
instead of forwarding image bytes.
"""

from __future__ import annotations

import hmac
from io import BytesIO
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, status
from PIL import Image

from src.config import InferenceServiceSettings, IngestionSettings
from src.models.inference_schemas import (
    InferenceBatchItemResponse,
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceDetection,
    InferenceImageReference,
    InferenceRequest,
    InferenceResponse,
)
from src.models.real_dataset_schemas import RealDatasetBBox
from src.services.google_cloud import create_gcs_storage_client
from src.services.inference_client import INFERENCE_AUTH_HEADER
from src.services.yolo import TARGET_DETECTION_CLASSES, get_yolo_model_by_name, resolve_class_ids


def _settings() -> InferenceServiceSettings:
    return InferenceServiceSettings()


def _gcs_settings() -> IngestionSettings:
    return IngestionSettings()


def _authorize(
    authorization_token: Annotated[str | None, Header(alias=INFERENCE_AUTH_HEADER)] = None,
    settings: InferenceServiceSettings = Depends(_settings),
) -> None:
    expected = settings.inference_auth_token
    if expected is None:
        return
    supplied = authorization_token or ""
    if not hmac.compare_digest(supplied, expected.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid inference service token.",
        )


def _validated_object_key(key: str, settings: InferenceServiceSettings) -> str:
    normalized = key.strip().lstrip("/").replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="objectKey must be a normalized GCS object key.",
        )
    allowed_prefixes = settings.allowed_object_prefix_values
    if allowed_prefixes and not any(
        normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in allowed_prefixes
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The requested image object is outside the allowed inference prefixes.",
        )
    return normalized


def _download_gcs_bytes(
    *,
    bucket_name: str,
    object_key: str,
    settings: IngestionSettings,
    client: Any | None = None,
) -> tuple[bytes, str]:
    storage_client = client or create_gcs_storage_client(settings)
    blob = storage_client.bucket(bucket_name).blob(object_key)
    try:
        blob.reload(client=storage_client)
        return blob.download_as_bytes(client=storage_client), blob.content_type or "application/octet-stream"
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image object does not exist or cannot be read: gs://{bucket_name}/{object_key}",
        ) from error


def _resolve_model_name(
    *,
    settings: InferenceServiceSettings,
    storage_settings: IngestionSettings,
) -> str:
    configured_name = settings.inference_model_name
    if not configured_name.startswith("gs://"):
        return configured_name

    parsed = urlparse(configured_name)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")
    parts = PurePosixPath(object_key).parts
    if not bucket_name or not object_key or any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="INFERENCE_MODEL_NAME must be a valid gs://bucket/object path.",
        )

    cache_root = settings.inference_model_cache_dir.resolve()
    cache_path = (cache_root / bucket_name / Path(*parts)).resolve()
    if not cache_path.is_relative_to(cache_root):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="INFERENCE_MODEL_NAME resolves outside the model cache directory.",
        )
    if cache_path.is_file():
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    client = create_gcs_storage_client(storage_settings)
    blob = client.bucket(bucket_name).blob(object_key)
    try:
        blob.download_to_filename(str(cache_path), client=client)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact does not exist or cannot be read: {configured_name}",
        ) from error
    return str(cache_path)


def _prepare_yolo_runtime(
    *,
    settings: InferenceServiceSettings,
    storage_settings: IngestionSettings,
) -> tuple[Any, dict[str, Any], dict[str, Any], float]:
    metadata: dict[str, Any] = {}
    model_load_start = perf_counter()
    resolved_model_name = _resolve_model_name(settings=settings, storage_settings=storage_settings)
    model = get_yolo_model_by_name(resolved_model_name)
    model_load_ms = (perf_counter() - model_load_start) * 1000

    matched_ids, unmatched_names = resolve_class_ids(model, TARGET_DETECTION_CLASSES)
    predict_kwargs: dict[str, Any] = {
        "conf": settings.inference_confidence_threshold,
        "verbose": False,
    }
    if matched_ids:
        predict_kwargs["classes"] = matched_ids
    if unmatched_names:
        metadata["unmatched_target_classes"] = unmatched_names
    if resolved_model_name != settings.inference_model_name:
        metadata["resolved_model_path"] = resolved_model_name
    return model, predict_kwargs, metadata, model_load_ms


def _decode_rgb_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image bytes cannot be decoded: {error}",
        ) from error


def _result_latency(result: Any) -> dict[str, float | None]:
    speed = getattr(result, "speed", None) or {}
    return {
        "preprocess": round(float(speed["preprocess"]), 3) if "preprocess" in speed else None,
        "inference": round(float(speed["inference"]), 3) if "inference" in speed else None,
        "postprocess": round(float(speed["postprocess"]), 3) if "postprocess" in speed else None,
    }


def _detections_from_result(result: Any) -> list[InferenceDetection]:
    names = result.names
    detections: list[InferenceDetection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        detections.append(
            InferenceDetection(
                class_name=names[class_id],
                bbox=RealDatasetBBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(box.conf[0]),
            )
        )
    return detections


def _run_yolo(
    image_bytes: bytes,
    *,
    settings: InferenceServiceSettings,
    storage_settings: IngestionSettings,
) -> tuple[list[InferenceDetection], dict[str, float | None], dict[str, Any]]:
    model, predict_kwargs, metadata, model_load_ms = _prepare_yolo_runtime(
        settings=settings,
        storage_settings=storage_settings,
    )
    rgb_image = _decode_rgb_image(image_bytes)
    inference_start = perf_counter()
    results = model(rgb_image, **predict_kwargs)
    inference_wall_ms = (perf_counter() - inference_start) * 1000

    first_result = results[0]
    latency_ms = {
        "model_load": round(model_load_ms, 3),
        "inference_wall": round(inference_wall_ms, 3),
        **_result_latency(first_result),
    }
    return _detections_from_result(first_result), latency_ms, metadata


def _run_yolo_batch(
    image_payloads: list[bytes],
    *,
    settings: InferenceServiceSettings,
    storage_settings: IngestionSettings,
) -> tuple[
    list[tuple[list[InferenceDetection], dict[str, float | None]]],
    dict[str, float | None],
    dict[str, Any],
]:
    batch_start = perf_counter()
    model, predict_kwargs, metadata, model_load_ms = _prepare_yolo_runtime(
        settings=settings,
        storage_settings=storage_settings,
    )
    decode_start = perf_counter()
    rgb_images = [_decode_rgb_image(image_bytes) for image_bytes in image_payloads]
    decode_ms = (perf_counter() - decode_start) * 1000

    inference_start = perf_counter()
    results = list(model(rgb_images, **predict_kwargs))
    inference_wall_ms = (perf_counter() - inference_start) * 1000
    if len(results) != len(image_payloads):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YOLO returned a different number of results than requested images.",
        )
    item_results = [
        (_detections_from_result(result), _result_latency(result))
        for result in results
    ]
    latency_ms = {
        "model_load": round(model_load_ms, 3),
        "image_decode": round(decode_ms, 3),
        "inference_wall": round(inference_wall_ms, 3),
        "total_wall": round((perf_counter() - batch_start) * 1000, 3),
        "images": float(len(image_payloads)),
    }
    return item_results, latency_ms, metadata


def create_inference_app(
    *,
    settings: InferenceServiceSettings | None = None,
    gcs_settings: IngestionSettings | None = None,
) -> FastAPI:
    service_settings = settings or _settings()
    storage_settings = gcs_settings or _gcs_settings()

    def authorize_request(
        authorization_token: Annotated[str | None, Header(alias=INFERENCE_AUTH_HEADER)] = None,
    ) -> None:
        expected = service_settings.inference_auth_token
        if expected is None:
            return
        supplied = authorization_token or ""
        if not hmac.compare_digest(supplied, expected.get_secret_value()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid inference service token.",
            )

    application = FastAPI(
        title=service_settings.inference_app_name,
        version=service_settings.inference_app_version,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "label-guardian-inference",
            "environment": service_settings.inference_app_env,
            "version": service_settings.inference_app_version,
        }

    @application.get("/ready")
    async def readiness() -> dict[str, str]:
        _ = storage_settings.bucket_name
        create_gcs_storage_client(storage_settings)
        return {
            "status": "ok",
            "service": "label-guardian-inference",
            "environment": service_settings.inference_app_env,
            "version": service_settings.inference_app_version,
        }

    @application.post(
        "/v1/detect",
        response_model=InferenceResponse,
        dependencies=[Depends(authorize_request)],
    )
    async def detect(request: InferenceRequest) -> InferenceResponse:
        requested_bucket = request.image.bucket or storage_settings.bucket_name
        if requested_bucket != storage_settings.bucket_name:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The requested bucket is not served by this inference service.",
            )
        object_key = _validated_object_key(request.image.object_key, service_settings)
        image_bytes, content_type = _download_gcs_bytes(
            bucket_name=requested_bucket,
            object_key=object_key,
            settings=storage_settings,
        )
        detections, latency_ms, metadata = _run_yolo(
            image_bytes,
            settings=service_settings,
            storage_settings=storage_settings,
        )
        return InferenceResponse(
            model_name=service_settings.inference_model_name,
            model_version=service_settings.inference_model_version or service_settings.inference_model_name,
            detections=detections,
            latency_ms=latency_ms,
            metadata={
                **metadata,
                "content_type": content_type,
                "bucket": requested_bucket,
                "object_key": object_key,
                "dataset_id": request.image.dataset_id,
                "dataset_version": request.image.dataset_version,
                "split": request.image.split,
                "image_id": request.image.image_id,
            },
        )

    @application.post(
        "/v1/detect-batch",
        response_model=InferenceBatchResponse,
        dependencies=[Depends(authorize_request)],
    )
    async def detect_batch(request: InferenceBatchRequest) -> InferenceBatchResponse:
        if len(request.images) > service_settings.inference_max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "Batch size exceeds inference_max_batch_size="
                    f"{service_settings.inference_max_batch_size}."
                ),
            )

        storage_client = create_gcs_storage_client(storage_settings)
        payloads: list[tuple[InferenceImageReference, str, str, str, bytes]] = []
        download_start = perf_counter()
        for image in request.images:
            requested_bucket = image.bucket or storage_settings.bucket_name
            if requested_bucket != storage_settings.bucket_name:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="The requested bucket is not served by this inference service.",
                )
            object_key = _validated_object_key(image.object_key, service_settings)
            image_bytes, content_type = _download_gcs_bytes(
                bucket_name=requested_bucket,
                object_key=object_key,
                settings=storage_settings,
                client=storage_client,
            )
            payloads.append((image, requested_bucket, object_key, content_type, image_bytes))
        download_ms = (perf_counter() - download_start) * 1000

        image_payloads = [
            image_bytes
            for (_image, _bucket, _object_key, _content_type, image_bytes) in payloads
        ]
        yolo_results, latency_ms, metadata = _run_yolo_batch(
            image_payloads,
            settings=service_settings,
            storage_settings=storage_settings,
        )
        batch_latency_ms = {
            **latency_ms,
            "download": round(download_ms, 3),
        }
        results = [
            InferenceBatchItemResponse(
                image=image,
                detections=detections,
                latency_ms=item_latency,
                metadata={
                    "content_type": content_type,
                    "bucket": requested_bucket,
                    "object_key": object_key,
                    "dataset_id": image.dataset_id,
                    "dataset_version": image.dataset_version,
                    "split": image.split,
                    "image_id": image.image_id,
                },
            )
            for (
                image,
                requested_bucket,
                object_key,
                content_type,
                _image_bytes,
            ), (detections, item_latency) in zip(payloads, yolo_results, strict=True)
        ]
        return InferenceBatchResponse(
            model_name=service_settings.inference_model_name,
            model_version=service_settings.inference_model_version or service_settings.inference_model_name,
            results=results,
            latency_ms=batch_latency_ms,
            metadata={
                **metadata,
                "batch_size": len(results),
            },
        )

    return application


app = create_inference_app()
