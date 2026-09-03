from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.real_dataset import (
    _frame_sample_identity,
    _image_dataset_release,
    _official_cache_roots,
    _selected_split,
    _stream_gcs_image,
)
from src.config import Settings
from src.main import create_app
from src.models.inference_schemas import (
    InferenceDetection,
    InferenceImageReference,
    InferenceRequest,
    InferenceResponse,
)
from src.models.ingestion import QAImage, QAObject, QAReviewStatus
from src.models.real_dataset_schemas import RealDatasetBBox, RealDatasetImage
from src.services.real_dataset_service import RealDatasetService


class FakeAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.last_state: dict[str, Any] | None = None

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.last_state = state
        ground_truth = state["gt_labels"][0]
        return {
            "pred_labels": [{"class_name": "truck", "bbox": ground_truth["bbox"], "confidence": 0.9}],
            "matches": [{"gt_id": ground_truth["label_id"], "gt_class": ground_truth["class_name"], "pred_index": 0, "pred_class": "truck", "pred_confidence": 0.9, "iou": 1.0, "class_match": False}],
            "unmatched_gt": [],
            "unmatched_pred": [],
            "qa_report": {"image_path": state["image_path"], "status": "needs_review", "summary": "One deterministic issue.", "metrics": {}, "issues": [{"label_id": ground_truth["label_id"], "issue_type": "wrong_class", "severity": "high", "explanation": "Prediction differs.", "suggested_fix": "Review class.", "evidence": {}}]},
        }


class CloudImageAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.image_path: Path | None = None
        self.image_payload: bytes | None = None

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.image_path = Path(state["image_path"])
        self.image_payload = self.image_path.read_bytes()
        return await super().ainvoke(state)


class FakeInferenceClient:
    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    async def detect(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(
            model_name="remote-yolo",
            model_version="remote-yolo@2026-08-27",
            detections=[
                InferenceDetection(
                    class_name="car",
                    bbox=RealDatasetBBox(x1=10, y1=10, x2=30, y2=30),
                    confidence=0.88,
                )
            ],
            latency_ms={"inference": 12.5},
        )


@pytest.mark.parametrize(
    ("storage_key", "expected"),
    [
        (
            "datasets/official/nuscenes/v1.0-mini/smoke/frames/scene-0061/"
            "ca9a282c9e77460f8360f564131a8af5/CAM_FRONT.jpg",
            ("smoke", "scene-0061", "ca9a282c9e77460f8360f564131a8af5", "CAM_FRONT"),
        ),
        (
            "datasets/official/nuscenes/v1.0-mini/smoke/frames/scene-0061/"
            "sample-1532402927647951/CAM_BACK.jpg",
            ("smoke", "scene-0061", "sample-1532402927647951", "CAM_BACK"),
        ),
    ],
)
def test_frame_sample_identity_accepts_canonical_frame_ids(storage_key, expected):
    assert _frame_sample_identity(storage_key) == expected


def test_cloud_dataset_split_defaults_to_product(tmp_path):
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="product",
        default_split="product",
    )

    assert _selected_split(None, "nuscenes", service) == "product"
    assert _selected_split(None, "kitti", service) == "product"
    assert _selected_split("product", "kitti", service) == "product"


def test_kitti_cache_roots_return_product_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LABEL_GUARDIAN_GCS_CACHE_ROOT", str(tmp_path))
    service = RealDatasetService(tmp_path, dataset_backend="database")

    roots = _official_cache_roots(service, "kitti", "product")

    assert [(dataset, split) for dataset, split, _root in roots] == [
        ("kitti", "product"),
    ]


def test_image_dataset_release_prefers_image_identity(tmp_path):
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="v1.0-trainval",
    )
    image = RealDatasetImage.model_validate(
        {
            "id": "000001",
            "split": "full",
            "dataset": "KITTI",
            "release": "object",
            "filename": "000001.png",
            "width": 100,
            "height": 80,
            "labelCount": 0,
            "labels": [],
            "imageUrl": "/api/v1/dataset/images/full/000001/content",
        }
    )

    assert _image_dataset_release(service, image) == ("kitti", "object")


def test_gcs_content_is_read_in_chunks(monkeypatch):
    class FakeBlob:
        chunk_size = 0
        content_type = "image/jpeg"
        etag = "frame-etag"

        def open(self, mode):
            assert mode == "rb"
            return BytesIO(b"private-frame")

    monkeypatch.setattr("src.api.real_dataset._gcs_blob", lambda _image: FakeBlob())

    chunks, content_type, headers = _stream_gcs_image(object())  # type: ignore[arg-type]

    assert b"".join(chunks) == b"private-frame"
    assert content_type == "image/jpeg"
    assert headers["ETag"] == "frame-etag"


@pytest.mark.asyncio
async def test_database_pointcloud_is_scoped_to_configured_release(
    tmp_path,
    postgres_async_session_factory,
    postgres_test_database,
    monkeypatch,
):
    calls: list[tuple[str, str, str, str, str]] = []

    def fake_stream(dataset_id, dataset_version, split, sequence_id, sample_id):
        calls.append((dataset_id, dataset_version, split, sequence_id, sample_id))
        return iter([b"fake-pointcloud"]), {"Cache-Control": "private, max-age=300"}

    monkeypatch.setattr("src.api.real_dataset._stream_gcs_pointcloud", fake_stream)
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="v1.0-mini",
    )
    application = create_app(
        settings=Settings(
            app_env="test",
            auth_enabled=False,
            database_url=postgres_test_database.async_url,
            _env_file=None,
        ),
        db_session_factory=postgres_async_session_factory,
        real_dataset_service=service,
    )

    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/v1/dataset/pointclouds/nuscenes/v1.0-mini/smoke/"
                "scene-0061/sample-1/content"
            )
            foreign = await client.get(
                "/api/v1/dataset/pointclouds/nuscenes/v1.0-trainval/smoke/"
                "scene-0061/sample-1/content"
            )

    assert response.status_code == 200
    assert response.content == b"fake-pointcloud"
    assert foreign.status_code == 404
    assert calls == [
        ("nuscenes", "v1.0-mini", "smoke", "scene-0061", "sample-1")
    ]


@pytest.mark.asyncio
async def test_database_evaluation_uses_ephemeral_cloud_image(tmp_path):
    cloud_agent = CloudImageAgent()
    service = RealDatasetService(tmp_path, dataset_backend="database", agent_runner=cloud_agent)
    image = {
        "id": "cloud-image",
        "split": "smoke",
        "filename": "samples/CAM_FRONT/frame.jpg",
        "width": 100,
        "height": 80,
        "labelCount": 1,
        "labels": [
            {
                "id": "label-1",
                "className": "car",
                "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                "attributes": {},
            }
        ],
        "imageUrl": "/api/v1/dataset/images/smoke/cloud-image/content",
    }
    evaluation = await service.evaluate(
        "smoke",
        "cloud-image",
        image_override=RealDatasetImage.model_validate(image),
        image_payload=b"cloud-image-bytes",
    )

    assert cloud_agent.image_payload == b"cloud-image-bytes"
    assert cloud_agent.image_path is not None and not cloud_agent.image_path.exists()
    assert evaluation.report.image_path == image["imageUrl"]


@pytest.mark.asyncio
async def test_evaluation_scopes_identity_to_annotation_revision_and_supported_taxonomy(tmp_path):
    cloud_agent = CloudImageAgent()
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="v1.0-trainval",
        agent_runner=cloud_agent,
    )
    image = RealDatasetImage.model_validate(
        {
            "id": "cloud-image",
            "split": "smoke",
            "dataset": "kitti",
            "release": "object",
            "filename": "frame.jpg",
            "width": 100,
            "height": 80,
            "labelCount": 2,
            "labels": [
                {
                    "id": "car-label",
                    "className": "vehicle.car",
                    "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                },
                {
                    "id": "barrier-label",
                    "className": "movable_object.barrier",
                    "bbox": {"x1": 40, "y1": 10, "x2": 60, "y2": 30},
                },
            ],
            "imageUrl": "/api/v1/dataset/images/smoke/cloud-image/content",
        }
    )

    revision_zero = await service.evaluate(
        "smoke", "cloud-image", image_override=image, image_payload=b"image", revision=0
    )
    revision_one = await service.evaluate(
        "smoke", "cloud-image", image_override=image, image_payload=b"image", revision=1
    )

    assert revision_zero.evaluation_id != revision_one.evaluation_id
    assert revision_one.dataset_id == "kitti"
    assert revision_one.dataset_version == "object"
    assert cloud_agent.last_state is not None
    assert cloud_agent.last_state["gt_labels"] == [
        {
            "label_id": "car-label",
            "class_name": "car",
            "bbox": {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
        }
    ]
    assert cloud_agent.last_state["metadata"]["unsupported_ground_truth_count"] == 1


@pytest.mark.asyncio
async def test_remote_inference_uses_gcs_reference_and_injected_predictions(tmp_path):
    agent = FakeAgent()
    inference_client = FakeInferenceClient()
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="v1.0-mini",
        agent_runner=agent,
        inference_client=inference_client,
    )
    image = RealDatasetImage.model_validate(
        {
            "id": "cloud-image",
            "split": "smoke",
            "dataset": "nuscenes",
            "release": "v1.0-mini",
            "filename": "frame.jpg",
            "width": 100,
            "height": 80,
            "labelCount": 1,
            "labels": [
                {
                    "id": "car-label",
                    "className": "vehicle.car",
                    "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                }
            ],
            "imageUrl": "/api/v1/dataset/images/smoke/cloud-image/content",
        }
    )
    reference = InferenceImageReference(
        dataset_id="nuscenes",
        dataset_version="v1.0-mini",
        split="smoke",
        image_id="cloud-image",
        bucket="label-guardian",
        object_key="datasets/official/nuscenes/v1.0-mini/smoke/frames/scene/sample/CAM_FRONT.jpg",
    )

    evaluation = await service.evaluate(
        "smoke",
        "cloud-image",
        image_override=image,
        image_reference=reference,
    )

    assert service.uses_remote_inference is True
    assert inference_client.requests == [InferenceRequest(image=reference)]
    assert agent.last_state is not None
    assert agent.last_state["image_path"] == image.image_url
    assert agent.last_state["pred_labels"] == [
        {
            "class_name": "car",
            "bbox": {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
            "confidence": 0.88,
        }
    ]
    assert evaluation.model_name == "remote-yolo@2026-08-27"


@pytest.mark.asyncio
async def test_database_evaluate_remote_inference_does_not_download_image_bytes(
    tmp_path,
    postgres_async_session_factory,
    postgres_test_database,
    monkeypatch,
):
    async with postgres_async_session_factory() as session:
        image = QAImage(
            source_image_id="remote-image",
            filename="images/smoke/remote-image.jpg",
            width=100,
            height=80,
            dataset="nuscenes",
            release="v1.0-mini",
            storage_key="datasets/official/nuscenes/v1.0-mini/smoke/frames/scene/sample/CAM_FRONT.jpg",
        )
        session.add(image)
        await session.flush()
        session.add(
            QAObject(
                image_id=image.id,
                source_object_key="object-1",
                label="vehicle.car",
                xmin=10,
                ymin=10,
                xmax=30,
                ymax=30,
                review_status=QAReviewStatus.NEEDS_REVIEW,
            )
        )
        await session.commit()

    def fail_download(_image):
        raise AssertionError("App Service should not download image bytes when remote inference is enabled.")

    monkeypatch.setattr("src.api.real_dataset._download_gcs_image", fail_download)
    agent = FakeAgent()
    inference_client = FakeInferenceClient()
    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        dataset_id="nuscenes",
        dataset_version="v1.0-mini",
        default_split="smoke",
        agent_runner=agent,
        inference_client=inference_client,
    )
    application = create_app(
        settings=Settings(
            app_env="test",
            auth_enabled=False,
            database_url=postgres_test_database.async_url,
            _env_file=None,
        ),
        db_session_factory=postgres_async_session_factory,
        real_dataset_service=service,
    )

    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            response = await client.post("/api/v1/dataset/images/smoke/remote-image/evaluate")

    assert response.status_code == 200
    assert inference_client.requests[0].image.object_key == image.storage_key
    assert response.json()["modelName"] == "remote-yolo@2026-08-27"


@pytest.mark.asyncio
async def test_database_browse_is_scoped_to_configured_dataset_release_and_split(
    tmp_path,
    postgres_async_session_factory,
    postgres_test_database,
):
    async with postgres_async_session_factory() as session:
        session.add_all(
            [
                QAImage(
                    source_image_id="wanted-image",
                    filename="images/product/wanted.jpg",
                    width=100,
                    height=80,
                    dataset="nuscenes",
                    release="product",
                    storage_key="datasets/official/nuscenes/product/frames/scene-1/sample-1/CAM_FRONT.jpg",
                ),
                QAImage(
                    source_image_id="kitti-product",
                    filename="images/product/kitti.jpg",
                    width=100,
                    height=80,
                    dataset="kitti",
                    release="product",
                    storage_key="datasets/official/kitti/product/frames/sequence-default/000000/CAM_FRONT.png",
                ),
                QAImage(
                    source_image_id="wrong-release",
                    filename="images/product/wrong-release.jpg",
                    width=100,
                    height=80,
                    dataset="nuscenes",
                    release="v1.0-trainval",
                    storage_key="datasets/official/nuscenes/v1.0-trainval/product/frames/scene-2/sample-2/CAM_FRONT.jpg",
                ),
                QAImage(
                    source_image_id="wrong-split",
                    filename="images/train/wrong-split.jpg",
                    width=100,
                    height=80,
                    dataset="nuscenes",
                    release="product",
                    storage_key="datasets/official/nuscenes/product/train/frames/scene-3/sample-3/CAM_FRONT.jpg",
                ),
            ]
        )
        await session.commit()

    service = RealDatasetService(
        tmp_path,
        dataset_backend="database",
        default_split="product",
        dataset_id="nuscenes",
        dataset_version="product",
    )
    application = create_app(
        settings=Settings(
            app_env="test",
            auth_enabled=False,
            database_url=postgres_test_database.async_url,
            _env_file=None,
        ),
        db_session_factory=postgres_async_session_factory,
        real_dataset_service=service,
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            listed = await client.get("/api/v1/dataset/images?split=product")
            samples = await client.get("/api/v1/dataset/frame-samples?split=product")
            kitti_samples = await client.get("/api/v1/dataset/frame-samples?split=product&dataset=kitti")
            foreign = await client.get("/api/v1/dataset/images/product/wrong-release")
            wrong_split = await client.get("/api/v1/dataset/images/product/wrong-split")

    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()["results"]] == ["wanted-image"]
    assert listed.json()["availableDatasets"] == ["kitti", "nuscenes"]
    assert [camera["id"] for row in samples.json()["results"] for camera in row["cameras"]] == ["wanted-image"]
    assert samples.json()["availableDatasets"] == ["kitti", "nuscenes"]
    assert kitti_samples.status_code == 200
    assert [camera["id"] for row in kitti_samples.json()["results"] for camera in row["cameras"]] == ["kitti-product"]
    assert kitti_samples.json()["availableDatasets"] == ["kitti", "nuscenes"]
    assert foreign.status_code == 404
    assert wrong_split.status_code == 404


def _write_dataset(root: Path) -> None:
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "class.txt.txt").write_text("car\ntruck\n", encoding="utf-8")
    Image.new("RGB", (100, 80), (20, 30, 40)).save(root / "images" / "val" / "000001.png")
    (root / "labels" / "val" / "000001.txt").write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")


def _app(tmp_path, session_factory, database_url, agent=None):
    return create_app(
        settings=Settings(app_env="test", database_url=database_url.async_url, _env_file=None),
        db_session_factory=session_factory,
        real_dataset_service=RealDatasetService(tmp_path, agent_runner=agent or FakeAgent()),
    )


@pytest.mark.asyncio
async def test_dataset_browse_content_and_agent_evaluation(tmp_path, postgres_async_session_factory, postgres_test_database):
    _write_dataset(tmp_path)
    agent = FakeAgent()
    application = _app(tmp_path, postgres_async_session_factory, postgres_test_database, agent)
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            listed = await client.get("/api/v1/dataset/images?split=val")
            content = await client.get("/api/v1/dataset/images/val/000001/content")
            evaluated = await client.post("/api/v1/dataset/images/val/000001/evaluate")
            cached = await client.post("/api/v1/dataset/images/val/000001/evaluate")
    assert listed.status_code == 200
    assert listed.json()["results"][0]["labels"][0]["bbox"] == {"x1": 40.0, "y1": 30.0, "x2": 60.0, "y2": 50.0}
    assert content.status_code == 200
    assert evaluated.json()["report"]["issues"][0]["issueType"] == "wrong_class"
    assert cached.json()["cached"] is True
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_evaluation_persists_idempotent_qa_cases(tmp_path, postgres_async_session_factory, postgres_test_database):
    _write_dataset(tmp_path)
    application = _app(tmp_path, postgres_async_session_factory, postgres_test_database)
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            first = await client.post("/api/v1/dataset/images/val/000001/evaluate?persist=true")
            second = await client.post("/api/v1/dataset/images/val/000001/evaluate?persist=true")
            restored = await client.get("/api/v1/dataset/images/val/000001/evaluation")
            cases = await client.get("/api/v1/qa-cases")
    assert first.status_code == 200
    assert second.json()["createdCaseIds"] == first.json()["createdCaseIds"]
    assert restored.status_code == 200
    assert restored.json()["persisted"] is True
    assert restored.json()["predictions"][0]["className"] == "truck"
    assert cases.json()["results"][0]["sourceImageId"] == "000001"


@pytest.mark.asyncio
async def test_editor_save_conflict_history_and_restore(tmp_path, postgres_async_session_factory, postgres_test_database):
    _write_dataset(tmp_path)
    application = _app(tmp_path, postgres_async_session_factory, postgres_test_database)
    path = "/api/v1/dataset/images/val/000001/annotations"
    changed = [{"id": "label-1", "className": "truck", "trackId": "track-7", "bbox": {"x1": 35, "y1": 25, "x2": 65, "y2": 55}, "attributes": {"occluded": True}}]
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            original = await client.get(path)
            saved = await client.put(path, json={"expectedRevision": 0, "labels": changed, "actorId": "reviewer-1", "changeNote": "Fix class and box"})
            stale = await client.put(path, json={"expectedRevision": 0, "labels": changed})
            listed = await client.get("/api/v1/dataset/images?split=val")
            history = await client.get(f"{path}/history")
            restored = await client.post(f"{path}/restore", json={"expectedRevision": 1, "targetRevision": 0, "actorId": "reviewer-1"})
    assert original.json()["revision"] == 0
    assert saved.status_code == 200 and saved.json()["revision"] == 1
    assert saved.json()["labels"] == changed
    assert stale.status_code == 409
    assert listed.json()["results"][0]["labels"] == changed
    assert history.json()["results"][0]["changeNote"] == "Fix class and box"
    assert restored.json()["revision"] == 2
    assert restored.json()["labels"][0]["className"] == "car"


@pytest.mark.asyncio
async def test_dataset_rejects_path_traversal(client):
    response = await client.get("/api/v1/dataset/images/val/..%2F.env/content")
    assert response.status_code == 404
