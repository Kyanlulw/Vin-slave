from io import BytesIO
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.config import InferenceServiceSettings, IngestionSettings
from src.inference_app import create_inference_app


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeBox:
    class _XY:
        def tolist(self) -> list[float]:
            return [1.0, 2.0, 10.0, 12.0]

    xyxy = [_XY()]
    cls = [1]
    conf = [0.91]


class _FakeResult:
    names = {0: "person", 1: "car"}
    boxes = [_FakeBox()]
    speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}


class _FakeModel:
    names = {0: "person", 1: "car"}

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, images: Any, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append((images, kwargs))
        return [_FakeResult() for _ in images]


@pytest.mark.asyncio
async def test_detect_batch_runs_yolo_once_for_all_images(monkeypatch: pytest.MonkeyPatch) -> None:
    image_bytes = _jpeg_bytes()
    fake_model = _FakeModel()

    monkeypatch.setattr("src.inference_app.create_gcs_storage_client", lambda _settings: object())
    monkeypatch.setattr("src.inference_app.get_yolo_model_by_name", lambda _name: fake_model)
    monkeypatch.setattr(
        "src.inference_app._download_gcs_bytes",
        lambda **_kwargs: (image_bytes, "image/jpeg"),
    )

    application = create_inference_app(
        settings=InferenceServiceSettings(
            _env_file=None,
            inference_app_env="test",
            inference_model_name="fake.pt",
            inference_max_batch_size=8,
        ),
        gcs_settings=IngestionSettings(_env_file=None, gcs_bucket="bucket"),
    )
    request = {
        "images": [
            {
                "datasetId": "kitti",
                "datasetVersion": "product",
                "split": "product",
                "imageId": "000001",
                "bucket": "bucket",
                "objectKey": "datasets/official/kitti/product/frames/sequence/000001/CAM_FRONT.png",
            },
            {
                "datasetId": "kitti",
                "datasetVersion": "product",
                "split": "product",
                "imageId": "000002",
                "bucket": "bucket",
                "objectKey": "datasets/official/kitti/product/frames/sequence/000002/CAM_FRONT.png",
            },
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
        response = await client.post("/v1/detect-batch", json=request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["modelName"] == "fake.pt"
    assert payload["metadata"]["batch_size"] == 2
    assert [item["image"]["imageId"] for item in payload["results"]] == ["000001", "000002"]
    assert payload["results"][0]["detections"][0]["className"] == "car"
    assert len(fake_model.calls) == 1
    assert len(fake_model.calls[0][0]) == 2
    assert fake_model.calls[0][1]["classes"] == [0, 1]
