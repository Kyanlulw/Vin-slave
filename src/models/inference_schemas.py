"""Contracts shared by the App Service and the standalone Inference Service."""

from typing import Literal

from pydantic import Field, model_validator

from src.models.base_schemas import ApiModel
from src.models.real_dataset_schemas import RealDatasetBBox


class InferenceImageReference(ApiModel):
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    bucket: str | None = None
    content_type: str | None = None

    @model_validator(mode="after")
    def has_safe_object_key(self) -> "InferenceImageReference":
        normalized = self.object_key.strip().lstrip("/")
        parts = normalized.replace("\\", "/").split("/")
        if not normalized or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("object_key must be a normalized GCS object key")
        self.object_key = normalized
        return self


class InferenceRequest(ApiModel):
    image: InferenceImageReference
    mode: Literal["yolo"] = "yolo"


class InferenceBatchRequest(ApiModel):
    images: list[InferenceImageReference] = Field(min_length=1)
    mode: Literal["yolo"] = "yolo"


class InferenceDetection(ApiModel):
    class_name: str = Field(min_length=1)
    bbox: RealDatasetBBox
    confidence: float = Field(ge=0.0, le=1.0)


class InferenceResponse(ApiModel):
    model_name: str
    model_version: str
    detections: list[InferenceDetection]
    raw_risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class InferenceBatchItemResponse(ApiModel):
    image: InferenceImageReference
    detections: list[InferenceDetection]
    raw_risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    error: str | None = None


class InferenceBatchResponse(ApiModel):
    model_name: str
    model_version: str
    results: list[InferenceBatchItemResponse]
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
