"""API contracts for dataset browsing, QA and the built-in annotation editor."""

from datetime import datetime
from typing import Literal

from pydantic import Field, computed_field, model_validator

from src.models.agent_schemas import LabelQAReport
from src.models.base_schemas import ApiModel
from src.services.yolo import canonical_class_names, canonical_detection_class


class RealDatasetBBox(ApiModel):
    x1: float
    y1: float
    x2: float
    y2: float


class RealDatasetLabel(ApiModel):
    id: str
    class_name: str
    bbox: RealDatasetBBox
    track_id: str | None = None
    attributes: dict[str, bool | float | int | str] = Field(default_factory=dict)

    @computed_field
    @property
    def normalized_class_name(self) -> str | None:
        """Tên class theo taxonomy YOLO/COCO để hiển thị, ``None`` nếu YOLO không có class này.

        ``class_name`` gốc vẫn là nguồn sự thật cho annotation editor.
        """
        return canonical_detection_class(self.class_name)

    @model_validator(mode="after")
    def has_positive_area(self) -> "RealDatasetLabel":
        if self.bbox.x2 <= self.bbox.x1 or self.bbox.y2 <= self.bbox.y1:
            raise ValueError("annotation bounding box must have positive area")
        return self


class RealDatasetPrediction(ApiModel):
    id: str
    class_name: str
    bbox: RealDatasetBBox
    confidence: float

    @computed_field
    @property
    def normalized_class_name(self) -> str | None:
        return canonical_detection_class(self.class_name)


class RealDatasetMatch(ApiModel):
    ground_truth_id: str
    prediction_id: str
    ground_truth_class: str
    prediction_class: str
    iou: float
    class_match: bool


class RealDatasetImage(ApiModel):
    id: str
    split: str
    dataset: str | None = None
    release: str | None = None
    filename: str
    width: int
    height: int
    label_count: int
    labels: list[RealDatasetLabel]
    image_url: str
    frame_sample_id: str | None = None
    sequence_id: str | None = None
    camera_channel: str | None = None


class RealDatasetImageList(ApiModel):
    count: int
    results: list[RealDatasetImage]
    split: str
    dataset: str | None = None
    limit: int
    offset: int
    available_splits: list[str]
    available_datasets: list[str] = Field(default_factory=list)
    classes: list[str]

    @computed_field
    @property
    def normalized_classes(self) -> list[str]:
        return canonical_class_names(self.classes)


class RealDatasetFrameSample(ApiModel):
    id: str
    sample_id: str
    sequence_id: str
    split: str
    dataset: str | None = None
    camera_count: int
    label_count: int
    cameras: list[RealDatasetImage]


class RealDatasetFrameSampleList(ApiModel):
    count: int
    image_count: int
    results: list[RealDatasetFrameSample]
    split: str
    dataset: str | None = None
    limit: int
    offset: int
    available_splits: list[str]
    available_datasets: list[str] = Field(default_factory=list)
    classes: list[str]

    @computed_field
    @property
    def normalized_classes(self) -> list[str]:
        return canonical_class_names(self.classes)


class RealDatasetEvaluation(ApiModel):
    evaluation_id: str
    dataset_id: str
    dataset_version: str
    model_name: str
    image: RealDatasetImage
    report: LabelQAReport
    predictions: list[RealDatasetPrediction]
    matches: list[RealDatasetMatch]
    unmatched_ground_truth: list[dict]
    unmatched_predictions: list[dict]
    cached: bool
    persisted: bool = False
    created_case_ids: list[str] = Field(default_factory=list)
    inference_mode: Literal["yolo"] = "yolo"


class RealDatasetBatchEvaluationRequest(ApiModel):
    image_ids: list[str] = Field(min_length=1, max_length=100)
    force: bool = False
    persist: bool = True

    @model_validator(mode="after")
    def has_image_ids(self) -> "RealDatasetBatchEvaluationRequest":
        normalized = [image_id.strip() for image_id in self.image_ids if image_id.strip()]
        if not normalized:
            raise ValueError("image_ids must include at least one image id")
        self.image_ids = list(dict.fromkeys(normalized))
        return self


class RealDatasetBatchEvaluationResult(ApiModel):
    image_id: str
    evaluation: RealDatasetEvaluation | None = None
    error: str | None = None


class RealDatasetBatchEvaluation(ApiModel):
    count: int
    succeeded: int
    failed: int
    inference_batch_used: bool = False
    results: list[RealDatasetBatchEvaluationResult]


class RealDatasetError(ApiModel):
    detail: str = Field(min_length=1)


class AnnotationDocument(ApiModel):
    dataset_id: str
    dataset_version: str
    split: str
    image_id: str
    revision: int
    image: RealDatasetImage
    labels: list[RealDatasetLabel]
    original_labels: list[RealDatasetLabel]
    updated_at: datetime | None = None
    updated_by: str | None = None
    change_note: str | None = None


class AnnotationSaveRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    labels: list[RealDatasetLabel] = Field(max_length=5000)
    actor_id: str | None = Field(default=None, max_length=128)
    change_note: str | None = Field(default=None, max_length=2000)


class AnnotationRestoreRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    target_revision: int = Field(ge=0)
    actor_id: str | None = Field(default=None, max_length=128)
    change_note: str | None = Field(default=None, max_length=2000)


class AnnotationRevisionSummary(ApiModel):
    revision: int
    label_count: int
    actor_id: str | None = None
    change_note: str | None = None
    created_at: datetime | None = None


class AnnotationRevisionList(ApiModel):
    count: int
    results: list[AnnotationRevisionSummary]
