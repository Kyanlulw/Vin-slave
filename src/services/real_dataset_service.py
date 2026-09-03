"""Real-dataset browser and Label QA Agent orchestration."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

from src.agents.graph import agent
from src.models.agent_schemas import LabelQAReport
from src.models.inference_schemas import InferenceImageReference, InferenceRequest, InferenceResponse
from src.models.real_dataset_schemas import (
    RealDatasetBBox,
    RealDatasetEvaluation,
    RealDatasetImage,
    RealDatasetImageList,
    RealDatasetLabel,
    RealDatasetMatch,
    RealDatasetPrediction,
)
from src.services.inference_client import InferenceClient, InferenceClientError
from src.services.ingestion.yolo_detection_adapter import (
    YoloDatasetLayoutError,
    YoloDetectionAdapter,
    available_yolo_splits,
)
from src.services.yolo import canonical_detection_class


class LabelQAAgentRunner(Protocol):
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]: ...


class RealDatasetService:
    """Expose one YOLO dataset safely and evaluate frames with the QA Agent."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_backend: str = "filesystem",
        default_split: str = "val",
        dataset_id: str = "local-yolo",
        dataset_version: str = "workspace",
        model_name: str = "yolo26n.pt",
        evaluation_cache_entries: int = 128,
        agent_runner: LabelQAAgentRunner = agent,
        inference_client: InferenceClient | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.dataset_backend = dataset_backend
        self.default_split = default_split
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.model_name = model_name
        self.evaluation_cache_entries = evaluation_cache_entries
        self.adapter = YoloDetectionAdapter(self.root)
        self.agent_runner = agent_runner
        self.inference_client = inference_client
        self._evaluation_cache: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
        self._inference_lock = asyncio.Lock()

    @property
    def uses_remote_inference(self) -> bool:
        return self.inference_client is not None

    def _cached_evaluation(self, key: tuple[str, ...]) -> dict[str, Any] | None:
        result = self._evaluation_cache.get(key)
        if result is not None:
            self._evaluation_cache.move_to_end(key)
        return result

    def _store_evaluation(self, key: tuple[str, ...], result: dict[str, Any]) -> None:
        self._evaluation_cache[key] = result
        self._evaluation_cache.move_to_end(key)
        while len(self._evaluation_cache) > self.evaluation_cache_entries:
            self._evaluation_cache.popitem(last=False)

    def available_splits(self) -> list[str]:
        splits = available_yolo_splits(self.root)
        if not splits:
            raise YoloDatasetLayoutError(f"No YOLO dataset is available at {self.root}")
        return [split or "root" for split in splits]

    def list_images(self, *, split: str | None, limit: int, offset: int) -> RealDatasetImageList:
        selected_split = split or self.default_split
        image_ids = self.adapter.image_ids(selected_split)
        results = [self.get_image(selected_split, image_id) for image_id in image_ids[offset : offset + limit]]
        return RealDatasetImageList(
            count=len(image_ids),
            results=results,
            split=selected_split,
            limit=limit,
            offset=offset,
            available_splits=self.available_splits(),
            classes=self.adapter.class_names(),
        )

    def get_image(self, split: str, image_id: str) -> RealDatasetImage:
        metadata, objects = self.adapter.load_image(split, image_id)
        labels = [
            RealDatasetLabel(
                id=item.provenance[0].source_annotation_id,
                class_name=item.label,
                bbox=RealDatasetBBox(
                    x1=item.bbox.xmin,
                    y1=item.bbox.ymin,
                    x2=item.bbox.xmax,
                    y2=item.bbox.ymax,
                ),
            )
            for item in objects
        ]
        return RealDatasetImage(
            id=image_id,
            split=split,
            filename=metadata.filename,
            width=metadata.width,
            height=metadata.height,
            label_count=len(labels),
            labels=labels,
            image_url=f"/api/v1/dataset/images/{split}/{image_id}/content",
        )

    def image_path(self, split: str, image_id: str) -> Path:
        path = self.adapter.image_path(split, image_id).resolve()
        if not path.is_relative_to(self.root):
            raise FileNotFoundError("Dataset image resolves outside the configured root")
        return cast(Path, path)

    async def evaluate(
        self,
        split: str,
        image_id: str,
        *,
        force: bool = False,
        image_override: RealDatasetImage | None = None,
        image_payload: bytes | None = None,
        image_reference: InferenceImageReference | None = None,
        inference_response: InferenceResponse | None = None,
        revision: int = 0,
    ) -> RealDatasetEvaluation:
        image = image_override or self.get_image(split, image_id)
        dataset_id = image.dataset or self.dataset_id
        dataset_version = image.release or self.dataset_version
        cache_key = (dataset_id, dataset_version, split, image_id, f"revision:{revision}")
        cached_result = None if force else self._cached_evaluation(cache_key)
        if cached_result is not None:
            return self._to_evaluation(image, cached_result, revision=revision, cached=True)

        supported_ground_truth = [
            (label, canonical)
            for label in image.labels
            if (canonical := canonical_detection_class(label.class_name)) is not None
        ]
        ground_truth = [
            {
                "label_id": label.id,
                "class_name": canonical,
                "bbox": label.bbox.model_dump(),
            }
            for label, canonical in supported_ground_truth
        ]

        async def invoke(image_path: Path | str, pred_labels: list[dict] | None = None) -> dict[str, Any]:
            state: dict[str, Any] = {
                "image_path": str(image_path),
                "gt_labels": ground_truth,
                "metadata": {
                    "dataset_split": split,
                    "dataset_image_id": image_id,
                    "unsupported_ground_truth_count": len(image.labels) - len(supported_ground_truth),
                },
            }
            if pred_labels is not None:
                state["pred_labels"] = pred_labels
                state["enable_rtdetr"] = False
            return await asyncio.to_thread(asyncio.run, self.agent_runner.ainvoke(state))

        async def invoke_inference_response(response: InferenceResponse) -> dict[str, Any]:
            pred_labels = [
                detection.model_dump(mode="json", by_alias=False)
                for detection in response.detections
            ]
            result = await invoke(image.image_url, pred_labels=pred_labels)
            result = dict(result)
            metadata = dict(result.get("metadata") or {})
            metadata["inference_mode"] = "remote"
            metadata["inference_model_name"] = response.model_name
            metadata["inference_model_version"] = response.model_version
            metadata["inference_latency_ms"] = response.latency_ms
            metadata["inference_metadata"] = response.metadata
            result["metadata"] = metadata
            report = dict(result.get("qa_report", {}))
            report["image_path"] = image.image_url
            result["qa_report"] = report
            return result

        async def invoke_remote(reference: InferenceImageReference) -> dict[str, Any]:
            if self.inference_client is None:
                raise InferenceClientError("Remote inference client is not configured.")
            try:
                response = await self.inference_client.detect(InferenceRequest(image=reference))
            except InferenceClientError as error:
                return {
                    "pred_labels": [],
                    "matches": [],
                    "unmatched_gt": [],
                    "unmatched_pred": [],
                    "qa_report": {
                        "image_path": image.image_url,
                        "status": "error",
                        "summary": str(error),
                        "metrics": {},
                        "issues": [],
                    },
                    "metadata": {
                        "inference_mode": "remote",
                        "inference_error": str(error),
                    },
                }
            return await invoke_inference_response(response)

        async with self._inference_lock:
            cached_result = None if force else self._cached_evaluation(cache_key)
            if cached_result is not None:
                return self._to_evaluation(image, cached_result, revision=revision, cached=True)
            if inference_response is not None:
                result = await invoke_inference_response(inference_response)
            elif self.inference_client is not None:
                if image_reference is None:
                    raise FileNotFoundError("Remote inference requires a GCS image reference.")
                result = await invoke_remote(image_reference)
            elif image_payload is None:
                result = await invoke(self.image_path(split, image_id))
            else:
                suffix = Path(image.filename).suffix.lower()
                if suffix not in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
                    suffix = ".jpg"
                with TemporaryDirectory(prefix="label-guardian-agent-") as temporary_directory:
                    temporary_image = Path(temporary_directory) / f"frame{suffix}"
                    temporary_image.write_bytes(image_payload)
                    result = await invoke(temporary_image)
                result = dict(result)
                report = dict(result.get("qa_report", {}))
                report["image_path"] = image.image_url
                result["qa_report"] = report
        self._store_evaluation(cache_key, result)
        return self._to_evaluation(image, result, revision=revision, cached=False)

    def _to_evaluation(
        self,
        image: RealDatasetImage,
        result: dict[str, Any],
        *,
        revision: int,
        cached: bool,
    ) -> RealDatasetEvaluation:
        dataset_id = image.dataset or self.dataset_id
        dataset_version = image.release or self.dataset_version
        metadata = result.get("metadata") or {}
        model_name = str(metadata.get("inference_model_version") or self.model_name)
        evaluation_key = ":".join(
            (
                dataset_id,
                dataset_version,
                image.split,
                image.id,
                model_name,
                f"revision:{revision}",
            )
        )
        predictions = [
            RealDatasetPrediction(
                id=f"pred-{index}",
                class_name=item["class_name"],
                bbox=item["bbox"],
                confidence=item["confidence"],
            )
            for index, item in enumerate(result.get("pred_labels", []))
        ]
        matches = [
            RealDatasetMatch(
                ground_truth_id=item["gt_id"],
                prediction_id=f"pred-{item['pred_index']}",
                ground_truth_class=item["gt_class"],
                prediction_class=item["pred_class"],
                iou=item["iou"],
                class_match=item["class_match"],
            )
            for item in result.get("matches", [])
        ]
        return RealDatasetEvaluation(
            evaluation_id=f"eval-{sha256(evaluation_key.encode()).hexdigest()[:24]}",
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_name=model_name,
            image=image,
            report=LabelQAReport.model_validate(result["qa_report"]),
            predictions=predictions,
            matches=matches,
            unmatched_ground_truth=result.get("unmatched_gt", []),
            unmatched_predictions=result.get("unmatched_pred", []),
            cached=cached,
        )
