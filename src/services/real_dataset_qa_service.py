"""Persist real-dataset Agent evaluations as idempotent QA work items."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.agent_schemas import LabelQAReport
from src.models.audit_log import AuditLog
from src.models.qa_case import QaCase
from src.models.qa_evaluation import QaEvaluation
from src.models.real_dataset_schemas import (
    RealDatasetEvaluation,
    RealDatasetImage,
    RealDatasetMatch,
    RealDatasetPrediction,
)

RISK_BY_SEVERITY = {"high": 90, "medium": 65, "low": 35}
QUEUE_ERROR_TYPE = {
    "wrong_class": "wrong_class",
    "missing_label": "missing_object",
    "extra_or_wrong_label": "wrong_class",
    "bbox_misaligned": "box_misalignment",
    "loose_bbox": "box_misalignment",
    "duplicate_label": "duplicate_annotation",
}


class RealDatasetQaService:
    async def latest_evaluation(
        self,
        session: AsyncSession,
        *,
        dataset_id: str,
        dataset_version: str,
        split: str,
        image_id: str,
        image: RealDatasetImage,
    ) -> RealDatasetEvaluation | None:
        stored_evaluation = await session.scalar(
            select(QaEvaluation)
            .where(
                QaEvaluation.dataset_id == dataset_id,
                QaEvaluation.dataset_version == dataset_version,
                QaEvaluation.split == split,
                QaEvaluation.image_id == image_id,
            )
            .order_by(QaEvaluation.updated_at.desc(), QaEvaluation.created_at.desc())
            .limit(1)
        )
        if stored_evaluation is None:
            return None

        status = stored_evaluation.status
        if status not in {"pass", "needs_review", "error"}:
            status = "error"
        case_ids = list(
            (
                await session.scalars(
                    select(QaCase.id)
                    .where(QaCase.evaluation_id == stored_evaluation.id)
                    .order_by(QaCase.created_at)
                )
            ).all()
        )
        report = (
            LabelQAReport.model_validate(stored_evaluation.report_json)
            if stored_evaluation.report_json
            else LabelQAReport(
                image_path=image.image_url,
                status=cast(Literal["pass", "needs_review", "error"], status),
                summary="Loaded persisted YOLO evaluation from database.",
                metrics=stored_evaluation.metrics_json or {},
                issues=[],
            )
        )
        return RealDatasetEvaluation(
            evaluation_id=stored_evaluation.id,
            dataset_id=stored_evaluation.dataset_id,
            dataset_version=stored_evaluation.dataset_version,
            model_name=stored_evaluation.model_name,
            image=image,
            report=report,
            predictions=[
                RealDatasetPrediction.model_validate(item)
                for item in stored_evaluation.predictions_json or []
            ],
            matches=[
                RealDatasetMatch.model_validate(item)
                for item in stored_evaluation.matches_json or []
            ],
            unmatched_ground_truth=stored_evaluation.unmatched_ground_truth_json or [],
            unmatched_predictions=stored_evaluation.unmatched_predictions_json or [],
            cached=True,
            persisted=True,
            created_case_ids=case_ids,
        )

    async def persist(
        self,
        session: AsyncSession,
        evaluation: RealDatasetEvaluation,
    ) -> list[str]:
        now = datetime.now(UTC)
        stored_evaluation = await session.get(QaEvaluation, evaluation.evaluation_id)
        evaluation_values = {
            "dataset_id": evaluation.dataset_id,
            "dataset_version": evaluation.dataset_version,
            "split": evaluation.image.split,
            "image_id": evaluation.image.id,
            "model_name": evaluation.model_name,
            "status": evaluation.report.status,
            "metrics_json": evaluation.report.metrics,
            "report_json": evaluation.report.model_dump(mode="json", by_alias=True),
            "predictions_json": [item.model_dump(mode="json", by_alias=True) for item in evaluation.predictions],
            "matches_json": [item.model_dump(mode="json", by_alias=True) for item in evaluation.matches],
            "unmatched_ground_truth_json": evaluation.unmatched_ground_truth,
            "unmatched_predictions_json": evaluation.unmatched_predictions,
            "updated_at": now,
        }
        if stored_evaluation is None:
            stored_evaluation = QaEvaluation(
                id=evaluation.evaluation_id,
                created_at=now,
                **evaluation_values,
            )
            session.add(stored_evaluation)
        else:
            for key, value in evaluation_values.items():
                setattr(stored_evaluation, key, value)

        label_lookup = {label.id: label for label in evaluation.image.labels}
        prediction_evidence = [
            {
                "id": prediction.id,
                "trackId": prediction.id,
                "label": prediction.class_name,
                "bbox": [
                    prediction.bbox.x1,
                    prediction.bbox.y1,
                    prediction.bbox.x2 - prediction.bbox.x1,
                    prediction.bbox.y2 - prediction.bbox.y1,
                ],
                "confidence": prediction.confidence,
            }
            for prediction in evaluation.predictions
        ]
        ground_truth_evidence = [
            label.model_dump(mode="json", by_alias=True) for label in evaluation.image.labels
        ]
        case_ids: list[str] = []
        for issue_index, issue in enumerate(evaluation.report.issues):
            prediction_index = issue.evidence.get("prediction_index")
            issue_target = (
                issue.label_id
                if issue.label_id is not None
                else f"prediction-{prediction_index}"
                if prediction_index is not None
                else f"issue-{issue_index}"
            )
            identity = ":".join((evaluation.evaluation_id, issue.issue_type, issue_target))
            case_id = f"LGR-{sha256(identity.encode()).hexdigest()[:20]}"
            case_ids.append(case_id)
            stored_case = await session.get(QaCase, case_id)
            label = label_lookup.get(issue.label_id or "")
            class_name = label.class_name if label is not None else str(
                issue.evidence.get("class_name")
                or issue.evidence.get("gt_class")
                or issue.evidence.get("pred_class")
                or "unknown"
            )
            evidence = {
                "summary": issue.explanation or evaluation.report.summary,
                "issueEvidence": issue.evidence,
                "imageUrl": evaluation.image.image_url,
                "imageWidth": evaluation.image.width,
                "imageHeight": evaluation.image.height,
                "groundTruthLabels": ground_truth_evidence,
                "observedPredictions": prediction_evidence,
                "metrics": evaluation.report.metrics,
                "evaluationId": evaluation.evaluation_id,
            }
            values = {
                "dataset_id": evaluation.dataset_id,
                "dataset_version": evaluation.dataset_version,
                "source_split": evaluation.image.split,
                "source_image_id": evaluation.image.id,
                "evaluation_id": evaluation.evaluation_id,
                "sequence_id": evaluation.image.split,
                "frame_index": int(evaluation.image.id) if evaluation.image.id.isdigit() else 0,
                "frame_file_name": evaluation.image.filename,
                "class_name": class_name,
                "target_track_id": issue.label_id,
                "error_type": QUEUE_ERROR_TYPE[issue.issue_type],
                "risk_score": RISK_BY_SEVERITY[issue.severity],
                "priority": issue.severity,
                "evidence_json": evidence,
                "recommendation": issue.suggested_fix or "Review evidence and confirm the annotation.",
                "updated_at": now,
            }
            if stored_case is None:
                stored_case = QaCase(
                    id=case_id,
                    status="unreviewed",
                    assigned_to=None,
                    created_at=now,
                    **values,
                )
                session.add(stored_case)
                session.add(
                    AuditLog(
                        id=str(uuid4()),
                        case_id=case_id,
                        event_type="agent_case_created",
                        actor_type="agent",
                        actor_id=evaluation.model_name,
                        before_json=None,
                        after_json={"status": "unreviewed", "evaluationId": evaluation.evaluation_id},
                        metadata_json={"source": "local_dataset"},
                        created_at=now,
                    )
                )
            else:
                for key, value in values.items():
                    setattr(stored_case, key, value)
                if stored_case.status == "corrected":
                    stored_case.status = "unreviewed"
                    session.add(
                        AuditLog(
                            id=str(uuid4()),
                            case_id=case_id,
                            event_type="agent_case_reopened",
                            actor_type="agent",
                            actor_id=evaluation.model_name,
                            before_json={"status": "corrected"},
                            after_json={"status": "unreviewed", "evaluationId": evaluation.evaluation_id},
                            metadata_json={"source": "agent_reevaluation"},
                            created_at=now,
                        )
                    )

        current_case_ids = set(case_ids)
        previous_active_cases = (
            await session.scalars(
                select(QaCase)
                .join(QaEvaluation, QaCase.evaluation_id == QaEvaluation.id)
                .where(
                    QaEvaluation.dataset_id == evaluation.dataset_id,
                    QaEvaluation.dataset_version == evaluation.dataset_version,
                    QaEvaluation.split == evaluation.image.split,
                    QaEvaluation.image_id == evaluation.image.id,
                    QaEvaluation.model_name == evaluation.model_name,
                    QaCase.status.in_(("unreviewed", "in_review")),
                )
            )
        ).all()
        for stale_case in previous_active_cases:
            if stale_case.id in current_case_ids:
                continue
            before_status = stale_case.status
            stale_case.status = "corrected"
            stale_case.updated_at = now
            session.add(
                AuditLog(
                    id=str(uuid4()),
                    case_id=stale_case.id,
                    event_type="agent_case_resolved",
                    actor_type="agent",
                    actor_id=evaluation.model_name,
                    before_json={"status": before_status},
                    after_json={"status": "corrected", "evaluationId": evaluation.evaluation_id},
                    metadata_json={"source": "agent_reevaluation"},
                    created_at=now,
                )
            )

        await session.commit()
        return case_ids
