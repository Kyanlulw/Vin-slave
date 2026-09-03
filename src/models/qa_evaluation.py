from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class QaEvaluation(Base):
    __tablename__ = "qa_evaluations"
    __table_args__ = (
        CheckConstraint("status IN ('pass', 'needs_review', 'error')", name="status_values"),
        UniqueConstraint(
            "dataset_id",
            "dataset_version",
            "split",
            "image_id",
            "model_name",
            name="uq_qa_evaluation_identity",
        ),
        Index("ix_qa_evaluations_dataset_split", "dataset_id", "split"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(128), index=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(64), index=True)
    image_id: Mapped[str] = mapped_column(String(255), index=True)
    model_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), index=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    predictions_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    matches_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    unmatched_ground_truth_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    unmatched_predictions_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
