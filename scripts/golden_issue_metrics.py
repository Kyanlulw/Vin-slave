"""Score QA flags against golden issues, independently of detector box metrics.

Every emitted flag is scored, including nonblocking flags and ``loose_bbox``.
Dataset eligibility and unsupported-category filtering belong to the caller.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import isfinite
from typing import Any

from src.agents.geometry import iou
from src.services.yolo import canonical_detection_class

ISSUE_TYPES = (
    "wrong_class",
    "missing_label",
    "extra_or_wrong_label",
    "bbox_misaligned",
    "duplicate_label",
    "loose_bbox",
)


def _class_name(value: str) -> str:
    return canonical_detection_class(value) or value.strip().lower()


def _matches_issue(
    expected: dict[str, Any],
    predicted: dict[str, Any],
    gold_labels: dict[str, dict[str, Any]],
    missing_iou: float,
) -> bool:
    kind = expected["issue_type"]
    if predicted.get("issue_type") != kind:
        return False
    evidence = predicted.get("evidence") or {}
    if kind == "missing_label":
        gold = gold_labels[expected["gold_label_id"]]
        predicted_class = evidence.get("class_name")
        bbox = evidence.get("bbox")
        return bool(
            isinstance(predicted_class, str)
            and isinstance(bbox, dict)
            and _class_name(predicted_class) == _class_name(gold["class_name"])
            and iou(bbox, gold["bbox"]) >= missing_iou
        )
    if kind == "duplicate_label":
        return {evidence.get("label_a"), evidence.get("label_b")} == {
            expected["source_label_id"],
            expected["duplicate_label_id"],
        }
    if kind in {"wrong_class", "extra_or_wrong_label", "bbox_misaligned"}:
        return predicted.get("label_id") == expected["source_label_id"]
    return False


def score_sample(
    sample: dict[str, Any],
    gold_document: dict[str, Any],
    flags: list[dict[str, Any]],
    missing_iou: float = 0.5,
) -> dict[str, Any]:
    """Match issue type and object target with one-to-one maximum cardinality.

    Missing-label flags have no source ID, so their evidence must match the gold
    class after canonicalization and the gold box at ``missing_iou`` or higher.
    Duplicate flags must identify the exact unordered pair of source label IDs.
    Severity and blocking status do not change whether an issue is correct.
    """
    if not isfinite(missing_iou) or not 0 <= missing_iou <= 1:
        raise ValueError("missing_iou must be finite and between 0 and 1")
    expected = sample["gold_issues"]
    gold_labels = {label["label_id"]: label for label in gold_document["labels"]}
    for issue in expected:
        if issue["issue_type"] == "missing_label" and issue["gold_label_id"] not in gold_labels:
            raise ValueError(f"Missing gold label for issue {issue.get('issue_id', issue)!r}")

    candidates = [
        [pred_index for pred_index, flag in enumerate(flags) if _matches_issue(issue, flag, gold_labels, missing_iou)]
        for issue in expected
    ]
    pred_to_gold: dict[int, int] = {}

    def augment(gold_index: int, visited: set[int]) -> bool:
        for pred_index in candidates[gold_index]:
            if pred_index in visited:
                continue
            visited.add(pred_index)
            previous = pred_to_gold.get(pred_index)
            if previous is None or augment(previous, visited):
                pred_to_gold[pred_index] = gold_index
                return True
        return False

    for gold_index in range(len(expected)):
        augment(gold_index, set())

    matched_gold = set(pred_to_gold.values())
    matched = [
        {
            "pred_index": pred_index,
            "gold_index": gold_index,
            "issue_type": expected[gold_index]["issue_type"],
        }
        for pred_index, gold_index in sorted(pred_to_gold.items())
    ]
    tp = len(matched)
    fp = len(flags) - tp
    fn = len(expected) - tp
    # Blocking-only view: advisory (blocking=False) flags do not force review.
    # A gold issue only counts as found here when matched by a blocking flag.
    matched_pred_indices = {m["pred_index"] for m in matched}
    blocking_pred_indices = {i for i, flag in enumerate(flags) if flag.get("blocking", True)}
    tp_blocking = len(matched_pred_indices & blocking_pred_indices)
    fp_blocking = len(blocking_pred_indices - matched_pred_indices)
    fn_blocking = len(expected) - tp_blocking
    clean = not expected
    return {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "primary_issue_type": sample["primary_issue_type"],
        "use_for_metric": sample["use_for_metric"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_blocking": tp_blocking,
        "fp_blocking": fp_blocking,
        "fn_blocking": fn_blocking,
        "exact": fp == 0 and fn == 0,
        "clean": clean,
        "clean_pass": clean and not flags,
        "matched_issues": matched,
        "unmatched_prediction_indices": [index for index in range(len(flags)) if index not in pred_to_gold],
        "unmatched_gold_indices": [index for index in range(len(expected)) if index not in matched_gold],
        "predicted_issue_counts": dict(Counter(flag["issue_type"] for flag in flags)),
        "expected_issue_counts": dict(Counter(issue["issue_type"] for issue in expected)),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _issue_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "expected_issues": tp + fn,
        "predicted_issues": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
    }


def summarize(sample_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average exactly the records supplied; do not silently filter them."""
    records = list(sample_results)
    result = _issue_counts(
        sum(record["tp"] for record in records),
        sum(record["fp"] for record in records),
        sum(record["fn"] for record in records),
    )
    blocking = _issue_counts(
        sum(record.get("tp_blocking", record["tp"]) for record in records),
        sum(record.get("fp_blocking", record["fp"]) for record in records),
        sum(record.get("fn_blocking", record["fn"]) for record in records),
    )
    n = len(records)
    exact_images = sum(record["exact"] for record in records)
    clean_images = sum(record["clean"] for record in records)
    clean_passes = sum(record["clean_pass"] for record in records)
    expected: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for record in records:
        expected.update(record["expected_issue_counts"])
        predicted.update(record["predicted_issue_counts"])
        matched.update(issue["issue_type"] for issue in record["matched_issues"])
    issue_types = list(dict.fromkeys((*ISSUE_TYPES, *expected, *predicted)))
    result.update(
        {
            "samples": n,
            "exact_images": exact_images,
            "clean_images": clean_images,
            "clean_passes": clean_passes,
            "exact_sample_accuracy": _rate(exact_images, n),
            "false_flags_per_image": _rate(result["fp"], n),
            "clean_pass_rate": _rate(clean_passes, clean_images),
            "by_issue": {
                kind: _issue_counts(matched[kind], predicted[kind] - matched[kind], expected[kind] - matched[kind])
                for kind in issue_types
            },
            "blocking": {
                **blocking,
                "false_flags_per_image": _rate(blocking["fp"], n),
            },
        }
    )
    return result
