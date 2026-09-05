"""Validate a single-frame golden QA dataset without a database or model.

The KITTI v0.1 layout and the nuImages v0.2 layout share this contract. Source
labels may explicitly link to a gold label with ``gold_label_id``; legacy
``source-`` IDs are resolved to their corresponding ``gold-`` IDs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image

ISSUE_ACTIONS = {
    "wrong_class": "change_class",
    "missing_label": "add_label",
    "extra_or_wrong_label": "delete_label",
    "bbox_misaligned": "manual_edit_bbox",
    "duplicate_label": "merge_or_delete_duplicate",
    "clean_no_issue": "no_action",
}
REQUIRED_ISSUE_FIELDS = {
    "wrong_class": ("source_label_id", "gold_label_id", "source_class", "gold_class"),
    "missing_label": ("gold_label_id", "gold_class"),
    "extra_or_wrong_label": ("source_label_id", "source_class"),
    "bbox_misaligned": ("source_label_id", "gold_label_id", "source_class", "gold_class"),
    "duplicate_label": ("source_label_id", "duplicate_label_id", "gold_label_id", "source_class"),
    "clean_no_issue": (),
}
REVIEW_STATUSES = {
    "pending_human_review",
    "pending_ai_review",
    "approved",
    "ai_reviewed",
    "rejected",
    "ambiguous",
}


class InvalidGoldenDatasetError(ValueError):
    """An annotation or manifest field violates the golden dataset contract."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise InvalidGoldenDatasetError(message)


def _read_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(result, dict), f"{path.name} must contain a JSON object")
    return result


def _safe_path(root: Path, value: Any, field: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a nonempty relative path")
    windows_path = PureWindowsPath(value)
    normalized = value.replace("\\", "/")
    relative = Path(normalized)
    _require(
        not windows_path.drive and not windows_path.root and not relative.is_absolute() and ".." not in relative.parts,
        f"{field} must stay inside the dataset root",
    )
    resolved = (root / relative).resolve()
    _require(resolved.is_relative_to(root), f"{field} escapes the dataset root")
    _require(resolved.is_file(), f"{field} does not exist: {value}")
    return resolved


def bbox_iou(first: dict[str, float], second: dict[str, float]) -> float:
    intersection = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"])) * max(
        0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"])
    )
    area_first = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    area_second = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _annotation(path: Path, size: tuple[int, int]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _read_object(path)
    _require(isinstance(document.get("image_id"), str) and document["image_id"], "annotation image_id is required")
    for field, expected in zip(("image_width", "image_height"), size, strict=True):
        value = document.get(field)
        _require(type(value) is int and value == expected, f"{path.name}: {field} must equal image size {expected}")
    labels = document.get("labels")
    _require(isinstance(labels, list), f"{path.name}: labels must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels):
        context = f"{path.parent.name}/{path.name} labels[{index}]"
        _require(isinstance(label, dict), f"{context} must be an object")
        label_id = label.get("label_id")
        _require(isinstance(label_id, str) and label_id.strip(), f"{context} requires label_id")
        _require(label_id not in by_id, f"{context}: duplicate label_id {label_id}")
        _require(
            isinstance(label.get("class_name"), str) and label["class_name"].strip(), f"{context} requires class_name"
        )
        _require(isinstance(label.get("attributes", {}), dict), f"{context}: attributes must be an object")
        box = label.get("bbox")
        _require(isinstance(box, dict), f"{context} requires a bbox object")
        for coordinate in ("x1", "y1", "x2", "y2"):
            value = box.get(coordinate)
            _require(
                type(value) in (int, float) and math.isfinite(value), f"{context}: bbox {coordinate} must be finite"
            )
        _require(
            0 <= box["x1"] < box["x2"] <= size[0],
            f"{context}: bbox x coordinates must have positive width and stay in bounds",
        )
        _require(
            0 <= box["y1"] < box["y2"] <= size[1],
            f"{context}: bbox y coordinates must have positive height and stay in bounds",
        )
        by_id[label_id] = label
    return document, by_id


def _same_label(first: dict[str, Any], second: dict[str, Any], *, except_field: str | None = None) -> bool:
    return all(
        first.get(field, {} if field == "attributes" else None)
        == second.get(field, {} if field == "attributes" else None)
        for field in ("class_name", "bbox", "attributes")
        if field != except_field
    )


def _recorded_iou(issue: dict[str, Any], field: str, actual: float) -> None:
    if field in issue:
        recorded = issue[field]
        _require(type(recorded) in (int, float) and math.isfinite(recorded), f"{field} must be finite")
        _require(math.isclose(recorded, actual, abs_tol=1e-5), f"{field}={recorded} differs from measured {actual:.6f}")


def _review(sample: dict[str, Any]) -> None:
    status = sample.get("review_status")
    _require(status in REVIEW_STATUSES, f"invalid review_status: {status!r}")
    _require(type(sample.get("use_for_metric")) is bool, "use_for_metric must be a boolean")
    review = sample.get("review", {})
    _require(isinstance(review, dict), "review must be an object")
    tags = sample.get("slice_tags", [])
    _require(
        isinstance(tags, list) and all(isinstance(tag, str) for tag in tags), "slice_tags must be an array of strings"
    )
    if status == "ai_reviewed":
        _require(review.get("reviewer_type") == "ai", "ai_reviewed requires review.reviewer_type='ai'")
    if review.get("reviewer_type") == "ai":
        _require("human_reviewed" not in tags, "AI review cannot claim the human_reviewed tag")
        _require(status != "approved", "AI review must use ai_reviewed, not human approved status")
    if sample["use_for_metric"]:
        _require(status in {"approved", "ai_reviewed"}, "use_for_metric requires an approved or ai_reviewed sample")
        _require(not any(tag.startswith("pending_") for tag in tags), "metric sample cannot retain pending review tags")


def _issue(sample: dict[str, Any], sources: dict, golds: dict, taxonomy: dict) -> dict[str, Any]:
    kind = sample["primary_issue_type"]
    issues = sample.get("gold_issues")
    _require(isinstance(issues, list), "gold_issues must be an array")
    _require(
        len(issues) == (0 if kind == "clean_no_issue" else 1),
        "sample must have exactly one primary issue, or zero for clean",
    )
    if not issues:
        return {}
    issue = issues[0]
    _require(isinstance(issue, dict), "gold_issues[0] must be an object")
    _require(isinstance(issue.get("issue_id"), str) and issue["issue_id"].strip(), "issue_id is required")
    _require(issue.get("issue_type") == kind, "issue_type must equal primary_issue_type")
    _require(
        issue.get("severity") in taxonomy.get("allowed_severities", ["low", "medium", "high"]), "invalid issue severity"
    )
    _require(issue.get("expected_action") == ISSUE_ACTIONS[kind], f"{kind} requires action {ISSUE_ACTIONS[kind]}")
    definition = taxonomy.get("issue_types", {}).get(kind, {})
    _require(
        issue["expected_action"] in definition.get("expected_actions", [ISSUE_ACTIONS[kind]]),
        "action is not allowed by taxonomy",
    )
    required = set(REQUIRED_ISSUE_FIELDS[kind]) | set(definition.get("required_gold_issue_fields", []))
    for field in required:
        _require(isinstance(issue.get(field), str) and issue[field].strip(), f"{kind} requires {field}")
    for field, labels in (("source_label_id", sources), ("duplicate_label_id", sources), ("gold_label_id", golds)):
        if issue.get(field) is not None:
            _require(issue[field] in labels, f"{field} does not exist: {issue[field]}")
    for id_field, class_field, labels in (
        ("source_label_id", "source_class", sources),
        ("gold_label_id", "gold_class", golds),
    ):
        if issue.get(class_field) is not None:
            _require(issue.get(id_field) in labels, f"{class_field} requires a valid {id_field}")
            _require(
                issue[class_field] == labels[issue[id_field]]["class_name"],
                f"{class_field} differs from its annotation",
            )
    if kind == "missing_label":
        _require(issue.get("source_label_id") is None, "missing_label source_label_id must be null or absent")
    if kind == "extra_or_wrong_label":
        _require(issue.get("gold_label_id") is None, "extra_or_wrong_label gold_label_id must be null or absent")
    return issue


def _mutation(sample: dict[str, Any], sources: dict, golds: dict, issue: dict, taxonomy: dict) -> None:
    kind = sample["primary_issue_type"]
    target_gold = issue.get("gold_label_id")
    target_source = issue.get("source_label_id")
    duplicate_id = issue.get("duplicate_label_id")
    links: dict[str, list[dict]] = defaultdict(list)
    extras = []
    for label in sources.values():
        source_id = label["label_id"]
        if "gold_label_id" in label:
            gold_id = label["gold_label_id"]
        elif source_id == duplicate_id:
            gold_id = target_gold
        elif kind == "extra_or_wrong_label" and source_id == target_source:
            gold_id = None
        else:
            gold_id = source_id.replace("source-", "gold-", 1) if source_id.startswith("source-") else source_id
        if gold_id is None:
            extras.append(label)
        else:
            _require(
                isinstance(gold_id, str) and gold_id in golds,
                f"source label {source_id} links to unknown gold label {gold_id!r}",
            )
            links[gold_id].append(label)

    for gold_id, gold in golds.items():
        linked = links[gold_id]
        if gold_id == target_gold and kind == "missing_label":
            _require(not linked, "missing target still exists in source")
        elif gold_id == target_gold and kind == "duplicate_label":
            _require(target_source != duplicate_id, "duplicate must reference two distinct source labels")
            _require(
                {label["label_id"] for label in linked} == {target_source, duplicate_id},
                "duplicate target must have exactly the two referenced source labels",
            )
            _require(_same_label(sources[target_source], gold), "original duplicate target changed from gold")
            _require(
                _same_label(sources[duplicate_id], gold, except_field="bbox"),
                "duplicate class or attributes differ from gold",
            )
        elif gold_id == target_gold and kind in {"wrong_class", "bbox_misaligned"}:
            _require(
                len(linked) == 1 and linked[0]["label_id"] == target_source,
                "mutation target must match its source/gold links",
            )
            changed_field = "class_name" if kind == "wrong_class" else "bbox"
            _require(
                _same_label(linked[0], gold, except_field=changed_field), "target mutation changed unrelated fields"
            )
            _require(linked[0][changed_field] != gold[changed_field], f"target {changed_field} did not change")
        else:
            _require(
                len(linked) == 1 and _same_label(linked[0], gold),
                f"unrelated gold label {gold_id} was changed, removed, or duplicated",
            )

    if kind == "extra_or_wrong_label":
        _require(
            len(extras) == 1 and extras[0]["label_id"] == target_source,
            "extra mutation must add exactly the referenced source label",
        )
        max_iou = max((bbox_iou(extras[0]["bbox"], gold["bbox"]) for gold in golds.values()), default=0.0)
        _recorded_iou(issue, "max_gold_iou", max_iou)
    else:
        _require(not extras, "unexpected source labels without a gold link")

    criteria = taxonomy.get("issue_types", {}).get(kind, {}).get("recommended_criteria", {})
    if kind in {"wrong_class", "bbox_misaligned"}:
        score = bbox_iou(sources[target_source]["bbox"], golds[target_gold]["bbox"])
        minimum = criteria.get("source_gold_iou_min", 0.7 if kind == "wrong_class" else 0.2)
        _require(score >= minimum, f"{kind} IoU {score:.6f} is below {minimum}")
        if kind == "bbox_misaligned":
            maximum = criteria.get("source_gold_iou_max_exclusive", 0.7)
            _require(score < maximum, f"bbox_misaligned IoU {score:.6f} must be below {maximum}")
        _recorded_iou(issue, "source_gold_iou", score)
    if kind == "duplicate_label":
        score = bbox_iou(sources[target_source]["bbox"], sources[duplicate_id]["bbox"])
        minimum = criteria.get("duplicate_iou_min", 0.8)
        _require(score >= minimum, f"duplicate IoU {score:.6f} is below {minimum}")
        _recorded_iou(issue, "duplicate_iou", score)


def _validate_sample(sample: dict[str, Any], root: Path, taxonomy: dict) -> dict[str, Any]:
    _require(isinstance(sample.get("sample_id"), str) and sample["sample_id"].strip(), "sample_id is required")
    _require(sample.get("primary_issue_type") in ISSUE_ACTIONS, "invalid primary_issue_type")
    _require(
        sample["primary_issue_type"] in taxonomy.get("issue_types", ISSUE_ACTIONS),
        "primary_issue_type not present in taxonomy",
    )
    _require(sample.get("split") in taxonomy.get("allowed_splits", ["dev", "blind"]), "invalid split")
    _review(sample)
    paths = {
        field: _safe_path(root, sample.get(field), field)
        for field in ("image_path", "source_annotation_path", "gold_annotation_path")
    }
    _require(paths["source_annotation_path"] != paths["gold_annotation_path"], "source and gold must be separate files")
    with Image.open(paths["image_path"]) as image:
        size = image.size
        image.verify()
    source_document, sources = _annotation(paths["source_annotation_path"], size)
    gold_document, golds = _annotation(paths["gold_annotation_path"], size)
    _require(source_document["image_id"] == gold_document["image_id"], "source and gold image_id differ")
    issue = _issue(sample, sources, golds, taxonomy)
    _mutation(sample, sources, golds, issue, taxonomy)
    provenance = sample.get("provenance", {})
    _require(isinstance(provenance, dict), "provenance must be an object")
    return {
        "paths": paths,
        "image_id": gold_document["image_id"],
        "issue_id": issue.get("issue_id"),
        "log_token": provenance.get("log_token"),
    }


def validate_manifest(manifest_path: str | Path, *, expected_count: int | None = None) -> dict[str, Any]:
    """Return a JSON-serializable validation report; never edit the dataset."""
    manifest = Path(manifest_path).resolve()
    root = manifest.parent.parent if manifest.parent.name == "manifests" else manifest.parent
    errors: list[str] = []
    samples: list[dict[str, Any]] = []
    taxonomy: dict[str, Any] = {}
    try:
        taxonomy_path = root / "manifests" / "issue_taxonomy.json"
        if taxonomy_path.exists():
            taxonomy = _read_object(taxonomy_path)
            _require(
                taxonomy.get("scope") == "single_frame_2d_label_qa", "taxonomy scope must be single_frame_2d_label_qa"
            )
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                _require(isinstance(sample, dict), "manifest row must be an object")
                samples.append(sample)
            except (ValueError, TypeError) as error:
                errors.append(f"line {line_number}: {error}")
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(error))
    if not samples:
        errors.append("manifest has no samples")
    if expected_count is not None and len(samples) != expected_count:
        errors.append(f"expected {expected_count} samples, found {len(samples)}")

    seen: dict[str, set[Any]] = defaultdict(set)
    log_splits: dict[str, str] = {}
    issue_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    metric_count = 0
    valid_sample_count = 0
    for index, sample in enumerate(samples, 1):
        context = sample.get("sample_id", f"row {index}")
        try:
            data = _validate_sample(sample, root, taxonomy)
            identities = {"sample_id": sample["sample_id"], "image_id": data["image_id"], **data["paths"]}
            if data["issue_id"]:
                identities["issue_id"] = data["issue_id"]
            for field, value in identities.items():
                _require(value not in seen[field], f"duplicate {field}: {value}")
                seen[field].add(value)
            log_token = data["log_token"]
            if log_token is not None:
                _require(isinstance(log_token, str) and log_token, "provenance.log_token must be a nonempty string")
                _require(
                    log_token not in log_splits or log_splits[log_token] == sample["split"],
                    f"log_token {log_token} crosses dev/blind splits",
                )
                log_splits[log_token] = sample["split"]
            issue_counts[sample["primary_issue_type"]] += 1
            split_counts[sample["split"]] += 1
            metric_count += sample["use_for_metric"]
            valid_sample_count += 1
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(f"{context}: {error}")
    return {
        "valid": not errors,
        "manifest_path": str(manifest),
        "sample_count": len(samples),
        "valid_sample_count": valid_sample_count,
        "metric_sample_count": metric_count,
        "issue_counts": dict(sorted(issue_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "log_count": len(log_splits),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifests/samples.jsonl")
    parser.add_argument("--expected-count", type=int, help="Require this exact number of samples")
    parser.add_argument("--report", type=Path, help="Write the validation report as JSON")
    arguments = parser.parse_args()
    if arguments.expected_count is not None and arguments.expected_count < 1:
        parser.error("--expected-count must be positive")
    report = validate_manifest(arguments.manifest, expected_count=arguments.expected_count)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
