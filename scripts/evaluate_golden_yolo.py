"""Run real YOLO26x predictions and evaluate Label QA issue detection offline.

Predictions depend only on image pixels and fixed inference settings. The gold
annotations are opened by the scorer after inference; no LLM or database is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".venv/.cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/.cache/matplotlib"))

from golden_issue_metrics import score_sample, summarize  # noqa: E402
from validate_golden_manifest import validate_manifest  # noqa: E402

from src.agents.geometry import iou  # noqa: E402
from src.agents.nodes import flagging, matching  # noqa: E402
from src.services.yolo import TARGET_DETECTION_CLASSES, canonical_detection_class  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def intersection_over_prediction(prediction: dict, region: dict) -> float:
    p, r = prediction, region
    intersection = max(0.0, min(p["x2"], r["x2"]) - max(p["x1"], r["x1"])) * max(
        0.0, min(p["y2"], r["y2"]) - max(p["y1"], r["y1"])
    )
    area = (p["x2"] - p["x1"]) * (p["y2"] - p["y1"])
    return intersection / area if area > 0 else 0.0


def qa_flags(source: dict, predictions: list[dict], mapping: dict) -> dict:
    labels = [
        {key: label[key] for key in ("label_id", "class_name", "bbox")}
        for label in source["labels"]
        if mapping.get(label["class_name"])
    ]
    unsupported = [label for label in source["labels"] if not mapping.get(label["class_name"])]
    assignments = matching.match_labels(labels, predictions)
    image_size = (
        (source["image_width"], source["image_height"])
        if source.get("image_width") and source.get("image_height")
        else None
    )
    raw_flags = flagging.flag_issues(
        assignments["matches"], assignments["unmatched_gt"], assignments["unmatched_pred"], labels,
        image_size=image_size,
    )
    scored, ignored = [], []
    for issue in raw_flags:
        # Only unmatched missing-label flags may be outside the declared category
        # scope. Existing source matches and related boxes always take precedence.
        if issue["issue_type"] == "missing_label":
            box = issue["evidence"]["bbox"]
            related = any(iou(box, label["bbox"]) >= 0.1 for label in labels)
            region = next(
                (label for label in unsupported if intersection_over_prediction(box, label["bbox"]) >= 0.5),
                None,
            )
            if region is not None and not related:
                ignored.append({"flag": issue, "reason": "unsupported_category_region", "region_source_label_id": region["label_id"]})
                continue
        scored.append(issue)
    return {"flags": scored, "raw_flags": raw_flags, "ignored_flags": ignored,
            "supported_source_labels": len(labels), "unsupported_source_labels": len(unsupported)}


def metric_text(value, *, percentage: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.1f}%" if percentage else f"{value:.3f}"


def write_report(output: Path, result: dict) -> None:
    eligible, all_samples = result["eligible"], result["all_samples"]
    metrics = [
        ("Issue precision", "precision", False), ("Issue recall", "recall", False), ("Issue F1", "f1", False),
        ("Exact sample accuracy", "exact_sample_accuracy", True), ("False flags / image", "false_flags_per_image", False),
        ("Clean pass rate", "clean_pass_rate", True),
    ]
    lines = [
        f"# YOLO26x + Label QA evaluation — {result['dataset']}", "",
        f"Actual image inference completed for **{all_samples['samples']} samples**. The metric-eligible subset contains **{eligible['samples']} samples**.", "",
        "| Metric | Eligible samples | All samples (diagnostic) |", "|---|---:|---:|",
    ]
    for label, key, percentage in metrics:
        lines.append(f"| {label} | {metric_text(eligible[key], percentage=percentage)} | {metric_text(all_samples[key], percentage=percentage)} |")
    lines += ["", "| Count | Eligible | All |", "|---|---:|---:|"]
    for name, key in (("True-positive issues", "tp"), ("False-positive flags", "fp"), ("Missed issues", "fn"),
                      ("Expected issues", "expected_issues"), ("Predicted flags", "predicted_issues"),
                      ("Exact samples", "exact_images"), ("Clean images", "clean_images"), ("Clean passes", "clean_passes")):
        lines.append(f"| {name} | {eligible[key]} | {all_samples[key]} |")
    blocking_e, blocking_a = eligible.get("blocking", {}), all_samples.get("blocking", {})
    lines += ["", "## Blocking-only view (diagnostic)", "",
              "The headline metrics above count every emitted in-scope flag. This view scores only flags that force a review (`blocking: true`); advisory flags are excluded, and a gold issue counts as found only when matched by a blocking flag.", "",
              "| Metric | Eligible | All samples |", "|---|---:|---:|"]
    for label, key, percentage in (("Blocking precision", "precision", False), ("Blocking recall", "recall", False),
                                   ("Blocking F1", "f1", False), ("Blocking false flags / image", "false_flags_per_image", False)):
        lines.append(f"| {label} | {metric_text(blocking_e.get(key), percentage=percentage)} | {metric_text(blocking_a.get(key), percentage=percentage)} |")
    lines += ["", "## Recall by expected issue (eligible samples)", "",
              "| Issue | Expected | TP | FN | Recall |", "|---|---:|---:|---:|---:|"]
    for name, values in eligible["by_issue"].items():
        if values["expected_issues"]:
            lines.append(f"| {name} | {values['expected_issues']} | {values['tp']} | {values['fn']} | {metric_text(values['recall'], percentage=True)} |")
    lines += ["", "## Eligible split results", "",
              "| Split | Images | Precision | Recall | F1 | Exact | FP/image |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, values in result["eligible_by_split"].items():
        lines.append(f"| {split} | {values['samples']} | {metric_text(values['precision'])} | {metric_text(values['recall'])} | {metric_text(values['f1'])} | {metric_text(values['exact_sample_accuracy'], percentage=True)} | {metric_text(values['false_flags_per_image'])} |")
    lines += ["", "## Fixed protocol", "",
              "These are annotation-issue metrics for a pretrained detector plus the existing QA matching/flagging rules. They are not YOLO mAP or ordinary object-detection precision/recall.", "",
              "- Micro precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2TP / (2TP + FP + FN).",
              "- A TP requires the correct issue type and target source label. Duplicate matches require the same unordered source-label pair. Missing labels require the correct canonical class and IoU >= 0.5 to the gold target. Matching is one-to-one.",
              "- Exact sample accuracy requires FP = FN = 0. False flags/image = FP / evaluated images. Clean pass rate = zero-flag clean images / all clean images.",
              "- All emitted in-scope flags count, including loose_bbox and nonblocking flags. Evidence-gate thresholds (wrong-class confidence floor and sibling groups, missing-label low band, extra-label minimum area) were tuned on the dev split only; the blind split was scored once with the frozen configuration.",
              "- Unsupported native source classes are excluded according to the frozen dataset mapping. An unmatched missing-label flag is ignored only when >=50% of its predicted area lies inside an unsupported source box and it has IoU <0.1 to every supported source box. No gold object or issue is used by this scope filter.",
              "- The original source/gold files and metric-eligibility flags remain unchanged. Raw predictions, raw QA flags, ignored flags and individual matches are retained for audit.",
              "- No gold annotations were used as predictions. No LLM explanations or database calls were used.", "",
              "## Historical screenshot", "",
              "The user-supplied v0.1 screenshot reports YOLO precision 0.216, recall 0.923, F1 0.350, exact accuracy 13.3%, false flags/image 2.9 and clean pass 0%. Its evaluator/configuration was not available. The current v0.1 manifest has 25 expected issues, so its micro recall cannot equal 0.923; current rules also differ from behavior documented in the old guide. Treat those numbers as historical context, not an exact baseline under this protocol.", "",
              "## Reproducibility", "",
              f"Model SHA-256: `{result['run']['model_sha256']}`.",
              f"Inference: confidence {result['run']['inference']['conf']}, image size {result['run']['inference']['imgsz']}, device `{result['run']['device']}`, native end-to-end head, fixed application class list.",
              f"Ignored unsupported-region flags across all images: {result['ignored_flag_count']}. Raw prediction count: {result['raw_prediction_count']}.",
              "See `run_config.json` for package versions, checkpoint identity, rule thresholds, file hashes and exact command. `raw_predictions.jsonl` contains only image inference. `sample_results.jsonl` records QA decisions and scoring. `metrics.json` contains full precision values and per-issue counts.", "",
              "The nuImages fixture contains AI-reviewed synthetic source mutations. Its eligible subset is not independently human-adjudicated. Small per-issue counts limit generalization to production performance.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 8), facecolor="white")
    figure.suptitle("YOLO26x + Label QA | " + result["dataset"], fontsize=17, fontweight="bold", y=0.98)
    for axis, title, values in zip(axes, ("Eligible samples", "All samples — diagnostic"), (eligible, all_samples), strict=True):
        axis.axis("off")
        axis.text(0, 1.0, f"{title} ({values['samples']})", fontsize=13, fontweight="bold", transform=axis.transAxes)
        for index, (label, key, percentage) in enumerate(metrics):
            y = 0.91 - index * 0.155
            axis.text(0, y, label, fontsize=10, color="#555555", transform=axis.transAxes)
            axis.text(0, y - 0.059, metric_text(values[key], percentage=percentage), fontsize=24, color="#172a3a", transform=axis.transAxes)
    figure.text(0.06, 0.018, f"Issue-level evaluation • conf {result['run']['inference']['conf']} • imgsz {result['run']['inference']['imgsz']} • fixed rules • all in-scope flags counted", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0.025, 0.05, 0.98, 0.95), w_pad=8)
    figure.savefig(output / "metrics.png", dpi=170, facecolor="white")
    plt.close(figure)


def evaluate(arguments) -> None:
    dataset = arguments.dataset.resolve()
    model_path = arguments.model.resolve()
    if model_path.name.lower() != "yolo26x.pt" or not model_path.is_file():
        raise ValueError("This evaluation requires the existing local yolo26x.pt checkpoint.")
    if arguments.output:
        output = arguments.output.resolve()
    else:
        output = ROOT / "eval/results" / (dataset.name + "_yolo26x_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    if output.exists() and not arguments.resume:
        raise FileExistsError("Result directory exists. Use --resume for its verified prediction cache, or a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dataset / "manifests/samples.jsonl"
    validation = validate_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(json.dumps(validation))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_path = dataset / "manifests/class_mapping.json"
    if class_path.is_file():
        class_mapping = read_json(class_path)["detector_mapping"]
        if any(value != canonical_detection_class(name) for name, value in class_mapping.items()):
            raise ValueError("Frozen dataset mapping differs from current app; explicitly version this protocol before running.")
    else:
        names = {label["class_name"] for row in rows for label in read_json(dataset / row["source_annotation_path"])["labels"]}
        class_mapping = {name: canonical_detection_class(name) for name in names}
    # Imports happen after the offline integrity checks, and weights never download.
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(arguments.threads)
    device = arguments.device if arguments.device != "auto" else "0" if torch.cuda.is_available() else "cpu"
    settings = {"conf": arguments.conf, "imgsz": arguments.imgsz, "iou": 0.7, "max_det": 300, "rect": True,
                "half": False, "augment": False, "end2end": True, "batch": 1}
    model_hash = digest(model_path)
    image_hashes = {row["sample_id"]: digest(dataset / row["image_path"]) for row in rows}
    cache_identity = {"model_sha256": model_hash, "inference": settings, "target_classes": TARGET_DETECTION_CLASSES,
                      "images": image_hashes, "ultralytics": ultralytics.__version__, "torch": torch.__version__, "device": device}
    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    cached = {}
    predictions_path = output / "raw_predictions.jsonl"
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["cache_key"] != cache_key or row["sample_id"] not in image_hashes:
                raise ValueError("Cached predictions belong to another model, input set, version or configuration.")
            if row["sample_id"] in cached:
                raise ValueError("Duplicate cached sample.")
            cached[row["sample_id"]] = row
    model = YOLO(str(model_path), task="detect")
    architecture = model.model.yaml
    if architecture.get("scale") != "x" or "26" not in str(architecture.get("yaml_file", "")):
        raise ValueError(f"Checkpoint architecture is not confirmed as YOLO26x: {architecture.get('yaml_file')}, scale={architecture.get('scale')}")
    model_names = model.names
    class_ids = [index for index, name in model_names.items() if name in TARGET_DETECTION_CLASSES]
    if len(class_ids) != len(TARGET_DETECTION_CLASSES):
        raise ValueError("Checkpoint does not contain every configured detection class.")
    tracked_files = [ROOT / "scripts/evaluate_golden_yolo.py", ROOT / "scripts/golden_issue_metrics.py",
                     ROOT / "src/agents/nodes/matching.py", ROOT / "src/agents/nodes/flagging.py", ROOT / "src/services/yolo.py"]
    configuration = {"dataset": dataset.name, "manifest_sha256": digest(manifest), "model_path": str(model_path), "model_sha256": model_hash,
                     "model_yaml": architecture.get("yaml_file"), "model_scale": architecture.get("scale"),
                     "model_parameter_count_before_fusion": sum(parameter.numel() for parameter in model.model.parameters()),
                     "inference": settings, "device": device, "torch_threads": arguments.threads,
                     "target_classes": TARGET_DETECTION_CLASSES, "class_ids": class_ids, "class_mapping": class_mapping,
                     "python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
                     "packages": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "numpy", "scipy", "pillow")},
                     "code_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in tracked_files},
                     "rule_thresholds": {"matching_iou": matching.IOU_MATCH_THRESHOLD, "bbox_misaligned_min_iou": flagging.BBOX_MISALIGN_IOU_MIN,
                                         "loose_bbox_max_iou": flagging.LOOSE_BBOX_IOU_MAX, "duplicate_iou": flagging.DUPLICATE_GT_IOU_THRESHOLD,
                                         "missing_conf_low": flagging.MISSING_LABEL_CONF_LOW, "missing_conf_high": flagging.MISSING_LABEL_CONF_HIGH,
                                         "wrong_class_conf_min": flagging.WRONG_CLASS_CONF_MIN,
                                         "wrong_class_sibling_conf_min": flagging.WRONG_CLASS_SIBLING_CONF_MIN,
                                         "wrong_class_sibling_groups": [sorted(g) for g in flagging.WRONG_CLASS_SIBLING_GROUPS],
                                         "extra_label_min_area_fraction": flagging.EXTRA_LABEL_MIN_AREA_FRACTION,
                                         "bbox_misalign_min_area_fraction": flagging.BBOX_MISALIGN_MIN_AREA_FRACTION},
                     "scoring": {"missing_label_target_iou": 0.5, "count_loose_and_nonblocking_flags": True, "ignore_overlap_over_prediction": 0.5,
                                 "protect_source_related_iou": 0.1, "one_to_one_target_matching": True, "blocking_view": True},
                     "cache_key": cache_key, "command": subprocess.list2cmdline(sys.argv), "started_at": datetime.now(UTC).isoformat(),
                     "annotation_sha256": {row[role]: digest(dataset / row[role]) for row in rows for role in ("source_annotation_path", "gold_annotation_path")}}
    write_json(output / "run_config.json", configuration)
    print(json.dumps({"output": str(output), "checkpoint": model_path.name, "scale": architecture.get("scale"), "device": device,
                      "samples": len(rows), "cached": len(cached), "ultralytics": ultralytics.__version__}), flush=True)
    with predictions_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in cached:
                continue
            started = time.perf_counter()
            result = model.predict(str(dataset / row["image_path"]), device=device, classes=class_ids, verbose=False, save=False, **settings)[0]
            predictions = [{"class_name": model_names[int(box.cls.item())], "confidence": float(box.conf.item()),
                            "bbox": dict(zip(("x1", "y1", "x2", "y2"), [float(value) for value in box.xyxy[0].tolist()], strict=True))}
                           for box in result.boxes]
            record = {"sample_id": row["sample_id"], "image_path": row["image_path"], "image_sha256": image_hashes[row["sample_id"]],
                      "cache_key": cache_key, "prediction_seconds": time.perf_counter() - started,
                      "predictions": predictions, "ultralytics_speed_ms": result.speed}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            cached[row["sample_id"]] = record
            print(f"[{index}/{len(rows)}] {Path(row['image_path']).stem}: {len(predictions)} detections, {record['prediction_seconds']:.2f}s", flush=True)
    samples = []
    with (output / "sample_results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            source = read_json(dataset / row["source_annotation_path"])
            qa = qa_flags(source, cached[row["sample_id"]]["predictions"], class_mapping)
            gold = read_json(dataset / row["gold_annotation_path"])
            scored = score_sample(row, gold, qa["flags"], missing_iou=0.5)
            scored.update(qa)
            stream.write(json.dumps(scored, ensure_ascii=False, allow_nan=False) + "\n")
            samples.append(scored)
    eligible = [row for row in samples if row["use_for_metric"]]
    result = {"dataset": dataset.name, "run": configuration, "eligible": summarize(eligible), "all_samples": summarize(samples),
              "eligible_by_split": {split: summarize([row for row in eligible if row["split"] == split]) for split in ("dev", "blind")},
              "all_by_split": {split: summarize([row for row in samples if row["split"] == split]) for split in ("dev", "blind")},
              "ignored_flag_count": sum(len(row["ignored_flags"]) for row in samples),
              "raw_prediction_count": sum(len(row["predictions"]) for row in cached.values()),
              "raw_flag_counts": dict(Counter(flag["issue_type"] for row in samples for flag in row["raw_flags"])),
              "completed_at": datetime.now(UTC).isoformat()}
    write_json(output / "metrics.json", result)
    write_report(output, result)
    print(json.dumps({"eligible": result["eligible"], "all_samples": result["all_samples"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/golden_v0_2_nuimages")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo26x.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
