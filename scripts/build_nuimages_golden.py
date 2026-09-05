"""Build a versioned, source-traceable nuImages golden evaluation fixture offline.

Run with the repository .venv. Existing output is never overwritten. Review
decisions are applied separately after inspecting the generated review sheets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.yolo import canonical_detection_class  # noqa: E402

VERSION = "golden-v0.2-nuimages"
ISSUES = (
    "wrong_class",
    "missing_label",
    "extra_or_wrong_label",
    "bbox_misaligned",
    "duplicate_label",
    "clean_no_issue",
)
ACTIONS = ("change_class", "add_label", "delete_label", "manual_edit_bbox", "merge_or_delete_duplicate", "no_action")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bbox_area(box: dict) -> float:
    return (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])


def intersection(a: dict, b: dict) -> float:
    return max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])) * max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))


def iou(a: dict, b: dict) -> float:
    overlap = intersection(a, b)
    return overlap / (bbox_area(a) + bbox_area(b) - overlap)


def select_target(labels: list[dict], issue: str) -> dict:
    supported = [label for label in labels if canonical_detection_class(label["class_name"]) is not None]
    if not supported:
        raise ValueError(f"No detector-supported target for {issue}.")

    # Prefer substantial objects; missing labels additionally prefer isolated boxes.
    def score(label):
        overlap = max((iou(label["bbox"], other["bbox"]) for other in labels if other is not label), default=0)
        return (overlap < 0.1 if issue == "missing_label" else True, bbox_area(label["bbox"]))

    return max(supported, key=score)


def background_box(labels: list[dict], width: int, height: int) -> dict:
    # These are proposals. The visual review must confirm actual background.
    for y in (0.035, 0.13, 0.79, 0.87, 0.24):
        for x in (0.65, 0.42, 0.15, 0.79, 0.03):
            box = {
                "x1": round(x * width, 2),
                "y1": round(y * height, 2),
                "x2": round((x + 0.10) * width, 2),
                "y2": round((y + 0.08) * height, 2),
            }
            if all(intersection(box, label["bbox"]) == 0 for label in labels):
                return box
    raise ValueError("No non-overlapping extra-label proposal; choose a box explicitly.")


def mutate(gold: dict, issue: str) -> tuple[dict, list[dict], dict]:
    source = copy.deepcopy(gold)
    for label in source["labels"]:
        label["gold_label_id"] = label["label_id"]
        label["label_id"] = label["label_id"].replace("gold-", "source-", 1)
    source["annotation_role"] = "synthetic_source"
    plan = {"method": "controlled_synthetic_mutation", "version": "source-plan-v0.2-nuimages", "issue_type": issue}
    if issue == "clean_no_issue":
        return source, [], plan
    change = {
        "issue_id": f"issue-{gold['image_id']}-001",
        "issue_type": issue,
        "severity": "high" if issue in {"wrong_class", "missing_label"} else "medium",
        "expected_action": ACTIONS[ISSUES.index(issue)],
    }
    if issue == "extra_or_wrong_label":
        box = background_box(gold["labels"], gold["image_width"], gold["image_height"])
        extra_id = f"source-{gold['image_id']}-extra"
        source["labels"].append(
            {
                "label_id": extra_id,
                "gold_label_id": None,
                "class_name": "vehicle.car",
                "bbox": box,
                "attributes": {"occluded": None, "truncated": None},
                "provenance": {"synthetic": True},
            }
        )
        change.update(source_label_id=extra_id, gold_label_id=None, source_class="vehicle.car")
        plan.update(extra_class="vehicle.car", extra_bbox=box)
    else:
        target = select_target(gold["labels"], issue)
        label = next(row for row in source["labels"] if row["gold_label_id"] == target["label_id"])
        change.update(
            source_label_id=label["label_id"],
            gold_label_id=target["label_id"],
            source_class=label["class_name"],
            gold_class=target["class_name"],
        )
        plan["target_gold_label_id"] = target["label_id"]
        if issue == "wrong_class":
            label["class_name"] = (
                "human.pedestrian.adult" if canonical_detection_class(target["class_name"]) == "car" else "vehicle.car"
            )
            change.update(source_class=label["class_name"], source_gold_iou=1.0)
            plan["replacement_class"] = label["class_name"]
        elif issue == "missing_label":
            source["labels"].remove(label)
            change.update(source_label_id=None)
            change.pop("source_class")
        elif issue == "duplicate_label":
            duplicate = copy.deepcopy(label)
            duplicate["label_id"] += "-duplicate"
            source["labels"].append(duplicate)
            change.update(duplicate_label_id=duplicate["label_id"], duplicate_iou=1.0)
        else:
            box = target["bbox"]
            # Shrink inside the original bbox: exact IoU 0.49, valid even at image edges.
            dx, dy = (box["x2"] - box["x1"]) * 0.15, (box["y2"] - box["y1"]) * 0.15
            label["bbox"] = {
                "x1": round(box["x1"] + dx, 3),
                "y1": round(box["y1"] + dy, 3),
                "x2": round(box["x2"] - dx, 3),
                "y2": round(box["y2"] - dy, 3),
            }
            change["source_gold_iou"] = round(iou(box, label["bbox"]), 6)
            plan["replacement_bbox"] = label["bbox"]
    return source, [change], plan


def font(size: int):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def render_review(image: Image.Image, gold: dict, source: dict, row: dict, path: Path) -> None:
    width, height = image.size
    panel_width = 1000
    panel_height = round(height * panel_width / width)
    sheet = Image.new("RGB", (2000, panel_height + 660), "#101923")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (24, 14), f"{gold['image_id']} | {row['primary_issue_type']} | {row['split']}", font=font(28), fill="white"
    )
    draw.text((24, 50), "GOLD: official nuImages boxes", font=font(21), fill="#60f7bd")
    source_title = (
        "SOURCE: unchanged clean control"
        if row["primary_issue_type"] == "clean_no_issue"
        else "SOURCE: exactly one controlled change (pink)"
    )
    draw.text((1024, 50), source_title, font=font(21), fill="#ff76b9")
    changed = row["gold_issues"][0] if row["gold_issues"] else {}
    target_gold = changed.get("gold_label_id")
    source_ids = {changed.get("source_label_id"), changed.get("duplicate_label_id")}
    annotated = []
    for document, xoffset in ((gold, 0), (source, 1000)):
        canvas = image.copy().convert("RGB")
        painter = ImageDraw.Draw(canvas)
        for label in document["labels"]:
            target = label["label_id"] == target_gold if xoffset == 0 else label["label_id"] in source_ids
            color = "#ff3897" if target and xoffset else "#28e8a3" if target else "#5edfc4"
            box = tuple(label["bbox"][key] for key in ("x1", "y1", "x2", "y2"))
            painter.rectangle(box, outline=color, width=5 if target else 2)
            short_name = (
                label["class_name"]
                .replace("human.pedestrian.", "ped.")
                .replace("vehicle.", "")
                .replace("movable_object.", "")
            )
            title = label["label_id"].split("-")[-1] + ":" + short_name
            tx, ty = box[0], max(0, box[1] - 22)
            bounds = painter.textbbox((tx, ty), title, font=font(18))
            painter.rectangle(bounds, fill="#101923")
            painter.text((tx, ty), title, fill=color, font=font(18))
        annotated.append(canvas)
        sheet.paste(canvas.resize((panel_width, panel_height)), (xoffset, 86))
    boxes = [
        label["bbox"]
        for document in (gold, source)
        for label in document["labels"]
        if label["label_id"] == target_gold or label["label_id"] in source_ids
    ]
    if boxes:
        x1, y1 = min(b["x1"] for b in boxes), min(b["y1"] for b in boxes)
        x2, y2 = max(b["x2"] for b in boxes), max(b["y2"] for b in boxes)
        margin = max(70, (x2 - x1) * 0.3, (y2 - y1) * 0.3)
        crop = (
            int(max(0, x1 - margin)),
            int(max(0, y1 - margin)),
            int(min(width, x2 + margin)),
            int(min(height, y2 + margin)),
        )
        for index, canvas in enumerate(annotated):
            detail = canvas.crop(crop)
            detail.thumbnail((970, 465))
            sheet.paste(detail, (index * 1000 + 15, panel_height + 120))
    else:
        draw.text(
            (30, panel_height + 150),
            "Clean control: source class, boxes and attributes equal gold.",
            fill="white",
            font=font(25),
        )
    decision = row.get("review", {}).get("decision", "pending")
    draw.text(
        (24, panel_height + 595),
        f"Gold labels: {len(gold['labels'])} | Source labels: {len(source['labels'])} | AI review: {decision}",
        fill="white",
        font=font(20),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def build(source_root: Path, output: Path, template: Path, count: int) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    metadata_root = source_root / "v1.0-mini"
    tables = {path.stem: read_json(path) for path in metadata_root.glob("*.json")}
    if count != 50 or len(tables["sample"]) < count:
        raise ValueError("This release requires the first 50 sample.json records.")
    samples = tables["sample"][:count]
    sample_data = {row["token"]: row for row in tables["sample_data"]}
    categories = {row["token"]: row["name"] for row in tables["category"]}
    attributes = {row["token"]: row["name"] for row in tables["attribute"]}
    sensors = {row["token"]: row for row in tables["sensor"]}
    calibrated = {row["token"]: row for row in tables["calibrated_sensor"]}
    annotations = defaultdict(list)
    for annotation in tables["object_ann"]:
        annotations[annotation["sample_data_token"]].append(annotation)
    for sample in samples:
        data = sample_data[sample["key_camera_token"]]
        if not data["is_key_frame"] or data["sample_token"] != sample["token"]:
            raise ValueError("Sample does not reference its annotated keyframe.")
        if not (source_root / data["filename"]).is_file():
            raise FileNotFoundError(data["filename"])
    issue_assignment = [ISSUES[index % len(ISSUES)] for index in range(count)]
    # These two first-50 frames contain only categories outside the detector's vocabulary.
    # Swap with existing clean slots so no unsupported object is used as an error target.
    for empty_index in [
        i
        for i, sample in enumerate(samples)
        if not any(
            canonical_detection_class(categories[a["category_token"]]) for a in annotations[sample["key_camera_token"]]
        )
    ]:
        if issue_assignment[empty_index] != "clean_no_issue":
            clean_index = next(
                i
                for i, issue in enumerate(issue_assignment)
                if issue == "clean_no_issue"
                and i != empty_index
                and any(
                    canonical_detection_class(categories[a["category_token"]])
                    for a in annotations[samples[i]["key_camera_token"]]
                )
            )
            issue_assignment[clean_index], issue_assignment[empty_index] = (
                issue_assignment[empty_index],
                "clean_no_issue",
            )
    # Last 15 records are independent capture logs in this exact first-50 release.
    dev_logs = {sample["log_token"] for sample in samples[:35]}
    blind_logs = {sample["log_token"] for sample in samples[35:]}
    if dev_logs & blind_logs:
        raise ValueError("Capture log leakage across dev/blind; choose grouped splits.")
    for name in ("images", "gold_annotations", "source_annotations", "manifests", "review", "provenance"):
        (output / name).mkdir(parents=True)
    shutil.copy2(source_root / "LICENSE", output / "LICENSE.nuimages")
    # Preserve the exact selected tables and all mini object annotations, including instance masks.
    for name in ("sample", "object_ann", "category", "attribute", "log", "sensor", "calibrated_sensor"):
        shutil.copy2(metadata_root / f"{name}.json", output / "provenance" / f"{name}.json")
    write_json(
        output / "provenance" / "keyframe_sample_data.json", [sample_data[s["key_camera_token"]] for s in samples]
    )
    taxonomy = read_json(template / "manifests" / "issue_taxonomy.json")
    taxonomy["version"] = VERSION
    write_json(output / "manifests" / "issue_taxonomy.json", taxonomy)
    shutil.copy2(template / "manifests" / "annotation_template.json", output / "manifests" / "annotation_template.json")
    rows, selections, plans = [], [], []
    class_counts = Counter()
    for index, sample in enumerate(samples, start=1):
        data = sample_data[sample["key_camera_token"]]
        image_id = f"nuimages-{index:06d}"
        source_image = source_root / data["filename"]
        image_path = output / "images" / f"{image_id}.jpg"
        shutil.copy2(source_image, image_path)
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        if image.size != (data["width"], data["height"]):
            raise ValueError(f"Dimension mismatch: {image_id}")
        labels = []
        for label_index, annotation in enumerate(annotations[data["token"]], start=1):
            class_name = categories[annotation["category_token"]]
            class_counts[class_name] += 1
            labels.append(
                {
                    "label_id": f"gold-{image_id}-{label_index:03d}",
                    "class_name": class_name,
                    "bbox": dict(zip(("x1", "y1", "x2", "y2"), map(float, annotation["bbox"]), strict=True)),
                    "attributes": {
                        "occluded": None,
                        "truncated": None,
                        "nuimages_attributes": [attributes[token] for token in annotation["attribute_tokens"]],
                    },
                    "provenance": {
                        "object_ann_token": annotation["token"],
                        "category_token": annotation["category_token"],
                        "sample_data_token": data["token"],
                    },
                }
            )
        gold = {
            "image_id": image_id,
            "image_width": image.width,
            "image_height": image.height,
            "annotation_role": "official_reference",
            "labels": labels,
        }
        if index == 1:
            write_json(
                output / "manifests/annotation_template.json",
                {
                    "image_id": image_id,
                    "image_width": image.width,
                    "image_height": image.height,
                    "labels": [{key: labels[0][key] for key in ("label_id", "class_name", "bbox", "attributes")}],
                },
            )
        issue = issue_assignment[index - 1]
        source, gold_issues, plan = mutate(gold, issue)
        selection = {
            "sample_index": index,
            "sample_token": sample["token"],
            "sample_data_token": data["token"],
            "log_token": sample["log_token"],
            "timestamp": sample["timestamp"],
            "original_image_path": data["filename"],
            "camera_channel": sensors[calibrated[data["calibrated_sensor_token"]]["sensor_token"]]["channel"],
            "image_sha256": sha256(image_path),
            "image_id": image_id,
        }
        row = {
            "sample_id": f"{VERSION}-{issue}-{index:06d}",
            "split": "dev" if index <= 35 else "blind",
            "image_path": f"images/{image_id}.jpg",
            "source_annotation_path": f"source_annotations/{image_id}.json",
            "gold_annotation_path": f"gold_annotations/{image_id}.json",
            "primary_issue_type": issue,
            "gold_issues": gold_issues,
            "slice_tags": [
                "nuimages",
                "official_reference",
                "synthetic_source",
                "pending_ai_review",
                selection["camera_channel"].lower(),
            ],
            "review_status": "pending_ai_review",
            "use_for_metric": False,
            "source_generation": {
                "method": "controlled_synthetic_mutation",
                "plan_version": "source-plan-v0.2-nuimages",
            },
            "evaluation_scope": {
                "type": "detector_supported_2d_categories",
                "unsupported_gold_label_ids": [
                    label["label_id"] for label in labels if canonical_detection_class(label["class_name"]) is None
                ],
            },
            "provenance": selection,
            "review_sheet_path": f"review/{image_id}.jpg",
        }
        write_json(output / row["gold_annotation_path"], gold)
        write_json(output / row["source_annotation_path"], source)
        render_review(image, gold, source, row, output / row["review_sheet_path"])
        rows.append(row)
        selections.append(selection)
        plans.append({"image_id": image_id, "sample_index": index, "split": row["split"], **plan})
    write_jsonl(output / "manifests" / "samples.jsonl", rows)
    write_jsonl(
        output / "manifests" / "samples.example.jsonl",
        [next(row for row in rows if row["primary_issue_type"] == issue) for issue in ISSUES],
    )
    write_json(
        output / "manifests" / "selection.json",
        {
            "source": "nuimages-v1.0-mini",
            "order": "sample.json array order, first 50 records; no filtering, sorting or shuffling",
            "samples": selections,
        },
    )
    write_json(
        output / "manifests" / "source_generation_plan.json", {"version": "source-plan-v0.2-nuimages", "samples": plans}
    )
    write_json(
        output / "manifests" / "class_mapping.json",
        {
            "policy": "Keep native nuImages class names and pixel xyxy boxes. Map only when comparing to detector predictions.",
            "detector_mapping": {name: canonical_detection_class(name) for name in categories.values()},
            "gold_label_counts": dict(class_counts),
            "unsupported_policy": "Preserve all official labels; exclude unmapped labels from detector-based scoring. Never mutate an unsupported object.",
        },
    )
    write_json(
        output / "manifests" / "build_info.json",
        {
            "version": VERSION,
            "template": "golden-v0.1 (KITTI)",
            "source_release": "nuImages v1.0-mini",
            "sample_count": count,
            "gold_label_count": sum(class_counts.values()),
            "issue_counts": dict(Counter(row["primary_issue_type"] for row in rows)),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "source_metadata_sha256": {path.name: sha256(path) for path in metadata_root.glob("*.json")},
            "template_taxonomy_sha256": sha256(template / "manifests" / "issue_taxonomy.json"),
            "builder_sha256": sha256(Path(__file__)),
            "python_version": sys.version.split()[0],
            "virtual_environment": Path(sys.prefix).name,
            "annotation_semantics_source": "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuimages.md",
            "review_policy": "AI visual review, never described as independent human adjudication. Metric eligibility is set only after review.",
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": count,
                "gold_labels": sum(class_counts.values()),
                "issues": dict(Counter(issue_assignment)),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "nuimages-v1.0-mini")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/golden_v0_2_nuimages")
    parser.add_argument("--template", type=Path, default=ROOT / "eval/golden_v0_1")
    parser.add_argument("--count", type=int, default=50)
    arguments = parser.parse_args()
    build(arguments.source_root, arguments.output, arguments.template, arguments.count)


if __name__ == "__main__":
    main()
