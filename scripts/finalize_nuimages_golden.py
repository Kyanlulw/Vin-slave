"""Apply recorded AI reviews, verify source fidelity, and package a nuImages fixture.

This does not perform a review or model evaluation. Decisions must come from
actual image inspection, recorded separately in the supplied review files.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from build_nuimages_golden import ISSUES, ROOT, read_json, render_review, sha256, write_json, write_jsonl
from PIL import Image
from validate_golden_manifest import validate_manifest


def load_reviews(paths: list[Path]) -> list[dict]:
    reviews = []
    for path in paths:
        if path.suffix == ".jsonl":
            reviews.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            reviews.extend(read_json(path))
    if sorted(row["sample_index"] for row in reviews) != list(range(1, 51)):
        raise ValueError("Require exactly one recorded visual review for each of the first 50 samples.")
    for row in reviews:
        if row.get("reviewer_type") != "ai" or row.get("decision") not in {"accepted", "ambiguous", "needs_change"}:
            raise ValueError("Expected explicit AI reviewer and accepted/ambiguous/needs_change decision.")
        if not row.get("notes") or not row.get("reviewed_artifacts"):
            raise ValueError("Every decision requires notes and inspected artifact references.")
    return sorted(reviews, key=lambda row: row["sample_index"])


def verify_provenance(root: Path, rows: list[dict]) -> dict:
    samples = read_json(root / "provenance/sample.json")[:50]
    objects = {row["token"]: row for row in read_json(root / "provenance/object_ann.json")}
    categories = {row["token"]: row["name"] for row in read_json(root / "provenance/category.json")}
    attributes = {row["token"]: row["name"] for row in read_json(root / "provenance/attribute.json")}
    seen = set()
    for index, (row, sample) in enumerate(zip(rows, samples, strict=True), 1):
        provenance = row["provenance"]
        if (
            provenance["sample_index"],
            provenance["sample_token"],
            provenance["sample_data_token"],
            provenance["log_token"],
        ) != (index, sample["token"], sample["key_camera_token"], sample["log_token"]):
            raise ValueError("Manifest no longer follows the first-50 sample order.")
        if sha256(root / row["image_path"]) != provenance["image_sha256"]:
            raise ValueError("Image changed after source copy.")
        for label in read_json(root / row["gold_annotation_path"])["labels"]:
            token = label["provenance"]["object_ann_token"]
            original = objects[token]
            if token in seen or original["sample_data_token"] != sample["key_camera_token"]:
                raise ValueError("Duplicate object or object assigned to the wrong keyframe.")
            seen.add(token)
            if (
                label["class_name"] != categories[original["category_token"]]
                or list(label["bbox"].values()) != original["bbox"]
            ):
                raise ValueError("Official class or bbox was modified.")
            if label["attributes"] != {
                "occluded": None,
                "truncated": None,
                "nuimages_attributes": [attributes[t] for t in original["attribute_tokens"]],
            }:
                raise ValueError("Official attributes changed or unsupported attributes were invented.")
    expected = {
        token
        for token, row in objects.items()
        if row["sample_data_token"] in {sample["key_camera_token"] for sample in samples}
    }
    if seen != expected:
        raise ValueError("Official objects were lost or added during conversion.")
    return {
        "first_50_sample_order_verified": True,
        "original_image_hashes_verified": 50,
        "official_object_annotations_verified": len(seen),
    }


def finalize(root: Path, reviews: list[dict], archive: bool) -> None:
    rows = [
        json.loads(line)
        for line in (root / "manifests/samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fidelity = verify_provenance(root, rows)
    example = read_json(root / rows[0]["gold_annotation_path"])
    write_json(
        root / "manifests/annotation_template.json",
        {
            "image_id": example["image_id"],
            "image_width": example["image_width"],
            "image_height": example["image_height"],
            "labels": [{key: example["labels"][0][key] for key in ("label_id", "class_name", "bbox", "attributes")}],
        },
    )
    reviewed_at = datetime.now(UTC).isoformat()
    enriched_reviews = []
    for row, review in zip(rows, reviews, strict=True):
        for hash_field, path_field in (
            ("image_sha256", "image_path"),
            ("source_annotation_sha256", "source_annotation_path"),
            ("gold_annotation_sha256", "gold_annotation_path"),
        ):
            if review.get(hash_field) and review[hash_field] != sha256(root / row[path_field]):
                raise ValueError(f"Cannot replay review after {path_field} changed.")
        for reference in review["reviewed_artifacts"]:
            artifact = (root / reference).resolve()
            if not artifact.is_relative_to(root.resolve()) or not artifact.is_file():
                raise ValueError(f"Invalid reviewed artifact: {reference}")
        review = {**review, "sample_id": row["sample_id"], "reviewed_at": review.get("reviewed_at", reviewed_at)}
        row["review"] = review
        row["review_status"] = "ai_reviewed"
        row["use_for_metric"] = review["decision"] == "accepted"
        row["slice_tags"] = [
            tag for tag in row["slice_tags"] if tag not in {"pending_ai_review", "ai_reviewed", "metric_excluded"}
        ]
        row["slice_tags"].append("ai_reviewed")
        if not row["use_for_metric"]:
            row["slice_tags"].append("metric_excluded")
            row["metric_exclusion_reason"] = review["notes"]
        gold = read_json(root / row["gold_annotation_path"])
        source = read_json(root / row["source_annotation_path"])
        ignored_ids = row["evaluation_scope"]["unsupported_gold_label_ids"]
        row["ignore_regions"] = [
            {
                "gold_label_id": label["label_id"],
                "class_name": label["class_name"],
                "bbox": label["bbox"],
                "reason": "outside_detector_category_scope",
            }
            for label in gold["labels"]
            if label["label_id"] in ignored_ids
        ]
        review["image_sha256"] = sha256(root / row["image_path"])
        review["source_annotation_sha256"] = sha256(root / row["source_annotation_path"])
        review["gold_annotation_sha256"] = sha256(root / row["gold_annotation_path"])
        with Image.open(root / row["image_path"]) as image:
            render_review(image.convert("RGB"), gold, source, row, root / row["review_sheet_path"])
        enriched_reviews.append(review)
    write_jsonl(root / "manifests/samples.jsonl", rows)
    write_jsonl(root / "manifests/review_log.jsonl", enriched_reviews)
    write_jsonl(
        root / "manifests/samples.example.jsonl",
        [next(row for row in rows if row["primary_issue_type"] == issue) for issue in ISSUES],
    )
    report = validate_manifest(root / "manifests/samples.jsonl", expected_count=50)
    report["source_fidelity"] = fidelity
    write_json(root / "manifests/validation_report.json", report)
    if not report["valid"]:
        raise ValueError(json.dumps(report))
    all_counts = Counter(row["primary_issue_type"] for row in rows)
    eligible = [row for row in rows if row["use_for_metric"]]
    metric_counts = Counter(row["primary_issue_type"] for row in eligible)
    excluded = [row["provenance"]["sample_index"] for row in rows if not row["use_for_metric"]]
    summary = {
        "samples": 50,
        "ai_reviewed": 50,
        "human_adjudicated": 0,
        "metric_eligible": len(eligible),
        "metric_excluded": len(excluded),
        "excluded_sample_indices": excluded,
        "issue_counts": dict(all_counts),
        "metric_issue_counts": dict(metric_counts),
        "metric_split_counts": dict(Counter(row["split"] for row in eligible)),
        "review_decision_counts": dict(Counter(review["decision"] for review in enriched_reviews)),
    }
    write_json(root / "manifests/review_summary.json", summary)
    table = "\n".join(f"| `{issue}` | {all_counts[issue]} | {metric_counts[issue]} |" for issue in ISSUES)
    readme = f"""# Golden Testset v0.2 — nuImages, first 50 samples

This is a separate version of the KITTI `golden_v0_1` evaluation template. The objective is unchanged: evaluate single-image 2D label QA with one primary issue per sample, including clean controls. Temporal drift, tracking, ID switches and multi-frame trajectories are outside scope.

The dataset contains the **first 50 records in the local nuImages v1.0-mini `sample.json`, in their original array order**. That mini release contains exactly 50 samples. Each selected image is the annotated `key_camera_token` image; neighboring unannotated sweeps are excluded. IDs `nuimages-000001` through `nuimages-000050` correspond to that order. Original image bytes are preserved.

## Contents and review status

- 50 images, 50 source JSON files, 50 gold JSON files and 50 visual review sheets.
- 506 original object annotations, with native category names and attributes preserved.
- 35 development samples and 15 blind samples; no capture log crosses the split.
- All 50 samples received **AI visual review**. This release is not independently human-adjudicated.
- **{len(eligible)} samples are eligible for the declared synthetic evaluation; {len(excluded)} are excluded from headline metrics**. Excluded sample indices: {", ".join(str(index) for index in excluded)}. They remain in the first-50 manifest for diagnostic use, with individual reasons.

| Primary issue | All samples | Metric eligible |
|---|---:|---:|
{table}

`use_for_metric` is the scoring gate. `review_status="ai_reviewed"` records who reviewed a sample; `review.decision` distinguishes accepted and ambiguous cases. No AI decision is labeled `human_reviewed` or human `approved`. Details: [review summary](manifests/review_summary.json), [per-image review log](manifests/review_log.jsonl), [validation report](manifests/validation_report.json).

## Same template contract

```text
images/                    Original keyframe JPEGs
source_annotations/        Agent inputs with a single controlled mutation, or clean copies
gold_annotations/          Official nuImages reference converted to template JSON
manifests/samples.jsonl     Paths, expected issues/actions, splits and metric eligibility
manifests/issue_taxonomy.json
manifests/annotation_template.json
manifests/source_generation_plan.json
manifests/selection.json    Exact source order, tokens, filenames and image hashes
manifests/class_mapping.json
manifests/review_log.jsonl  Actual AI review observations and annotation hashes
review/                    Gold/source overlays and target closeups for every image
provenance/                Source annotation/category/attribute/sample tables
LICENSE.nuimages           Copied upstream license
checksums.sha256            Final file-integrity inventory
```

Every ordinary source label links to its gold label with `gold_label_id`. Extra labels have a null link; duplicate labels share the original link. Each non-clean sample has exactly one expected issue. Class mutation preserves geometry; missing removes one label; extra adds one reviewed background box; bbox mutation shrinks both dimensions to 70% (IoU 0.49); duplicate adds one identical box (IoU 1.0); clean preserves all classes, boxes and attributes.

## Reference semantics and detector scope

nuImages supplies amodal pixel `[xmin, ymin, xmax, ymax]` boxes. They are copied into `x1/y1/x2/y2` without converting from normalized YOLO or inventing visible-only extents. Occlusion and truncation booleans are `null` because the source tables do not provide them; native attribute names are retained separately. See the [official nuImages schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuimages.md).

The gold references preserve all 506 official boxes, including **121 labels outside the current detector vocabulary**. The other 385 map to supported detector classes. Source copies differ only by their recorded mutations. `class_mapping.json` snapshots the application's mapping; unsupported categories map to null. All injected object-target errors use supported categories. Samples 25 and 45, which have no supported reference labels, are clean controls; sample 45 is metric-excluded after review because its bicycle-rack taxonomy makes a naive empty-negative interpretation misleading.

For a detector benchmark, filter source labels by that frozen mapping and honor `ignore_regions` when scoring. Match supported objects first, then ignore unmatched predictions attributable to unsupported regions; do not suppress a valid supported match merely because an amodal unsupported box overlaps it. In particular, a personal-mobility or bicycle-rack region must not become a fabricated missing-label error. These are evaluator metadata: the current Agent does not automatically consume `ignore_regions`.

The current standalone label loader accepts YOLO TXT and VOC XML, **not golden JSON via `label_path`**. Read the source document and pass its labels through the existing `gt_labels` input instead:

```python
import json
from pathlib import Path

root = Path("eval/golden_v0_2_nuimages")
sample = json.loads((root / "manifests/samples.jsonl").read_text().splitlines()[0])
mapping = json.loads((root / "manifests/class_mapping.json").read_text())["detector_mapping"]
source = json.loads((root / sample["source_annotation_path"]).read_text())
agent_input = {{
    "image_path": str((root / sample["image_path"]).resolve()),
    "gt_labels": [
        {{key: label[key] for key in ("label_id", "class_name", "bbox")}}
        for label in source["labels"] if mapping.get(label["class_name"])
    ],
}}
```

Here `gt_labels` is the Agent's existing name for annotations under inspection. Gold reference files, `gold_issues`, source-to-gold links and generation plans belong on the evaluator side; remove provenance and `gold_label_id` from labels before supplying them to a model. Do not pass gold boxes as detector predictions when reporting model accuracy. Structural validation and an oracle rule regression are different from a real detector evaluation. **No model accuracy result was produced while creating this dataset.**

The blind split is reserved from threshold/model tuning. Dataset construction and review inspect both splits to establish references; neither split has been used here to tune or score a model.

## Reproduce and validate with the project .venv

From the repository root in PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe scripts/validate_golden_manifest.py eval/golden_v0_2_nuimages/manifests/samples.jsonl --expected-count 50
.\\.venv\\Scripts\\python.exe -m unittest discover -s scripts/tests -p test_validate_golden_manifest.py
```

Build into a new directory (the builder refuses to overwrite any existing dataset):

```powershell
.\\.venv\\Scripts\\python.exe scripts/build_nuimages_golden.py --source-root nuimages-v1.0-mini --output scratch/nuimages_golden_rebuild --count 50
.\\.venv\\Scripts\\python.exe scripts/finalize_nuimages_golden.py --dataset scratch/nuimages_golden_rebuild --reviews eval/golden_v0_2_nuimages/manifests/review_log.jsonl
```

Replaying reviews is valid only for the same unchanged images and annotations. The release records source metadata hashes, per-image hashes, source/gold review hashes and a final checksum inventory. The included source license continues to apply.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    review_lines = [
        "# AI visual review record",
        "",
        "Each image and its gold/source sheet was inspected. Full observations are in `../manifests/review_log.jsonl`.",
        "",
        "| Sample | Issue | Decision | Sheet |",
        "|---|---|---|---|",
    ]
    for row in rows:
        index = row["provenance"]["sample_index"]
        review_lines.append(
            f"| {index:02d} | `{row['primary_issue_type']}` | {row['review']['decision']} | [Open](nuimages-{index:06d}.jpg) |"
        )
    (root / "review/README.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    build_info = read_json(root / "manifests/build_info.json")
    build_info.update(
        builder_sha256=sha256(ROOT / "scripts/build_nuimages_golden.py"),
        finalizer_sha256=sha256(Path(__file__)),
        validator_sha256=sha256(ROOT / "scripts/validate_golden_manifest.py"),
        review_summary=summary,
    )
    write_json(root / "manifests/build_info.json", build_info)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (root / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8"
    )
    if archive:
        archive_path = root.with_suffix(".zip")
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    package.write(path, f"{root.name}/{path.relative_to(root).as_posix()}")
        print(f"Archive: {archive_path}")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/golden_v0_2_nuimages")
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--archive", action="store_true")
    arguments = parser.parse_args()
    finalize(arguments.dataset, load_reviews(arguments.reviews), arguments.archive)


if __name__ == "__main__":
    main()
