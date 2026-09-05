"""Dev-split sweep of flagging evidence gates over a cached prediction run.

Reuses raw_predictions.jsonl from an existing evaluate_golden_yolo output, so
no image inference happens here. Only the dev split is scored; the blind split
must stay untouched until the final configuration is frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golden_issue_metrics import score_sample, summarize  # noqa: E402
from validate_golden_manifest import validate_manifest  # noqa: E402

from src.agents import nodes as agent_nodes  # noqa: E402
from src.agents.nodes import flagging, matching  # noqa: E402
from scripts.evaluate_golden_yolo import qa_flags, read_json  # noqa: E402


def load_cache(cache_dir: Path) -> dict[str, dict]:
    cache = {}
    for line in (cache_dir / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["sample_id"]] = row["predictions"]
    return cache


def score_with_gates(rows: list[dict], dataset: Path, predictions: dict[str, list], class_mapping: dict) -> dict:
    samples = []
    for row in rows:
        source = read_json(dataset / row["source_annotation_path"])
        qa = qa_flags(source, predictions[row["sample_id"]], class_mapping)
        gold = read_json(dataset / row["gold_annotation_path"])
        scored = score_sample(row, gold, qa["flags"], missing_iou=0.5)
        samples.append(scored)
    return summarize(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/golden_v0_2_nuimages")
    parser.add_argument("--cache", type=Path, required=True, help="evaluate_golden_yolo output with raw_predictions.jsonl")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    manifest = dataset / "manifests/samples.jsonl"
    validation = validate_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(json.dumps(validation))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_mapping = read_json(dataset / "manifests/class_mapping.json")["detector_mapping"]
    predictions = load_cache(args.cache.resolve())
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in predictions]
    if missing:
        raise ValueError(f"Cache is missing samples: {missing}")

    grid = {
        "EXTRA_LABEL_MIN_AREA_FRACTION": (0.002, 0.005, 0.008),
        "MISSING_LABEL_CONF_LOW": (0.3, 0.4, 0.5),
        "WRONG_CLASS_CONF_MIN": (0.35, 0.5),
        "WRONG_CLASS_SIBLING_CONF_MIN": (0.8, 0.9),
        "BBOX_MISALIGN_MIN_AREA_FRACTION": (0.0, 0.002, 0.005),
    }
    names = list(grid)
    dev_rows = [row for row in rows if row["split"] == "dev"]
    dev_predictions = {row["sample_id"]: predictions[row["sample_id"]] for row in dev_rows}

    results = []
    defaults = {name: getattr(flagging, name) for name in names}
    for values in itertools.product(*(grid[name] for name in names)):
        combo = dict(zip(names, values, strict=True))
        for name, value in combo.items():
            setattr(flagging, name, value)
        metrics = score_with_gates(dev_rows, dataset, dev_predictions, class_mapping)
        results.append((combo, metrics))
    for name, value in defaults.items():
        setattr(flagging, name, value)

    results.sort(key=lambda item: (
        -(item[1]["f1"] or 0.0),
        item[1]["false_flags_per_image"],
        -(item[1]["recall"] or 0.0),
    ))
    print(f"dev samples: {len(dev_rows)} | combos: {len(results)} | cache: {args.cache}")
    print(f"{'combo':<64}{'P':>7}{'R':>7}{'F1':>7}{'FP/img':>8}{'exact':>7}{'blkF1':>7}{'blkFP':>7}")
    for combo, metrics in results:
        combo_text = " ".join(
            f"{name.split('_')[0][:3]}{''.join(part[0] for part in name.split('_')[1:])}={value:g}"
            for name, value in combo.items()
        )
        blocking = metrics.get("blocking", {})
        print(f"{combo_text:<64}"
              f"{metrics['precision'] or 0:>7.3f}{metrics['recall'] or 0:>7.3f}{metrics['f1'] or 0:>7.3f}"
              f"{metrics['false_flags_per_image']:>8.2f}{metrics['exact_sample_accuracy'] or 0:>7.1%}"
              f"{blocking.get('f1') or 0:>7.3f}{blocking.get('false_flags_per_image') or 0:>7.2f}")

    best_combo, best = results[0]
    print("\nbest combo:", json.dumps(best_combo, indent=2))
    print(json.dumps({key: best[key] for key in ("precision", "recall", "f1", "false_flags_per_image",
                                                 "exact_sample_accuracy", "clean_pass_rate")}, indent=2))
    print("by_issue recall:", {kind: values["recall"] for kind, values in best["by_issue"].items() if values["expected_issues"]})


if __name__ == "__main__":
    main()
