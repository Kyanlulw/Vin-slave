"""Offline regression checks for golden annotation corruption and leakage."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_golden_manifest import ISSUE_ACTIONS, validate_manifest


class GoldenManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("images", "gold_annotations", "source_annotations", "manifests"):
            (self.root / folder).mkdir()
        self.manifest = self.root / "manifests" / "samples.jsonl"

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def sample(self, kind="clean_no_issue", index=1, *, legacy=False):
        image_id = f"test-{index:06d}"
        Image.new("RGB", (200, 120)).save(self.root / "images" / f"{image_id}.png")
        gold_id = f"gold-{image_id}-001"
        source_id = f"source-{image_id}-001"
        gold = {
            "image_id": image_id,
            "image_width": 200,
            "image_height": 120,
            "labels": [
                {
                    "label_id": gold_id,
                    "class_name": "vehicle.car",
                    "bbox": {"x1": 40, "y1": 30, "x2": 100, "y2": 90},
                    "attributes": {"occluded": None},
                }
            ],
        }
        source = copy.deepcopy(gold)
        label = source["labels"][0]
        label["label_id"] = source_id
        if not legacy:
            label["gold_label_id"] = gold_id
        issue = {
            "issue_id": f"issue-{index}",
            "issue_type": kind,
            "severity": "medium",
            "expected_action": ISSUE_ACTIONS[kind],
            "source_label_id": source_id,
            "gold_label_id": gold_id,
            "source_class": "vehicle.car",
            "gold_class": "vehicle.car",
        }
        if kind == "wrong_class":
            label["class_name"] = issue["source_class"] = "human.pedestrian.adult"
        elif kind == "missing_label":
            source["labels"] = []
            issue["source_label_id"] = None
            issue.pop("source_class")
        elif kind == "extra_or_wrong_label":
            extra = copy.deepcopy(label)
            extra["label_id"] = issue["source_label_id"] = f"source-{image_id}-extra-001"
            extra["bbox"] = {"x1": 120, "y1": 10, "x2": 160, "y2": 30}
            if not legacy:
                extra["gold_label_id"] = None
            source["labels"].append(extra)
            issue["gold_label_id"] = None
            issue.pop("gold_class")
        elif kind == "bbox_misaligned":
            label["bbox"]["x1"] += 30
            label["bbox"]["x2"] += 30
        elif kind == "duplicate_label":
            duplicate = copy.deepcopy(label)
            duplicate["label_id"] = issue["duplicate_label_id"] = f"source-{image_id}-duplicate-001"
            duplicate["bbox"]["x1"] += 2
            duplicate["bbox"]["x2"] += 2
            source["labels"].append(duplicate)
        sample = {
            "sample_id": f"sample-{index}",
            "split": "dev",
            "image_path": f"images/{image_id}.png",
            "source_annotation_path": f"source_annotations/{image_id}.json",
            "gold_annotation_path": f"gold_annotations/{image_id}.json",
            "primary_issue_type": kind,
            "gold_issues": [] if kind == "clean_no_issue" else [issue],
            "review_status": "pending_human_review" if legacy else "pending_ai_review",
            "slice_tags": ["synthetic_source"],
            "use_for_metric": False,
            "provenance": {"log_token": f"log-{index}"},
        }
        self.write_json(self.root / sample["gold_annotation_path"], gold)
        self.write_json(self.root / sample["source_annotation_path"], source)
        return sample

    def validate(self, samples, **kwargs):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        return validate_manifest(self.manifest, **kwargs)

    def change_annotation(self, sample, field, edit):
        path = self.root / sample[field]
        document = json.loads(path.read_text())
        edit(document)
        self.write_json(path, document)

    def assert_invalid(self, result, text):
        self.assertFalse(result["valid"], result)
        self.assertIn(text, "\n".join(result["errors"]))

    def test_all_six_mutations_validate_with_explicit_links_and_legacy_ids(self):
        samples = [
            self.sample(kind, index, legacy=legacy)
            for index, (legacy, kind) in enumerate(
                ((legacy, kind) for legacy in (False, True) for kind in ISSUE_ACTIONS),
                1,
            )
        ]
        report = self.validate(samples, expected_count=12)
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["issue_counts"], dict.fromkeys(ISSUE_ACTIONS, 2))

    def test_unrelated_annotation_change_is_rejected(self):
        sample = self.sample("wrong_class")
        self.change_annotation(
            sample, "source_annotation_path", lambda doc: doc["labels"][0]["attributes"].update(occluded=True)
        )
        self.assert_invalid(self.validate([sample]), "unrelated fields")

    def test_bbox_must_be_finite_positive_and_in_image(self):
        for coordinate, value, expected in (
            ("x1", float("nan"), "finite"),
            ("x1", -1, "in bounds"),
            ("x2", 40, "positive width"),
        ):
            with self.subTest(value=value):
                sample = self.sample()
                self.change_annotation(
                    sample, "gold_annotation_path", lambda doc: doc["labels"][0]["bbox"].update({coordinate: value})
                )
                self.assert_invalid(self.validate([sample]), expected)

    def test_dimensions_and_label_identity_are_verified(self):
        sample = self.sample()
        self.change_annotation(sample, "gold_annotation_path", lambda doc: doc.update(image_width=201))
        self.assert_invalid(self.validate([sample]), "image_width")
        sample = self.sample()
        self.change_annotation(
            sample, "source_annotation_path", lambda doc: doc["labels"].append(copy.deepcopy(doc["labels"][0]))
        )
        self.assert_invalid(self.validate([sample]), "duplicate label_id")

    def test_paths_cannot_escape_dataset(self):
        for path in ("../outside.png", "C:\\outside.png", "/outside.png", "\\\\server\\share\\outside.png"):
            with self.subTest(path=path):
                sample = self.sample()
                sample["image_path"] = path
                self.assert_invalid(self.validate([sample]), "inside the dataset root")

    def test_issue_links_classes_actions_and_iou_are_verified(self):
        for kind, field, value, expected in (
            ("wrong_class", "source_label_id", "missing", "does not exist"),
            ("wrong_class", "gold_class", "bus", "differs from its annotation"),
            ("missing_label", "expected_action", "delete_label", "requires action"),
            ("bbox_misaligned", "source_gold_iou", 0.9, "differs from measured"),
        ):
            with self.subTest(field=field):
                sample = self.sample(kind)
                sample["gold_issues"][0][field] = value
                self.assert_invalid(self.validate([sample]), expected)

    def test_sample_identity_count_and_log_split_leakage(self):
        sample = self.sample()
        self.assert_invalid(self.validate([sample], expected_count=50), "expected 50 samples")
        self.assert_invalid(self.validate([sample, sample]), "duplicate sample_id")
        second = self.sample(index=2)
        second["split"] = "blind"
        second["provenance"]["log_token"] = sample["provenance"]["log_token"]
        self.assert_invalid(self.validate([sample, second]), "crosses dev/blind")

    def test_ai_review_can_enable_regression_metrics_without_claiming_human_review(self):
        sample = self.sample()
        sample["use_for_metric"] = True
        self.assert_invalid(self.validate([sample]), "requires an approved or ai_reviewed")
        sample.update(review_status="ai_reviewed", review={"reviewer_type": "ai"})
        self.assertTrue(self.validate([sample])["valid"])
        sample["slice_tags"].append("human_reviewed")
        self.assert_invalid(self.validate([sample]), "cannot claim")


if __name__ == "__main__":
    unittest.main()
