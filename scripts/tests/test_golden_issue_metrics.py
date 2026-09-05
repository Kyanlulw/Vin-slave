"""Issue evaluation regressions: object identity, one-to-one credit and noise."""

from __future__ import annotations

import copy
import unittest

from scripts.golden_issue_metrics import score_sample, summarize


def box(x1=0, y1=0, x2=100, y2=100):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def sample(*issues, eligible=True):
    return {
        "sample_id": "sample",
        "split": "dev",
        "primary_issue_type": issues[0]["issue_type"] if issues else "clean_no_issue",
        "use_for_metric": eligible,
        "gold_issues": list(issues),
    }


def gold(*labels):
    return {"labels": list(labels)}


def source_issue(kind="wrong_class", source_id="source-a"):
    return {"issue_type": kind, "source_label_id": source_id}


def source_flag(kind="wrong_class", source_id="source-a", **extra):
    return {"issue_type": kind, "label_id": source_id, **extra}


def missing_issue(gold_id="gold-a"):
    return {"issue_type": "missing_label", "gold_label_id": gold_id, "source_label_id": None}


def missing_flag(bbox=None, class_name="car"):
    return {
        "issue_type": "missing_label",
        "label_id": None,
        "evidence": {"class_name": class_name, "bbox": bbox or box(), "confidence": 0.9},
    }


class GoldenIssueMetricsTests(unittest.TestCase):
    def test_source_targets_require_type_and_exact_identity(self):
        for kind in ("wrong_class", "extra_or_wrong_label", "bbox_misaligned"):
            with self.subTest(kind=kind):
                record = score_sample(
                    sample(source_issue(kind)),
                    gold(),
                    [source_flag(kind, "another-object"), source_flag("loose_bbox"), source_flag(kind)],
                )
                self.assertEqual((record["tp"], record["fp"], record["fn"]), (1, 2, 0))
                self.assertEqual(record["matched_issues"][0]["pred_index"], 2)
                self.assertFalse(record["exact"])

    def test_missing_requires_class_and_spatial_overlap(self):
        document = gold({"label_id": "gold-a", "class_name": "vehicle.car", "bbox": box()})
        flags = [missing_flag(class_name="truck"), missing_flag(box(80, 0, 180, 100)), missing_flag(box(0, 0, 50, 100))]
        record = score_sample(sample(missing_issue()), document, flags)
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (1, 2, 0))
        self.assertEqual(record["matched_issues"][0]["pred_index"], 2)
        record = score_sample(sample(missing_issue()), document, flags, missing_iou=0.51)
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (0, 3, 1))

    def test_duplicate_requires_exact_pair_in_either_order(self):
        issue = {**source_issue("duplicate_label"), "duplicate_label_id": "source-copy"}
        flags = [
            source_flag("duplicate_label", evidence={"label_a": "source-a", "label_b": "unrelated"}),
            source_flag("duplicate_label", "source-copy", evidence={"label_a": "source-copy", "label_b": "source-a"}),
        ]
        record = score_sample(sample(issue), gold(), flags)
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (1, 1, 0))
        self.assertEqual(record["unmatched_prediction_indices"], [0])

    def test_repeated_flags_receive_only_one_true_positive(self):
        record = score_sample(sample(source_issue()), gold(), [source_flag(), source_flag()])
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (1, 1, 0))

    def test_one_prediction_cannot_cover_two_expected_issues(self):
        document = gold(
            {"label_id": "gold-a", "class_name": "car", "bbox": box()},
            {"label_id": "gold-b", "class_name": "car", "bbox": box(40, 0, 140, 100)},
        )
        record = score_sample(
            sample(missing_issue("gold-a"), missing_issue("gold-b")), document, [missing_flag(box(20, 0, 120, 100))]
        )
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (1, 0, 1))

    def test_maximum_cardinality_reassigns_earlier_match(self):
        document = gold(
            {"label_id": "gold-a", "class_name": "car", "bbox": box()},
            {"label_id": "gold-b", "class_name": "car", "bbox": box(40, 0, 140, 100)},
        )
        record = score_sample(
            sample(missing_issue("gold-a"), missing_issue("gold-b")),
            document,
            [missing_flag(box(20, 0, 120, 100)), missing_flag()],
        )
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (2, 0, 0))
        self.assertEqual({(m["pred_index"], m["gold_index"]) for m in record["matched_issues"]}, {(0, 1), (1, 0)})
        self.assertTrue(record["exact"])

    def test_clean_nonblocking_loose_box_remains_false_positive(self):
        record = score_sample(sample(), gold(), [source_flag("loose_bbox", blocking=False)])
        self.assertEqual((record["tp"], record["fp"], record["fn"]), (0, 1, 0))
        self.assertTrue(record["clean"])
        self.assertFalse(record["clean_pass"])
        self.assertFalse(record["exact"])

    def test_micro_aggregate_preserves_noise_and_provided_population(self):
        records = [
            score_sample(sample(source_issue()), gold(), [source_flag(), source_flag("loose_bbox", blocking=False)]),
            score_sample(sample(source_issue(), eligible=False), gold(), []),
            score_sample(sample(), gold(), []),
            score_sample(sample(), gold(), [source_flag("loose_bbox")]),
        ]
        result = summarize(iter(records))
        self.assertEqual((result["samples"], result["tp"], result["fp"], result["fn"]), (4, 1, 2, 1))
        self.assertEqual((result["expected_issues"], result["predicted_issues"]), (2, 3))
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.4)
        self.assertEqual(result["exact_sample_accuracy"], 0.25)
        self.assertEqual(result["false_flags_per_image"], 0.5)
        self.assertEqual(result["clean_pass_rate"], 0.5)
        self.assertEqual((result["exact_images"], result["clean_images"], result["clean_passes"]), (1, 2, 1))
        self.assertEqual(result["by_issue"]["wrong_class"]["recall"], 0.5)
        self.assertEqual(result["by_issue"]["loose_bbox"]["fp"], 2)
        self.assertIsNone(result["by_issue"]["loose_bbox"]["recall"])

    def test_undefined_rates_and_clean_success(self):
        empty = summarize([])
        for field in ("precision", "recall", "f1", "exact_sample_accuracy", "false_flags_per_image", "clean_pass_rate"):
            self.assertIsNone(empty[field])
        only_clean = summarize([score_sample(sample(), gold(), [])])
        self.assertEqual(only_clean["clean_pass_rate"], 1.0)
        self.assertEqual(only_clean["exact_sample_accuracy"], 1.0)
        self.assertIsNone(only_clean["precision"])
        self.assertIsNone(only_clean["recall"])
        missed = summarize([score_sample(sample(source_issue()), gold(), [])])
        self.assertIsNone(missed["precision"])
        self.assertEqual(missed["recall"], 0.0)
        self.assertEqual(missed["f1"], 0.0)
        self.assertIsNone(missed["clean_pass_rate"])

    def test_inputs_are_not_mutated(self):
        arguments = (sample(source_issue()), gold(), [source_flag()])
        original = copy.deepcopy(arguments)
        score_sample(*arguments)
        self.assertEqual(arguments, original)

    def test_missing_oracle_and_invalid_threshold_raise(self):
        with self.assertRaises(ValueError):
            score_sample(sample(missing_issue()), gold(), [])
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_sample(sample(), gold(), [], missing_iou=value)


if __name__ == "__main__":
    unittest.main()
