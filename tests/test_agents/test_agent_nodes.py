from pathlib import Path

import pytest
from PIL import Image

from src.agents.geometry import iou
from src.agents.nodes.flagging import flag_issues, flag_issues_node
from src.agents.nodes.load_gt_labels import load_gt_labels_node
from src.agents.nodes.matching import match_labels_node
from src.agents.nodes.metrics import compute_metrics_node
from src.agents.nodes.validate_input import validate_input_node
from src.agents.nodes.yolo_inference import run_yolo_inference_node


def test_iou_handles_overlap_and_disjoint_boxes():
    first = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}
    overlap = {"x1": 5.0, "y1": 5.0, "x2": 15.0, "y2": 15.0}
    disjoint = {"x1": 20.0, "y1": 20.0, "x2": 30.0, "y2": 30.0}

    assert iou(first, overlap) == pytest.approx(25 / 175)
    assert iou(first, disjoint) == 0.0


@pytest.mark.asyncio
async def test_loads_yolo_ground_truth_from_sibling_labels_directory(tmp_path: Path):
    image_path = tmp_path / "images" / "frame.png"
    label_path = tmp_path / "labels" / "frame.txt"
    image_path.parent.mkdir()
    label_path.parent.mkdir()
    Image.new("RGB", (100, 80)).save(image_path)
    label_path.write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    (label_path.parent / "classes.txt").write_text("car\n", encoding="utf-8")

    result = await load_gt_labels_node({"image_path": str(image_path)})

    assert result["gt_labels"] == [{"class_name": "car", "bbox": {"x1": 40.0, "y1": 30.0, "x2": 60.0, "y2": 50.0}}]


@pytest.mark.asyncio
async def test_loads_class_names_from_export_root_with_nonstandard_filename(tmp_path: Path):
    image_path = tmp_path / "images" / "train" / "frame.png"
    label_path = tmp_path / "labels" / "train" / "frame.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image_path)
    label_path.write_text("1 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    (tmp_path / "class.txt.txt").write_text("car\npedestrian\n", encoding="utf-8")

    result = await load_gt_labels_node({"image_path": str(image_path)})

    assert result["gt_labels"][0]["class_name"] == "pedestrian"


@pytest.mark.asyncio
async def test_matching_metrics_and_flagging_preserve_rule_based_decisions():
    gt_labels = [{"label_id": "gt-1", "class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    pred_labels = [
        {
            "class_name": "truck",
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "confidence": 0.9,
        }
    ]

    matched = await match_labels_node({"gt_labels": gt_labels, "pred_labels": pred_labels})
    metrics = await compute_metrics_node(matched)
    flagged = await flag_issues_node({**matched, **metrics, "gt_labels": gt_labels})

    assert metrics["metrics"]["class_accuracy"] == 0.0
    assert flagged["flagged_issues"][0]["issue_type"] == "wrong_class"
    assert flagged["flagged_issues"][0]["severity"] == "high"


@pytest.mark.asyncio
async def test_wrong_class_below_confidence_floor_is_not_accused():
    gt_labels = [{"label_id": "gt-1", "class_name": "pedestrian", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    pred_labels = [
        {"class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": 0.4}
    ]

    matched = await match_labels_node({"gt_labels": gt_labels, "pred_labels": pred_labels})
    metrics = await compute_metrics_node(matched)
    flagged = await flag_issues_node({**matched, **metrics, "gt_labels": gt_labels})

    # Vị trí khớp hoàn hảo nhưng detector chỉ tự tin 0.4 rằng đó là car —
    # không đủ bằng chứng để buộc tội sai class.
    assert flagged["flagged_issues"] == []


@pytest.mark.asyncio
async def test_sibling_vehicle_confusion_needs_near_certainty():
    gt_labels = [{"label_id": "gt-1", "class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    pred_labels = [
        {"class_name": "truck", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": 0.7}
    ]

    matched = await match_labels_node({"gt_labels": gt_labels, "pred_labels": pred_labels})
    metrics = await compute_metrics_node(matched)
    flagged = await flag_issues_node({**matched, **metrics, "gt_labels": gt_labels})

    # car/truck là nhóm dễ nhầm lẫn: 0.7 vẫn chưa đủ để buộc tội.
    assert flagged["flagged_issues"] == []


def test_missing_label_low_band_is_advisory_only():
    unmatched_pred = [
        {"class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, "confidence": 0.5,
         "best_iou": 0.0, "prediction_index": 0}
    ]

    issues = flag_issues([], [], unmatched_pred, [])

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "missing_label"
    assert issues[0]["severity"] == "low"
    assert issues[0]["blocking"] is False


def test_missing_label_below_low_band_is_not_flagged():
    unmatched_pred = [
        {"class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, "confidence": 0.3,
         "best_iou": 0.0, "prediction_index": 0}
    ]

    assert flag_issues([], [], unmatched_pred, []) == []


def test_extra_label_area_gate_skips_tiny_labels():
    tiny = {"label_id": "gt-1", "class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, "best_iou": 0.0}
    large = {"label_id": "gt-2", "class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}, "best_iou": 0.0}

    # 20x20 trên ảnh 1600x900 (~0.03% diện tích): detector bỏ sót vật thể nhỏ
    # là chuyện thường, không đủ bằng chứng bỏ tội "nhãn thừa".
    assert flag_issues([], [tiny], [], [tiny], image_size=(1600, 900)) == []
    # 100x100 (~0.7% diện tích): đủ lớn để nghi ngờ.
    issues = flag_issues([], [large], [], [large], image_size=(1600, 900))
    assert [issue["issue_type"] for issue in issues] == ["extra_or_wrong_label"]


@pytest.mark.asyncio
async def test_flagging_node_uses_image_size_from_label_scope():
    tiny = [{"label_id": "gt-1", "class_name": "car", "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 10}}]
    state = {
        "matches": [],
        "unmatched_gt": [{**tiny[0], "best_iou": 0.0}],
        "unmatched_pred": [],
        "gt_labels": tiny,
        "metadata": {"label_scope": {"image_width": 1600, "image_height": 900}},
    }

    flagged = await flag_issues_node(state)

    assert flagged["flagged_issues"] == []


@pytest.mark.asyncio
async def test_matching_normalizes_nuscenes_and_kitti_taxonomies():
    gt_labels = [
        {
            "label_id": "gt-vehicle",
            "class_name": "vehicle.car",
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        }
    ]
    pred_labels = [
        {
            "class_name": "car",
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "confidence": 0.9,
        }
    ]

    matched = await match_labels_node({"gt_labels": gt_labels, "pred_labels": pred_labels})

    assert matched["matches"][0]["class_match"] is True


@pytest.mark.asyncio
async def test_validation_excludes_outside_labels_and_clips_partial_boxes(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (100, 80)).save(image_path)
    result = await validate_input_node(
        {
            "image_path": str(image_path),
            "gt_labels": [
                {"label_id": "outside", "class_name": "car", "bbox": {"x1": -40, "y1": 10, "x2": -10, "y2": 30}},
                {"label_id": "partial", "class_name": "car", "bbox": {"x1": -10, "y1": 10, "x2": 20, "y2": 30}},
            ],
            "pred_labels": [
                {"class_name": "car", "bbox": {"x1": 110, "y1": 10, "x2": 130, "y2": 30}, "confidence": 0.9}
            ],
        }
    )

    assert [label["label_id"] for label in result["gt_labels"]] == ["partial"]
    assert result["gt_labels"][0]["bbox"] == {"x1": 0.0, "y1": 10.0, "x2": 20.0, "y2": 30.0}
    assert result["gt_labels"][0]["source_bbox"]["x1"] == -10.0
    assert result["pred_labels"] == []
    assert result["metadata"]["label_scope"]["ground_truth_excluded_outside"] == 1
    assert result["metadata"]["label_scope"]["ground_truth_clipped"] == 1


@pytest.mark.asyncio
async def test_yolo_is_skipped_when_predictions_are_supplied(monkeypatch: pytest.MonkeyPatch):
    def fail_if_loaded():
        raise AssertionError("YOLO model must not load when predictions are supplied")

    monkeypatch.setattr("src.agents.nodes.yolo_inference.get_yolo_model", fail_if_loaded)
    result = await run_yolo_inference_node({"image_path": "fixture.png", "pred_labels": []})

    assert result == {}


@pytest.mark.asyncio
async def test_yolo_load_failure_becomes_pipeline_error(monkeypatch: pytest.MonkeyPatch):
    def fail_to_load():
        raise ModuleNotFoundError("ultralytics is not installed")

    monkeypatch.setattr("src.agents.nodes.yolo_inference.get_yolo_model", fail_to_load)
    result = await run_yolo_inference_node({"image_path": "fixture.png"})

    assert "ultralytics is not installed" in result["error"]
