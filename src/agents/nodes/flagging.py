from src.agents.geometry import iou
from src.agents.state import LabelQAState
from src.services.yolo import canonical_detection_class

# Prediction có confidence >= ngưỡng này mà không có label nào gần đó
# -> nghi ngờ mạnh là bị thiếu nhãn (annotator bỏ sót).
MISSING_LABEL_CONF_HIGH = 0.6
# Dưới ngưỡng này thì coi model cũng không đủ chắc chắn để nghi ngờ.
# Nâng từ 0.25 lên 0.4: đo lường trên golden nuImages cho thấy hầu hết flag
# missing_label dưới 0.4 là detector hallucinate/nhìn nhầm, không phải nhãn thiếu.
MISSING_LABEL_CONF_LOW = 0.4

# best_iou >= ngưỡng này giữa gt/pred không match được coi là "có liên hệ"
# (cùng một vật thể nhưng bbox lệch), thay vì "không liên quan gì nhau".
BBOX_MISALIGN_IOU_MIN = 0.1

# Hai gt_label cùng class chồng lên nhau với IoU >= ngưỡng này -> nghi trùng nhãn.
DUPLICATE_GT_IOU_THRESHOLD = 0.8

# Match có iou dưới ngưỡng này (dù đã pass IOU_MATCH_THRESHOLD và đúng class)
# vẫn coi là bbox vẽ lỏng, không khít quanh vật thể -> nghi vấn nhẹ.
LOOSE_BBOX_IOU_MAX = 0.85

# Diện tích GT (px^2) dưới ngưỡng này coi là "vật thể nhỏ" (theo quy ước COCO:
# small = area < 32*32). Với vật thể nhỏ, lệch vài pixel đã kéo IoU giảm mạnh
# hơn nhiều so với vật thể lớn cùng độ lệch tuyệt đối -> loose_bbox trên vật thể
# nhỏ thường là nhiễu do độ nhạy IoU, không hẳn là annotator vẽ ẩu. Issue vẫn
# được tạo ra (không bị xoá, vẫn "ghi nhận" để audit) nhưng đánh dấu
# blocking=False để không tự động đẩy status ảnh lên "needs_review".
SMALL_OBJECT_AREA_MAX = 32 * 32

# --- Cổng bằng chứng (evidence gates) -------------------------------------------------
# Nguyên tắc: detector chỉ là "nhân chứng", không phải ground truth. Chỉ buộc tội
# nhãn khi bằng chứng phản kháng đủ mạnh; thiếu bằng chứng (detector không thấy gì)
# với vật thể nhỏ/xa thường là do detector bỏ sót, không phải do nhãn sai.

# Buộc tội wrong_class cần detector tự tin tới mức này về class của nó.
# Dưới ngưỡng: match vẫn hợp lệ về vị trí, chỉ không đổ lỗi sai class.
WRONG_CLASS_CONF_MIN = 0.5

# Nhóm class "anh em" dễ nhầm lẫn nhau (car/truck/bus trong COCO thường confuse
# trên ảnh giao thông). Trong cùng nhóm chỉ buộc tội khi detector gần như chắc chắn.
WRONG_CLASS_SIBLING_GROUPS: tuple[frozenset[str], ...] = (frozenset({"car", "truck", "bus"}),)
WRONG_CLASS_SIBLING_CONF_MIN = 0.9

# Label không match pred nào và detector "không thấy gì gần đó" (best_iou thấp):
# chỉ buộc tội extra/wrong khi label đủ lớn để một detector hoạt động bình thường
# lẽ ra phải thấy. Vật thể nhỏ hơn diện tích này (theo tỉ lệ diện tích ảnh) thì
# việc detector không thấy là yếu tố dự báo kém — bỏ qua để không đổ lỗi oan.
EXTRA_LABEL_MIN_AREA_FRACTION = 0.005

# Tương tự cho bbox_misaligned: với vật thể rất nhỏ, IoU giữa bbox amodal (nuImages
# vẽ cả phần bị che) và bbox visible của detector dễ tụt xuống vùng "nghi lệch" dù
# nhãn đúng. Giá trị do sweep trên dev split chọn (2026-09): recall bbox giữ 4/6,
# FP/image giảm ~2/3 so với không cổng.
BBOX_MISALIGN_MIN_AREA_FRACTION = 0.005


def _area_fraction(bbox: dict | None, image_size: tuple[float, float] | None) -> float | None:
    """Diện tích bbox theo tỉ lệ diện tích ảnh; None khi không biết kích thước ảnh."""
    if not image_size or not isinstance(bbox, dict):
        return None
    width, height = image_size
    if not width or not height:
        return None
    area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
    return area / (width * height)


def _canonical_class(class_name: str) -> str:
    return canonical_detection_class(class_name) or class_name.strip().lower()


def _same_sibling_group(gt_class: str, pred_class: str) -> bool:
    pair = {_canonical_class(gt_class), _canonical_class(pred_class)}
    return any(pair <= group for group in WRONG_CLASS_SIBLING_GROUPS)


def flag_issues(
    matches: list[dict],
    unmatched_gt: list[dict],
    unmatched_pred: list[dict],
    gt_labels: list[dict],
    image_size: tuple[float, float] | None = None,
) -> list[dict]:
    """Áp dụng rule dựa trên số liệu matching để gắn cờ nghi vấn.

    Đây là bước quyết định (issue_type, severity) hoàn toàn bằng code dựa
    trên số liệu — LLM ở node sau chỉ diễn giải/đề xuất fix cho các issue
    đã được xác định ở đây, không được tự ý đổi loại lỗi hay mức độ.

    ``image_size`` (width, height) bật các cổng theo tỉ lệ diện tích; khi None
    (không biết kích thước ảnh) các cổng đó không áp dụng.

    Hàm thuần (không đụng LabelQAState) để dễ test độc lập — xem flag_issues_node bên dưới.
    """
    issues: list[dict] = []
    gt_by_id = {gt["label_id"]: gt for gt in gt_labels}
    matched_gt_ids = {m["gt_id"] for m in matches}

    # 0. Phát hiện duplicate trước (để bước 3 bên dưới biết bỏ qua box nào đã
    #    được giải thích bởi duplicate_label). Khi 2 box GT trùng lặp cùng 1 vật
    #    thể nhưng chỉ có 1 prediction, Hungarian matching chỉ match được 1 trong
    #    2 (tối đa hoá tổng IoU toàn cục, 1-1). Box duplicate còn lại thành
    #    unmatched_gt, và vì nó overlap cao với chính box đã match (cùng vật thể)
    #    nên best_iou của nó với prediction cũng cao -> bị bbox_misaligned gắn cờ
    #    thêm dù không hề lệch vị trí, chỉ là bản sao thừa. redundant_unmatched_ids
    #    đánh dấu các box này để bước 3 bỏ qua, tránh double-flag 1 lỗi thành 2.
    duplicate_issues: list[dict] = []
    redundant_unmatched_ids: set[str] = set()
    for i in range(len(gt_labels)):
        for j in range(i + 1, len(gt_labels)):
            a, b = gt_labels[i], gt_labels[j]
            if a["class_name"] != b["class_name"]:
                continue
            if iou(a["bbox"], b["bbox"]) < DUPLICATE_GT_IOU_THRESHOLD:
                continue
            duplicate_issues.append(
                {
                    "label_id": a["label_id"],
                    "issue_type": "duplicate_label",
                    "severity": "medium",
                    "evidence": {"label_a": a["label_id"], "label_b": b["label_id"]},
                    "blocking": True,
                }
            )
            a_matched, b_matched = a["label_id"] in matched_gt_ids, b["label_id"] in matched_gt_ids
            if a_matched and not b_matched:
                redundant_unmatched_ids.add(b["label_id"])
            elif b_matched and not a_matched:
                redundant_unmatched_ids.add(a["label_id"])

    # 1. Khớp vị trí tốt nhưng sai class — chỉ buộc tội khi detector đủ tự tin;
    #    trong nhóm class dễ nhầm (car/truck/bus) thì cần gần như chắc chắn.
    for m in matches:
        if not m["class_match"]:
            confidence = m.get("pred_confidence") or 0.0
            bar = WRONG_CLASS_SIBLING_CONF_MIN if _same_sibling_group(m["gt_class"], m["pred_class"]) else WRONG_CLASS_CONF_MIN
            if confidence < bar:
                continue  # detector không đủ chắc để đổ lỗi sai class; match vị trí vẫn hợp lệ
            issues.append(
                {
                    "label_id": m["gt_id"],
                    "issue_type": "wrong_class",
                    "severity": "high" if m["iou"] >= 0.85 else "medium",
                    "evidence": m,
                    "blocking": True,
                }
            )
        elif m["iou"] < LOOSE_BBOX_IOU_MAX:
            # Match hợp lệ, đúng class, nhưng bbox không khít quanh vật thể
            # (vd GT vẽ lỏng/thừa nền) -> vẫn đáng nghi dù không đủ nặng để
            # tính là bbox_misaligned (case đó dành cho match thất bại hẳn).
            gt = gt_by_id.get(m["gt_id"])
            bbox = gt["bbox"] if gt else None
            area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]) if bbox else None
            is_small = area is not None and area < SMALL_OBJECT_AREA_MAX
            issues.append(
                {
                    "label_id": m["gt_id"],
                    "issue_type": "loose_bbox",
                    "severity": "low",
                    "evidence": m,
                    # Vật thể nhỏ: vẫn ghi nhận issue nhưng không chặn status
                    # "needs_review" (xem SMALL_OBJECT_AREA_MAX ở trên).
                    "blocking": not is_small,
                }
            )

    # 2. Model phát hiện vật thể tự tin nhưng không có label nào gần đó -> nghi thiếu nhãn
    for pred in unmatched_pred:
        if pred["best_iou"] >= BBOX_MISALIGN_IOU_MIN:
            continue  # đã có gt gần đó, xử lý ở nhánh bbox_misaligned từ phía gt
        if pred["confidence"] >= MISSING_LABEL_CONF_HIGH:
            severity = "high"
        elif pred["confidence"] >= MISSING_LABEL_CONF_LOW:
            severity = "low"
        else:
            continue
        issues.append(
            {
                "label_id": None,
                "issue_type": "missing_label",
                "severity": severity,
                "evidence": pred,
                # Band "low" (confidence vừa phải) chỉ là gợi ý xem lại,
                # không đủ mạnh để tự động đẩy ảnh lên needs_review.
                "blocking": severity == "high",
            }
        )

    # 3. Label không khớp bất kỳ prediction nào — chỉ buộc tội khi label đủ lớn
    #    để detector lẽ ra phải thấy/cần khít; vật thể nhỏ thì "detector không
    #    thấy" / "IoU thấp" là bằng chứng yếu, không đổ lỗi oan.
    for gt in unmatched_gt:
        if gt["label_id"] in redundant_unmatched_ids:
            continue  # đã giải thích bằng duplicate_label ở bước 0, không double-flag
        area_fraction = _area_fraction(gt.get("bbox"), image_size)
        if gt["best_iou"] >= BBOX_MISALIGN_IOU_MIN:
            if area_fraction is not None and area_fraction < BBOX_MISALIGN_MIN_AREA_FRACTION:
                continue
            issues.append(
                {
                    "label_id": gt["label_id"],
                    "issue_type": "bbox_misaligned",
                    "severity": "medium",
                    "evidence": gt,
                    "blocking": True,
                }
            )
        else:
            if area_fraction is not None and area_fraction < EXTRA_LABEL_MIN_AREA_FRACTION:
                continue
            issues.append(
                {
                    "label_id": gt["label_id"],
                    "issue_type": "extra_or_wrong_label",
                    "severity": "medium",
                    "evidence": gt,
                    "blocking": True,
                }
            )

    # 4. Hai gt_label trùng lặp (cùng class, overlap gần như hoàn toàn) — đã tính ở bước 0
    issues += duplicate_issues

    return issues


async def flag_issues_node(state: LabelQAState) -> dict:
    matches = state.get("matches", [])
    unmatched_gt = state.get("unmatched_gt", [])
    unmatched_pred = state.get("unmatched_pred", [])
    gt_labels = state.get("gt_labels", [])
    label_scope = (state.get("metadata") or {}).get("label_scope") or {}
    metrics = state.get("metrics") or {}
    width = label_scope.get("image_width") or metrics.get("image_width")
    height = label_scope.get("image_height") or metrics.get("image_height")
    image_size = (width, height) if width and height else None
    return {"flagged_issues": flag_issues(matches, unmatched_gt, unmatched_pred, gt_labels, image_size=image_size)}
