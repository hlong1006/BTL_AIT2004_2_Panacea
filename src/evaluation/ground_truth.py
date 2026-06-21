"""So sánh kết quả dự đoán với nhãn ground-truth YOLO."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.yolo_detector import Detection, YoloDetector

CLASS_NAMES = ["Platelets", "RBC", "WBC"]


def load_ground_truth_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[str, int, int, int, int]]:
    """Đọc file label YOLO và chuyển sang bbox pixel + tên lớp."""
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        boxes.append((name, x1, y1, x2, y2))
    return boxes


def find_label_file(image_path: Path) -> Optional[Path]:
    """Tìm file label tương ứng trong data/test, data/valid, data/train."""
    stem = image_path.stem
    for split in ("test", "valid", "train"):
        candidate = image_path.parent.parent.parent / split / "labels" / f"{stem}.txt"
        if candidate.exists():
            return candidate
        # image in test/images -> labels in test/labels
        candidate2 = image_path.parent.parent / "labels" / f"{stem}.txt"
        if candidate2.exists():
            return candidate2
    root = image_path
    for split in ("test", "valid", "train"):
        c = root.parents[1] / "data" / split / "labels" / f"{stem}.txt" if len(root.parents) > 1 else None
        if c and c.exists():
            return c
    from src.config.settings import PATHS
    for split in ("test", "valid", "train"):
        c = PATHS.root / "data" / split / "labels" / f"{stem}.txt"
        if c.exists():
            return c
    return None


def _iou_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_against_ground_truth(
    detections: List[Detection],
    predicted_labels: List[str],
    image_path: Path,
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.3,
) -> Optional[Dict]:
    """So khớp detection + ML label với GT, trả về precision/recall/F1."""
    label_path = find_label_file(image_path)
    if label_path is None:
        return None

    gt_boxes = load_ground_truth_labels(label_path, img_w, img_h)
    if not gt_boxes:
        return None

    matched_gt = set()
    tp = 0
    fp = 0
    class_tp: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    class_fp: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    class_fn: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

    for i, det in enumerate(detections):
        pred_label = predicted_labels[i] if i < len(predicted_labels) else det.class_name
        det_box = (det.x1, det.y1, det.x2, det.y2)
        best_iou, best_j = 0.0, -1
        for j, (gt_name, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = _iou_box(det_box, (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            gt_name = gt_boxes[best_j][0]
            if pred_label == gt_name:
                tp += 1
                class_tp[pred_label] = class_tp.get(pred_label, 0) + 1
            else:
                fp += 1
                class_fp[pred_label] = class_fp.get(pred_label, 0) + 1
                class_fn[gt_name] = class_fn.get(gt_name, 0) + 1
        else:
            fp += 1
            class_fp[pred_label] = class_fp.get(pred_label, 0) + 1

    for j, (gt_name, *_) in enumerate(gt_boxes):
        if j not in matched_gt:
            class_fn[gt_name] = class_fn.get(gt_name, 0) + 1

    fn = len(gt_boxes) - len(matched_gt) + sum(
        1 for i, det in enumerate(detections)
        if i < len(predicted_labels)
        and any(
            _iou_box((det.x1, det.y1, det.x2, det.y2), (g[1], g[2], g[3], g[4])) >= iou_threshold
            for k, g in enumerate(gt_boxes)
            if k in matched_gt and predicted_labels[i] != g[0]
        )
    )
    fn = max(0, len(gt_boxes) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class = {}
    for cls in CLASS_NAMES:
        ctp = class_tp.get(cls, 0)
        cfp = class_fp.get(cls, 0)
        cfn = class_fn.get(cls, 0)
        p = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        r = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    return {
        "has_ground_truth": True,
        "label_file": str(label_path),
        "gt_cells": len(gt_boxes),
        "detected_cells": len(detections),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "per_class": per_class,
    }
