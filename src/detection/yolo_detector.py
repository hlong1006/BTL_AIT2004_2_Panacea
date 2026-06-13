from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class YoloDetector:
    # Improved defaults for better detection of small cells and reduced misclassification
    # - Lower confidence threshold catches more small cells (previously 0.12)
    # - Higher max_det for dense smears with many cells
    # - Multi-scale detection support for better recall on small objects
    DEFAULT_CONF = 0.08  # Lowered from 0.12 for better small cell detection
    DEFAULT_IOU = 0.40   # Slightly reduced for better separation
    DEFAULT_IMGSZ = 416
    DEFAULT_MAX_DET = 800  # Increased from 500 for dense smears

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        max_det: int = DEFAULT_MAX_DET,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.model = None
        self.class_names = {}

        if model_path is not None and model_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(model_path))
                self.class_names = self.model.names
            except Exception as exc:
                print(f"[WARN] Could not load YOLO model: {exc}")

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.model is None:
            h, w = image.shape[:2]
            return [
                Detection(
                    x1=0,
                    y1=0,
                    x2=w - 1,
                    y2=h - 1,
                    confidence=1.0,
                    class_id=0,
                    class_name="cell",
                )
            ]

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Calculate area and apply minimum size filter
                # This helps filter out noise while preserving small but real cells
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Minimum 3x3 px to be considered a real cell
                min_area = 9
                if area < min_area:
                    continue
                    
                detections.append(
                    Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=str(self.class_names.get(cls_id, str(cls_id))),
                    )
                )
        
        # Post-process: merge very nearby detections to reduce duplicates
        detections = self._merge_nearby_detections(detections)
        return detections
    
    @staticmethod
    def _merge_nearby_detections(detections: List[Detection], threshold: float = 0.3) -> List[Detection]:
        """
        Merge nearby detections that likely represent the same cell.
        Uses IoU (Intersection over Union) threshold.
        """
        if len(detections) <= 1:
            return detections
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Calculate IoU
                iou = YoloDetector._calculate_iou(det1, det2)
                if iou > threshold:
                    group.append(det2)
                    used.add(j)
            
            # Merge group by averaging coordinates (weighted by confidence)
            if len(group) > 1:
                merged_det = YoloDetector._average_detections(group)
            else:
                merged_det = group[0]
            
            merged.append(merged_det)
            used.add(i)
        
        return merged
    
    @staticmethod
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
        """Calculate Intersection over Union between two detections."""
        x1_min, y1_min, x1_max, y1_max = det1.x1, det1.y1, det1.x2, det1.y2
        x2_min, y2_min, x2_max, y2_max = det2.x1, det2.y1, det2.x2, det2.y2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Union
        det1_area = (x1_max - x1_min) * (y1_max - y1_min)
        det2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = det1_area + det2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def _average_detections(detections: List[Detection]) -> Detection:
        """Average coordinates and confidence of multiple detections."""
        total_conf = sum(d.confidence for d in detections)
        avg_conf = total_conf / len(detections)
        
        avg_x1 = sum(d.x1 for d in detections) / len(detections)
        avg_y1 = sum(d.y1 for d in detections) / len(detections)
        avg_x2 = sum(d.x2 for d in detections) / len(detections)
        avg_y2 = sum(d.y2 for d in detections) / len(detections)
        
        return Detection(
            x1=int(avg_x1),
            y1=int(avg_y1),
            x2=int(avg_x2),
            y2=int(avg_y2),
            confidence=avg_conf,
            class_id=detections[0].class_id,
            class_name=detections[0].class_name,
        )

    @staticmethod
    def crop_detection(image: np.ndarray, det: Detection) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, min(det.x1, w - 1))
        x2 = max(1, min(det.x2, w))
        y1 = max(0, min(det.y1, h - 1))
        y2 = max(1, min(det.y2, h))
        return image[y1:y2, x1:x2]

    # BGR — màu tương phản trên nền lam máu (hồng/nhạt)
    CLASS_COLORS_BGR = {
        "Platelets": (0, 165, 255),   # cam
        "RBC": (0, 0, 255),           # đỏ
        "WBC": (255, 80, 0),          # xanh dương đậm
        "cell": (0, 255, 255),        # vàng (fallback)
    }

    @staticmethod
    def _color_for_class(class_name: str) -> tuple[int, int, int]:
        return YoloDetector.CLASS_COLORS_BGR.get(class_name, (0, 255, 255))

    @staticmethod
    def _draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        top = max(th + 6, y)
        cv2.rectangle(image, (x, top - th - 8), (x + tw + 6, top + baseline + 2), color, -1)
        cv2.putText(image, text, (x + 3, top), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Detection], labels: Optional[List[str]] = None) -> np.ndarray:
        output = image.copy()
        for i, det in enumerate(detections):
            label = labels[i] if labels and i < len(labels) else det.class_name
            color = YoloDetector._color_for_class(label)
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 3)
            YoloDetector._draw_label(output, label, det.x1, max(0, det.y1 - 6), color)
        return output
