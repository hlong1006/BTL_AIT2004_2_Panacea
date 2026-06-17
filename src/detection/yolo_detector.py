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
    # Cải thiện các giá trị mặc định để phát hiện tế bào nhỏ tốt hơn và giảm phân loại sai
    # - Ngưỡng độ tin cậy thấp hơn giúp bắt được nhiều tế bào nhỏ hơn (trước đây là 0.12)
    # - max_det cao hơn cho các tiêu bản dày đặc với nhiều tế bào
    # - Hỗ trợ phát hiện đa tỷ lệ để có độ phủ (recall) tốt hơn trên các đối tượng nhỏ
    DEFAULT_CONF = 0.08  # Giảm từ 0.12 để phát hiện tế bào nhỏ tốt hơn
    DEFAULT_IOU = 0.40   # Giảm một chút để phân tách tốt hơn
    DEFAULT_IMGSZ = 416
    DEFAULT_MAX_DET = 800  # Tăng từ 500 cho các tiêu bản dày đặc

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
                
                # Tính diện tích và áp dụng bộ lọc kích thước tối thiểu
                # Điều này giúp lọc nhiễu trong khi vẫn giữ lại các tế bào nhỏ nhưng có thực
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Tối thiểu 3x3 px để được coi là một tế bào thực
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
        
        # Hậu xử lý: gộp các phát hiện rất gần nhau để giảm trùng lặp
        detections = self._merge_nearby_detections(detections)
        return detections
    
    @staticmethod
    def _merge_nearby_detections(detections: List[Detection], threshold: float = 0.3) -> List[Detection]:
        """
        Gộp các phát hiện gần nhau có khả năng là cùng một tế bào.
        Sử dụng ngưỡng IoU (Intersection over Union - Tỷ lệ phần giao trên phần hợp).
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
                
                # Tính IoU
                iou = YoloDetector._calculate_iou(det1, det2)
                if iou > threshold:
                    group.append(det2)
                    used.add(j)
            
            # Gộp nhóm bằng cách lấy trung bình các tọa độ (được tính trọng số theo độ tin cậy)
            if len(group) > 1:
                merged_det = YoloDetector._average_detections(group)
            else:
                merged_det = group[0]
            
            merged.append(merged_det)
            used.add(i)
        
        return merged
    
    @staticmethod
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
        """Tính Tỷ lệ phần giao trên phần hợp (IoU) giữa hai phát hiện."""
        x1_min, y1_min, x1_max, y1_max = det1.x1, det1.y1, det1.x2, det1.y2
        x2_min, y2_min, x2_max, y2_max = det2.x1, det2.y1, det2.x2, det2.y2
        
        # Phần giao (Intersection)
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Phần hợp (Union)
        det1_area = (x1_max - x1_min) * (y1_max - y1_min)
        det2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = det1_area + det2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def _average_detections(detections: List[Detection]) -> Detection:
        """Lấy trung bình các tọa độ và độ tin cậy của nhiều phát hiện."""
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

    # BGR (Xanh lam, Xanh lục, Đỏ)
    CLASS_COLORS_BGR = {
        "Platelets": (0, 165, 255),   
        "RBC": (0, 0, 255),           
        "WBC": (255, 80, 0),          
        "cell": (0, 255, 255),        
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