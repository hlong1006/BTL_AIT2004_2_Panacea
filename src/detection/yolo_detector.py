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
    # Tối ưu cho tế bào nhỏ và ảnh mờ
    DEFAULT_CONF = 0.06
    DEFAULT_IOU = 0.45
    DEFAULT_IMGSZ = 640
    DEFAULT_MAX_DET = 1000
    # Bbox lớn hơn ngưỡng này (so với ảnh) thường là cụm nhiều tế bào → tái phát hiện trong vùng crop
    MAX_SINGLE_CELL_AREA_RATIO = 0.045

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

    @staticmethod
    def _enhance_for_detection(image: np.ndarray) -> np.ndarray:
        """Tăng tương phản cục bộ — giúp ảnh mờ/nhạt dễ phát hiện tế bào nhỏ hơn."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _predict_raw(self, image: np.ndarray, conf: Optional[float] = None, imgsz: Optional[int] = None) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(
            image,
            conf=conf if conf is not None else self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz if imgsz is not None else self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                if width * height < 9:
                    continue

                cls_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=float(box.conf[0].item()),
                        class_id=cls_id,
                        class_name=str(self.class_names.get(cls_id, str(cls_id))),
                    )
                )
        return detections

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

        # Phát hiện trên ảnh gốc + ảnh đã enhance, rồi gộp bằng NMS (không average bbox)
        detections = self._predict_raw(image)
        enhanced = self._enhance_for_detection(image)
        detections.extend(self._predict_raw(enhanced))

        detections = self._non_max_suppression(detections, iou_threshold=0.55)
        detections = self._refine_oversized_detections(image, detections)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _non_max_suppression(detections: List[Detection], iou_threshold: float = 0.55) -> List[Detection]:
        """Loại bbox trùng lặp thật sự — giữ bbox confidence cao, không gộp/average."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        for candidate in sorted_dets:
            suppress = False
            for kept_det in kept:
                if YoloDetector._calculate_iou(candidate, kept_det) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)

        return kept

    def _refine_oversized_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        max_area_ratio: float = MAX_SINGLE_CELL_AREA_RATIO,
    ) -> List[Detection]:
        """Tái phát hiện trong vùng bbox quá lớn (thường là cụm tế bào bị gom nhầm)."""
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        refined: List[Detection] = []

        for det in detections:
            det_w = det.x2 - det.x1
            det_h = det.y2 - det.y1
            det_area = det_w * det_h

            if det_area / img_area <= max_area_ratio:
                refined.append(det)
                continue

            crop = self.crop_detection(image, det)
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            refine_imgsz = min(640, max(crop_h, crop_w, 256))
            refine_conf = max(0.03, self.conf_threshold * 0.6)

            sub_dets = self._predict_raw(crop, conf=refine_conf, imgsz=refine_imgsz)
            sub_dets = self._non_max_suppression(sub_dets, iou_threshold=0.5)

            if len(sub_dets) >= 2:
                for sub in sub_dets:
                    sub_x1 = det.x1 + sub.x1
                    sub_y1 = det.y1 + sub.y1
                    sub_x2 = det.x1 + sub.x2
                    sub_y2 = det.y1 + sub.y2
                    refined.append(
                        Detection(
                            x1=min(sub_x1, img_w - 1),
                            y1=min(sub_y1, img_h - 1),
                            x2=min(sub_x2, img_w),
                            y2=min(sub_y2, img_h),
                            confidence=sub.confidence,
                            class_id=sub.class_id,
                            class_name=sub.class_name,
                        )
                    )
                continue

            # Thử lại trên ảnh đã enhance trong vùng crop
            enhanced_crop = self._enhance_for_detection(crop)
            sub_dets = self._predict_raw(enhanced_crop, conf=refine_conf, imgsz=refine_imgsz)
            sub_dets = self._non_max_suppression(sub_dets, iou_threshold=0.5)

            if len(sub_dets) >= 2:
                for sub in sub_dets:
                    refined.append(
                        Detection(
                            x1=min(det.x1 + sub.x1, img_w - 1),
                            y1=min(det.y1 + sub.y1, img_h - 1),
                            x2=min(det.x1 + sub.x2, img_w),
                            y2=min(det.y1 + sub.y2, img_h),
                            confidence=sub.confidence,
                            class_id=sub.class_id,
                            class_name=sub.class_name,
                        )
                    )
                continue

            # Không tách được → bỏ bbox cụm (tránh 1 nhãn ML sai cho cả cụm)
            if len(sub_dets) == 1:
                sub = sub_dets[0]
                sub_area = (sub.x2 - sub.x1) * (sub.y2 - sub.y1)
                if sub_area < det_area * 0.85:
                    refined.append(
                        Detection(
                            x1=min(det.x1 + sub.x1, img_w - 1),
                            y1=min(det.y1 + sub.y1, img_h - 1),
                            x2=min(det.x1 + sub.x2, img_w),
                            y2=min(det.y1 + sub.y2, img_h),
                            confidence=sub.confidence,
                            class_id=sub.class_id,
                            class_name=sub.class_name,
                        )
                    )
                    continue

            print(
                f"[WARN] Bỏ qua bbox cụm ({det_w}x{det_h}px, conf={det.confidence:.2f}) — "
                "không tách được thành tế bào riêng lẻ"
            )

        return refined

    @staticmethod
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
        inter_xmin = max(det1.x1, det2.x1)
        inter_ymin = max(det1.y1, det2.y1)
        inter_xmax = min(det1.x2, det2.x2)
        inter_ymax = min(det1.y2, det2.y2)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        det1_area = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        det2_area = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        union_area = det1_area + det2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def crop_detection(image: np.ndarray, det: Detection, padding_ratio: float = 0.05) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = int((det.x2 - det.x1) * padding_ratio)
        pad_y = int((det.y2 - det.y1) * padding_ratio)
        x1 = max(0, det.x1 - pad_x)
        x2 = min(w, det.x2 + pad_x)
        y1 = max(0, det.y1 - pad_y)
        y2 = min(h, det.y2 + pad_y)
        return image[y1:y2, x1:x2]

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
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            YoloDetector._draw_label(output, label, det.x1, max(0, det.y1 - 6), color)
        return output
