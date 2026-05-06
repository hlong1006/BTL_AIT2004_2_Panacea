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
    def __init__(self, model_path: Optional[Path] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
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

        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
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
        return detections

    @staticmethod
    def crop_detection(image: np.ndarray, det: Detection) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, min(det.x1, w - 1))
        x2 = max(1, min(det.x2, w))
        y1 = max(0, min(det.y1, h - 1))
        y2 = max(1, min(det.y2, h))
        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Detection], labels: Optional[List[str]] = None) -> np.ndarray:
        output = image.copy()
        for i, det in enumerate(detections):
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            text = det.class_name
            if labels and i < len(labels):
                text = f"{text} | {labels[i]}"
            cv2.putText(output, text, (det.x1, max(0, det.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output
