from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector
from src.features.extractor import CellFeatureExtractor
from src.utils.io import ensure_dirs


class HybridCellPipeline:
    def __init__(self, yolo_model_path: Optional[Path], ml_model_path: Path):
        ensure_dirs([PATHS.interim_crops, PATHS.processed_features])
        self.detector = YoloDetector(yolo_model_path)
        self.extractor = CellFeatureExtractor()
        self.ml_checkpoint = MLClassifier.load_model(ml_model_path)

    def run_on_image(self, image_path: Path, output_image_path: Path, save_feature_csv: Optional[Path] = None) -> Path:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections = self.detector.detect(image)
        feature_rows = []
        for idx, det in enumerate(detections):
            crop = self.detector.crop_detection(image, det)
            crop_name = f"{image_path.stem}_cell_{idx:04d}.png"
            cv2.imwrite(str(PATHS.interim_crops / crop_name), crop)
            features = self.extractor.extract(crop)
            features["det_confidence"] = det.confidence
            feature_rows.append(features)

        labels = MLClassifier.predict(self.ml_checkpoint, feature_rows) if feature_rows else []
        rendered = self.detector.draw_detections(image, detections, labels)

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), rendered)

        if save_feature_csv is not None and feature_rows:
            df = pd.DataFrame(feature_rows)
            df["predicted_label"] = labels
            save_feature_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_feature_csv, index=False)

        return output_image_path
