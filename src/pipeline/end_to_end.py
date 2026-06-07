from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import pandas as pd

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector
from src.features.extractor import CellFeatureExtractor
from src.utils.io import ensure_dirs
from src.utils.statistics import StatisticsCalculator, CellStatistics, ReportExporter


@dataclass
class PipelineOutput:
    """Container for pipeline execution results."""
    image_path: Path  # Output annotated image
    stats: CellStatistics  # Statistics object
    features_df: pd.DataFrame  # DataFrame with features and predictions
    report_text: str  # Text report


class HybridCellPipeline:
    def __init__(self, yolo_model_path: Optional[Path], ml_model_path: Path):
        ensure_dirs([PATHS.interim_crops, PATHS.processed_features])
        self.detector = YoloDetector(yolo_model_path)
        self.extractor = CellFeatureExtractor()
        self.ml_checkpoint = MLClassifier.load_model(ml_model_path)

    def run_on_image(
        self,
        image_path: Path,
        output_image_path: Path,
        save_feature_csv: Optional[Path] = None,
    ) -> Tuple[Path, pd.DataFrame, List[str]]:
        """
        Run pipeline on single image and return results.
        
        Returns:
            Tuple of (output_image_path, features_df, labels)
        """
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

        features_df = None
        if save_feature_csv is not None and feature_rows:
            features_df = pd.DataFrame(feature_rows)
            features_df["predicted_label"] = labels
            save_feature_csv.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(save_feature_csv, index=False)
        elif feature_rows:
            features_df = pd.DataFrame(feature_rows)
            features_df["predicted_label"] = labels

        return output_image_path, features_df, labels

    def run_on_image_full(
        self,
        image_path: Path,
        output_image_path: Path,
        output_stats_dir: Optional[Path] = None,
    ) -> PipelineOutput:
        """
        Run complete pipeline with full statistics and reporting.
        
        Args:
            image_path: Input microscopy image
            output_image_path: Path to save annotated image
            output_stats_dir: Directory to save reports (CSV, JSON, TXT)
        
        Returns:
            PipelineOutput with all results
        """
        # Run detection
        image_out_path, features_df, labels = self.run_on_image(
            image_path, output_image_path, save_feature_csv=None
        )
        
        # Calculate statistics
        stats = StatisticsCalculator.calculate_statistics(labels, features_df)
        
        # Generate text report
        report_text = StatisticsCalculator.generate_report_text(
            stats, image_name=image_path.stem
        )
        
        # Save reports if directory provided
        if output_stats_dir is not None:
            output_stats_dir.mkdir(parents=True, exist_ok=True)
            
            # Text report
            report_txt = output_stats_dir / f"{image_path.stem}_report.txt"
            ReportExporter.to_txt(report_text, str(report_txt))
            
            # CSV summary
            report_csv = output_stats_dir / f"{image_path.stem}_summary.csv"
            ReportExporter.to_csv(stats, str(report_csv))
            
            # JSON report
            report_json = output_stats_dir / f"{image_path.stem}_report.json"
            ReportExporter.to_json(stats, str(report_json))
            
            # Excel report (if available)
            try:
                report_excel = output_stats_dir / f"{image_path.stem}_report.xlsx"
                ReportExporter.to_excel(stats, features_df, str(report_excel))
            except Exception as e:
                print(f"[WARN] Could not export Excel: {e}")
            
            # Full features CSV
            features_csv = output_stats_dir / f"{image_path.stem}_features.csv"
            if features_df is not None:
                features_df.to_csv(str(features_csv), index=False)
        
        return PipelineOutput(
            image_path=image_out_path,
            stats=stats,
            features_df=features_df if features_df is not None else pd.DataFrame(),
            report_text=report_text,
        )
