from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd

from src.calibration.cbc_estimator import CBCEstimate, CBCEstimator
from src.classification.ml_classifier import MLClassifier
from src.classification.wbc_subtype import WBCSubtypeEstimator
from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector
from src.diagnosis.anomaly_detector import AnomalyDetector
from src.diagnosis.blood_health_analyzer import BloodHealthAnalyzer, BloodHealthReport
from src.evaluation.ground_truth import evaluate_against_ground_truth
from src.evaluation.hybrid_comparison import compare_yolo_vs_hybrid
from src.explainability.feature_explainer import FeatureExplainer
from src.features.extractor import CellFeatureExtractor
from src.llm.report_explainer import ReportExplainer
from src.reporting.pdf_exporter import PDFExporter
from src.utils.io import ensure_dirs
from src.utils.statistics import StatisticsCalculator, CellStatistics, ReportExporter


@dataclass
class PipelineOutput:
    image_path: Path
    stats: CellStatistics
    features_df: pd.DataFrame
    report_text: str
    health_report: Optional[BloodHealthReport] = None
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    per_cell_confidence: List[Dict] = field(default_factory=list)
    anomalies: List[Dict] = field(default_factory=list)
    explanation: Optional[Dict] = None
    global_importance: Optional[Dict] = None
    cbc_estimate: Optional[CBCEstimate] = None
    hybrid_comparison: Optional[Dict] = None
    ground_truth_metrics: Optional[Dict] = None
    plain_language_summary: Optional[str] = None
    wbc_subtypes: List[Dict] = field(default_factory=list)
    detections_count: int = 0


class HybridCellPipeline:
    def __init__(
        self,
        yolo_model_path: Optional[Path],
        ml_model_path: Path,
        yolo_conf: float = YoloDetector.DEFAULT_CONF,
        yolo_iou: float = YoloDetector.DEFAULT_IOU,
        yolo_imgsz: int = YoloDetector.DEFAULT_IMGSZ,
        yolo_max_det: int = YoloDetector.DEFAULT_MAX_DET,
    ):
        ensure_dirs([PATHS.interim_crops, PATHS.processed_features])
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.yolo_imgsz = yolo_imgsz
        self.yolo_max_det = yolo_max_det
        self.detector = YoloDetector(
            yolo_model_path,
            conf_threshold=yolo_conf,
            iou_threshold=yolo_iou,
            imgsz=yolo_imgsz,
            max_det=yolo_max_det,
        )
        self.extractor = CellFeatureExtractor()
        self.ml_checkpoint = MLClassifier.load_model(ml_model_path)
        self.anomaly_detector = AnomalyDetector()
        self.health_analyzer = BloodHealthAnalyzer()
        self.cbc_estimator = CBCEstimator()
        self.wbc_estimator = WBCSubtypeEstimator()
        self.report_explainer = ReportExplainer()

    def set_conf_threshold(self, conf: float) -> None:
        self.yolo_conf = conf
        self.detector.conf_threshold = conf

    def run_on_image(
        self,
        image_path: Path,
        output_image_path: Path,
        save_feature_csv: Optional[Path] = None,
        ml_confidence_threshold: float = 0.0,
    ) -> Tuple[Path, pd.DataFrame, List[str], List[Dict], list]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

        detections = self.detector.detect(image)
        feature_rows = []
        for idx, det in enumerate(detections):
            crop = self.detector.crop_detection(image, det)
            crop_name = f"{image_path.stem}_cell_{idx:04d}.png"
            cv2.imwrite(str(PATHS.interim_crops / crop_name), crop)
            features = self.extractor.extract(crop)
            features["det_confidence"] = det.confidence
            feature_rows.append(features)

        per_cell_conf = []
        if feature_rows:
            preds = MLClassifier.predict_with_confidence(
                self.ml_checkpoint, feature_rows, ml_confidence_threshold
            )
            labels = [p[0].rstrip("?") for p in preds]
            per_cell_conf = [
                {"index": i, "label": labels[i], "ml_confidence": round(preds[i][1], 4),
                 "yolo_confidence": round(detections[i].confidence, 4)}
                for i in range(len(labels))
            ]
        else:
            labels = []

        rendered = self.detector.draw_detections(image, detections, labels)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), rendered)

        features_df = None
        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            features_df["predicted_label"] = labels
            if per_cell_conf:
                features_df["ml_confidence"] = [c["ml_confidence"] for c in per_cell_conf]
            if save_feature_csv is not None:
                save_feature_csv.parent.mkdir(parents=True, exist_ok=True)
                features_df.to_csv(save_feature_csv, index=False)

        return output_image_path, features_df, labels, per_cell_conf, detections

    def run_on_image_full(
        self,
        image_path: Path,
        output_image_path: Path,
        output_stats_dir: Optional[Path] = None,
        ml_confidence_threshold: float = 0.0,
        include_ground_truth: bool = True,
    ) -> PipelineOutput:
        image_out_path, features_df, labels, per_cell_conf, detections = self.run_on_image(
            image_path, output_image_path, save_feature_csv=None,
            ml_confidence_threshold=ml_confidence_threshold,
        )

        stats = StatisticsCalculator.calculate_statistics(
            labels, features_df if features_df is not None else pd.DataFrame()
        )

        # Confidence stats per class
        confidence_stats = {}
        if features_df is not None and "ml_confidence" in features_df.columns:
            for ct in stats.cell_counts:
                mask = features_df["predicted_label"] == ct
                confs = features_df.loc[mask, "ml_confidence"]
                if len(confs) > 0:
                    confidence_stats[ct] = {
                        "avg": round(float(confs.mean()), 4),
                        "min": round(float(confs.min()), 4),
                        "low_confidence_cells": int((confs < 0.6).sum()),
                    }

        health_report = self.health_analyzer.analyze(
            cell_counts=stats.cell_counts,
            cell_percentages=stats.cell_percentages,
            total_cells=stats.total_cells,
            features_df=features_df,
        )

        anomalies_raw = self.anomaly_detector.detect(
            features_df if features_df is not None else pd.DataFrame(), labels
        )
        anomalies = self.anomaly_detector.to_dict_list(anomalies_raw)
        if anomalies:
            stats.add_warning(self.anomaly_detector.summary_text(anomalies_raw)[:200])

        explanation = None
        global_importance = None
        if feature_rows := (features_df.to_dict("records") if features_df is not None and len(features_df) > 0 else []):
            explanation = FeatureExplainer.explain_cell(self.ml_checkpoint, feature_rows[0])
            if features_df is not None and len(features_df) >= 10:
                global_importance = FeatureExplainer.global_importance(self.ml_checkpoint, features_df)

        cbc_estimate = self.cbc_estimator.estimate(stats.cell_counts)

        image = cv2.imread(str(image_path))
        h, w = image.shape[:2] if image is not None else (0, 0)
        hybrid_comparison = compare_yolo_vs_hybrid(detections, labels)

        ground_truth_metrics = None
        if include_ground_truth and image is not None:
            ground_truth_metrics = evaluate_against_ground_truth(
                detections, labels, image_path, w, h
            )

        wbc_subtypes = []
        if features_df is not None:
            wbc_subtypes = self.wbc_estimator.estimate_from_features(features_df, labels)

        report_text = StatisticsCalculator.generate_report_text(stats, image_name=image_path.stem)
        plain_language = self.report_explainer.explain(
            report_text, health_report.summary, stats.cell_counts, stats.cell_percentages
        )

        if output_stats_dir is not None:
            output_stats_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem

            ReportExporter.to_txt(report_text, str(output_stats_dir / f"{stem}_report.txt"))
            with open(output_stats_dir / f"{stem}_health_diagnosis.txt", "w", encoding="utf-8") as f:
                f.write(health_report.to_text())
            ReportExporter.to_csv(stats, str(output_stats_dir / f"{stem}_summary.csv"))
            ReportExporter.to_json(stats, str(output_stats_dir / f"{stem}_report.json"))

            if features_df is not None:
                features_df.to_csv(str(output_stats_dir / f"{stem}_features.csv"), index=False)

            with open(output_stats_dir / f"{stem}_plain_language.txt", "w", encoding="utf-8") as f:
                f.write(plain_language)

            try:
                ReportExporter.to_excel(stats, features_df, str(output_stats_dir / f"{stem}_report.xlsx"))
            except Exception as exc:
                print(f"[WARN] Excel export: {exc}")

            PDFExporter.export(
                output_stats_dir / f"{stem}_report.pdf",
                image_name=image_path.name,
                stats=stats.to_dict(),
                health_report_text=health_report.to_text(),
                plain_language=plain_language,
                analyzed_image_path=image_out_path,
            )

        return PipelineOutput(
            image_path=image_out_path,
            stats=stats,
            features_df=features_df if features_df is not None else pd.DataFrame(),
            report_text=report_text,
            health_report=health_report,
            confidence_stats=confidence_stats,
            per_cell_confidence=per_cell_conf,
            anomalies=anomalies,
            explanation=explanation,
            global_importance=global_importance,
            cbc_estimate=cbc_estimate,
            hybrid_comparison=hybrid_comparison,
            ground_truth_metrics=ground_truth_metrics,
            plain_language_summary=plain_language,
            wbc_subtypes=wbc_subtypes,
            detections_count=len(labels),
        )
