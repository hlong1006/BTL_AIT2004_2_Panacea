"""Unit tests cho các module Panacea."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnosis.anomaly_detector import AnomalyDetector
from src.diagnosis.blood_health_analyzer import BloodHealthAnalyzer
from src.calibration.cbc_estimator import CBCEstimator
from src.evaluation.hybrid_comparison import compare_yolo_vs_hybrid
from src.detection.yolo_detector import Detection


class TestBloodHealthAnalyzer(unittest.TestCase):
    def test_normal_percentages(self):
        analyzer = BloodHealthAnalyzer()
        report = analyzer.analyze(
            {"RBC": 450, "WBC": 20, "Platelets": 30},
            {"RBC": 90.0, "WBC": 4.0, "Platelets": 6.0},
            500,
        )
        self.assertEqual(report.overall_status.value, "Bình thường")

    def test_low_rbc_percentage(self):
        analyzer = BloodHealthAnalyzer()
        report = analyzer.analyze(
            {"RBC": 50, "WBC": 200, "Platelets": 50},
            {"RBC": 16.7, "WBC": 66.7, "Platelets": 16.7},
            300,
        )
        self.assertTrue(len(report.findings) > 0)

    def test_no_cells(self):
        analyzer = BloodHealthAnalyzer()
        report = analyzer.analyze({}, {}, 0)
        self.assertEqual(report.overall_status.value, "Nguy hiểm")


class TestAnomalyDetector(unittest.TestCase):
    def test_detect_anomaly(self):
        rows = []
        for i in range(5):
            rows.append({"area": 800 + i, "perimeter": 100, "circularity": 0.9,
                         "eccentricity": 0.2, "solidity": 0.9, "texture_laplacian_var": 50})
        rows.append({"area": 8000, "perimeter": 400, "circularity": 0.3,
                     "eccentricity": 0.9, "solidity": 0.5, "texture_laplacian_var": 200})
        df = pd.DataFrame(rows)
        df["predicted_label"] = "RBC"
        detector = AnomalyDetector()
        anomalies = detector.detect(df, ["RBC"] * 6, z_threshold=2.0)
        self.assertTrue(len(anomalies) >= 1)


class TestHybridComparison(unittest.TestCase):
    def test_agreement(self):
        dets = [
            Detection(0, 0, 10, 10, 0.9, 1, "RBC"),
            Detection(20, 20, 30, 30, 0.8, 2, "WBC"),
        ]
        result = compare_yolo_vs_hybrid(dets, ["RBC", "Platelets"])
        self.assertEqual(result["disagreements"], 1)
        self.assertEqual(result["agreements"], 1)


class TestCBCEstimator(unittest.TestCase):
    def test_estimate(self):
        est = CBCEstimator().estimate({"RBC": 450, "WBC": 50, "Platelets": 200})
        self.assertIsNotNone(est.rbc_per_ul)
        self.assertIsNotNone(est.wbc_per_ul)


if __name__ == "__main__":
    unittest.main()
