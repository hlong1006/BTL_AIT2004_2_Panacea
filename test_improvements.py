#!/usr/bin/env python3
"""
Comparison script to test detection & classification improvements.
Run on test images to see before/after results.

Usage:
    python test_improvements.py --image sample.jpg
    python test_improvements.py --folder ./test_images
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detection.yolo_detector import YoloDetector
from src.features.extractor import CellFeatureExtractor
from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS


class ImprovementTester:
    """Test and compare detection/classification improvements."""
    
    def __init__(self, ml_model_path: Path):
        self.ml_model_path = ml_model_path
        self.feature_extractor = CellFeatureExtractor()
        self.ml_checkpoint = MLClassifier.load_model(ml_model_path)
    
    def test_detection(self, image_path: Path, yolo_path: Path) -> Tuple[int, int, float]:
        """
        Compare detection with old vs new parameters.
        
        Returns:
            (old_count, new_count, improvement_percent)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return 0, 0, 0.0
        
        # Old settings
        detector_old = YoloDetector(
            model_path=yolo_path,
            conf_threshold=0.12,  # Old default
            iou_threshold=0.45,   # Old default
            imgsz=416,
            max_det=500,          # Old default
        )
        
        # New settings
        detector_new = YoloDetector(
            model_path=yolo_path,
            conf_threshold=0.08,  # New default
            iou_threshold=0.40,   # New default
            imgsz=416,
            max_det=800,          # New default
        )
        
        detections_old = detector_old.detect(image)
        detections_new = detector_new.detect(image)
        
        old_count = len(detections_old)
        new_count = len(detections_new)
        
        if old_count == 0:
            improvement = 100.0 if new_count > 0 else 0.0
        else:
            improvement = ((new_count - old_count) / old_count) * 100
        
        return old_count, new_count, improvement
    
    def test_features(self, image_path: Path, yolo_path: Path) -> dict:
        """
        Test feature extraction improvements.
        Compare old features (10) vs new features (23).
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return {}
        
        detector = YoloDetector(model_path=yolo_path)
        detections = detector.detect(image)
        
        if not detections:
            return {"total_detections": 0}
        
        # Extract features from first detection
        det = detections[0]
        crop = detector.crop_detection(image, det)
        features = self.feature_extractor.extract(crop)
        
        old_feature_count = 10  # Original: area, perimeter, circularity, mean_b/g/r, mean_h/s/v, texture
        new_feature_count = len(features)
        
        return {
            "total_detections": len(detections),
            "old_features": old_feature_count,
            "new_features": new_feature_count,
            "new_features_added": new_feature_count - old_feature_count,
            "sample_features": {
                "eccentricity": features.get("eccentricity", 0),
                "solidity": features.get("solidity", 0),
                "extent": features.get("extent", 0),
                "hu_moment_1": features.get("hu_moment_1", 0),
            }
        }
    
    def test_classification_confidence(self, image_path: Path, yolo_path: Path) -> dict:
        """
        Test classification with confidence scoring.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return {}
        
        detector = YoloDetector(model_path=yolo_path)
        detections = detector.detect(image)
        
        if not detections:
            return {"total_detections": 0}
        
        # Extract features and predict with confidence
        feature_rows = []
        for det in detections[:5]:  # Sample first 5
            crop = detector.crop_detection(image, det)
            features = self.feature_extractor.extract(crop)
            feature_rows.append(features)
        
        # Old method (no confidence)
        predictions_old = MLClassifier.predict(self.ml_checkpoint, feature_rows)
        
        # New method (with confidence)
        predictions_new = MLClassifier.predict_with_confidence(
            self.ml_checkpoint, 
            feature_rows,
            confidence_threshold=0.70
        )
        
        return {
            "detections_sampled": len(feature_rows),
            "predictions_old": predictions_old,
            "predictions_new": [f"{p[0]} ({p[1]:.2%})" for p in predictions_new],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test blood cell detection/classification improvements"
    )
    parser.add_argument("--image", type=Path, help="Single image to test")
    parser.add_argument("--folder", type=Path, help="Folder of images to test")
    parser.add_argument("--no-display", action="store_true", help="Don't display results")
    
    args = parser.parse_args()
    
    # Validate models exist
    yolo_path = PATHS.yolo_models / "best.pt"
    ml_path = PATHS.ml_models / "best_ml_model.pt"
    
    if not ml_path.exists():
        print(f"❌ ML model not found: {ml_path}")
        return 1
    
    if not yolo_path.exists():
        print(f"⚠️  YOLO model not found: {yolo_path}")
        print("   Continuing with feature extraction tests only...")
    
    tester = ImprovementTester(ml_path)
    
    # Get images to test
    images: List[Path] = []
    if args.image:
        images = [args.image]
    elif args.folder:
        images = list(args.folder.glob("*.jpg")) + list(args.folder.glob("*.png"))
    else:
        print("❌ Provide --image or --folder")
        return 1
    
    if not images:
        print(f"❌ No images found")
        return 1
    
    print("\n" + "="*70)
    print("🔬 BLOOD CELL DETECTION & CLASSIFICATION IMPROVEMENTS TEST")
    print("="*70)
    
    total_old = 0
    total_new = 0
    
    for image_path in images:
        print(f"\n📊 Testing: {image_path.name}")
        print("-" * 70)
        
        if not image_path.exists():
            print(f"   ⚠️  File not found: {image_path}")
            continue
        
        # Test detection
        if yolo_path.exists():
            old_count, new_count, improvement = tester.test_detection(image_path, yolo_path)
            total_old += old_count
            total_new += new_count
            
            print(f"\n   🔍 DETECTION RESULTS:")
            print(f"      Old parameters (0.12 conf): {old_count} cells detected")
            print(f"      New parameters (0.08 conf): {new_count} cells detected")
            if old_count > 0:
                print(f"      Improvement: {improvement:+.1f}%")
        
        # Test features
        features_info = tester.test_features(image_path, yolo_path)
        if features_info:
            print(f"\n   ✨ FEATURE EXTRACTION:")
            print(f"      Cells detected: {features_info.get('total_detections', 0)}")
            print(f"      Old features: {features_info.get('old_features', 10)}")
            print(f"      New features: {features_info.get('new_features', 23)}")
            print(f"      New features added: {features_info.get('new_features_added', 13)}")
            
            sample = features_info.get("sample_features", {})
            if sample:
                print(f"\n      Sample new features (first detected cell):")
                print(f"         • Eccentricity: {sample.get('eccentricity', 0):.3f}")
                print(f"         • Solidity: {sample.get('solidity', 0):.3f}")
                print(f"         • Extent: {sample.get('extent', 0):.3f}")
                print(f"         • Hu Moment 1: {sample.get('hu_moment_1', 0):.6f}")
        
        # Test classification
        class_info = tester.test_classification_confidence(image_path, yolo_path)
        if class_info.get("detections_sampled", 0) > 0:
            print(f"\n   🏷️  CLASSIFICATION (Confidence Scoring):")
            print(f"      Cells analyzed: {class_info['detections_sampled']}")
            print(f"\n      Old method (no confidence):")
            for i, pred in enumerate(class_info.get("predictions_old", [])):
                print(f"         Cell {i}: {pred}")
            print(f"\n      New method (with confidence):")
            for i, pred in enumerate(class_info.get("predictions_new", [])):
                print(f"         Cell {i}: {pred}")
    
    # Summary
    if total_old > 0 or total_new > 0:
        print("\n" + "="*70)
        print("📈 SUMMARY")
        print("="*70)
        print(f"Total cells detected (old): {total_old}")
        print(f"Total cells detected (new): {total_new}")
        if total_old > 0:
            overall_improvement = ((total_new - total_old) / total_old) * 100
            print(f"Overall improvement: {overall_improvement:+.1f}%")
        
        print("\n✅ Key Improvements:")
        print("   • Lower confidence threshold (0.12 → 0.08)")
        print("   • Detection merging to eliminate duplicates")
        print("   • Max detections increased (500 → 800)")
        print("   • 13 new discriminative features added")
        print("   • Confidence-based classification scoring")
        print("   • Better SVM tuning for WBC/RBC separation")
    
    print("\n" + "="*70)
    print("📚 For details, see: IMPROVEMENTS_GUIDE.md")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
