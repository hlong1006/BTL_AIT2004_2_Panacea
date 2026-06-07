#!/usr/bin/env python3
"""
Blood Cell Detection & Classification System
Main Application Interface

This application provides an easy-to-use interface for blood cell analysis
combining YOLOv8 detection with Machine Learning classification.

Usage:
    python app.py --image sample.png
    python app.py --folder ./images
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline
from src.utils.statistics import StatisticsCalculator, ReportExporter


class BloodCellAnalysisApp:
    """Main application for blood cell analysis."""
    
    def __init__(
        self,
        yolo_model_path: Optional[Path] = None,
        ml_model_path: Optional[Path] = None,
        verbose: bool = False,
    ):
        """Initialize the application."""
        self.verbose = verbose
        
        # Set defaults
        if yolo_model_path is None:
            yolo_model_path = PATHS.yolo_models / "best.pt"
        if ml_model_path is None:
            ml_model_path = PATHS.ml_models / ML_MODEL_FILENAME
        
        # Validate paths
        if not ml_model_path.exists():
            raise FileNotFoundError(f"ML model not found: {ml_model_path}")
        
        # Adjust YOLO path if it doesn't exist
        yolo_exists = yolo_model_path.exists()
        
        if self.verbose:
            print("📦 Initializing Blood Cell Analysis System...")
            print(f"   ML Model: {ml_model_path}")
            if yolo_exists:
                print(f"   YOLO Model: {yolo_model_path}")
            else:
                print(f"   ⚠️  YOLO Model not found (will use fallback)")
        
        # Initialize pipeline
        yolo_to_load = yolo_model_path if yolo_exists else None
        self.pipeline = HybridCellPipeline(
            yolo_model_path=yolo_to_load,
            ml_model_path=ml_model_path,
        )
    
    def analyze_image(
        self,
        image_path: Path,
        output_dir: Path,
        save_reports: bool = True,
    ) -> dict:
        """
        Analyze a single blood sample image.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save results
            save_reports: Whether to save detailed reports
        
        Returns:
            Dictionary with analysis results
        """
        # Validate input
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if self.verbose:
            print(f"\n🔍 Analyzing: {image_path.name}")
        
        # Create output paths
        output_image_path = output_dir / "images" / f"{image_path.stem}_annotated.png"
        output_reports_dir = (output_dir / "reports" / image_path.stem) if save_reports else None
        
        # Run pipeline
        result = self.pipeline.run_on_image_full(
            image_path=image_path,
            output_image_path=output_image_path,
            output_stats_dir=output_reports_dir,
        )
        
        # Prepare output
        output = {
            "success": True,
            "image_name": image_path.name,
            "annotated_image": output_image_path,
            "total_cells": result.stats.total_cells,
            "cell_counts": result.stats.cell_counts,
            "cell_percentages": result.stats.cell_percentages,
            "warnings": result.stats.warnings,
            "report_text": result.report_text,
        }
        
        if save_reports and output_reports_dir:
            output["reports_dir"] = output_reports_dir
        
        return output
    
    def analyze_folder(
        self,
        folder_path: Path,
        output_dir: Path,
        recursive: bool = False,
        save_reports: bool = True,
    ) -> dict:
        """
        Analyze multiple images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_dir: Directory to save results
            recursive: Whether to search subdirectories
            save_reports: Whether to save detailed reports
        
        Returns:
            Dictionary with batch analysis results
        """
        # Find images
        pattern = "**/*" if recursive else "*"
        image_files = [
            f for f in folder_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {folder_path}")
        
        if self.verbose:
            print(f"\n📁 Found {len(image_files)} image(s)")
        
        # Analyze each image
        results = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "images": [],
            "total_cells": 0,
            "cell_counts": {},
        }
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                if self.verbose:
                    print(f"  [{idx}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)
                
                result = self.analyze_image(image_path, output_dir, save_reports=save_reports)
                result["index"] = idx
                results["images"].append(result)
                results["successful"] += 1
                results["total_cells"] += result["total_cells"]
                
                # Aggregate cell counts
                for cell_type, count in result["cell_counts"].items():
                    results["cell_counts"][cell_type] = results["cell_counts"].get(cell_type, 0) + count
                
                if self.verbose:
                    print(f"✓ ({result['total_cells']} cells)")
                else:
                    print(".", end="", flush=True)
            
            except Exception as e:
                results["failed"] += 1
                if self.verbose:
                    print(f"❌ {e}")
                else:
                    print("✗", end="", flush=True)
        
        # Calculate averages
        if results["successful"] > 0:
            results["avg_cells_per_image"] = results["total_cells"] / results["successful"]
        else:
            results["avg_cells_per_image"] = 0
        
        return results
    
    def print_analysis_report(self, analysis_result: dict) -> None:
        """Print analysis results to console."""
        print("\n" + "=" * 70)
        print("✅ ANALYSIS REPORT")
        print("=" * 70)
        
        if "report_text" in analysis_result:
            # Single image result
            print(analysis_result["report_text"])
        else:
            # Batch result
            print(f"\nTotal Images:          {analysis_result['total_images']}")
            print(f"Successfully Analyzed: {analysis_result['successful']}")
            print(f"Failed:                {analysis_result['failed']}")
            print(f"Total Cells Detected:  {analysis_result['total_cells']}")
            print(f"Average per Image:     {analysis_result['avg_cells_per_image']:.1f}")
            
            print("\nCell Type Distribution:")
            for cell_type, count in analysis_result["cell_counts"].items():
                pct = (count / analysis_result["total_cells"] * 100) if analysis_result["total_cells"] > 0 else 0
                print(f"  {cell_type:15} {count:5d} ({pct:6.2f}%)")
        
        print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blood Cell Detection & Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python app.py --image sample.png --output results/

  # Analyze folder
  python app.py --folder images/ --output results/ --recursive

  # With custom models
  python app.py --image sample.png --yolo models/custom_yolo.pt --ml models/custom_ml.pt

  # Verbose output
  python app.py --image sample.png --output results/ --verbose
        """
    )
    
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to input image file (single image mode)",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Path to folder containing images (batch mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis_results"),
        help="Output directory for results (default: outputs/analysis_results)",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        help="Path to YOLO model (default: models/yolo/best.pt)",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        help="Path to ML model (default: models/ml/best_ml_model.pt)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories (for folder mode)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip saving detailed reports (only save annotated images)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        print("❌ Error: Please specify either --image or --folder")
        sys.exit(1)
    
    if args.image and args.folder:
        print("❌ Error: Cannot specify both --image and --folder")
        sys.exit(1)
    
    try:
        # Initialize app
        app = BloodCellAnalysisApp(
            yolo_model_path=args.yolo_model,
            ml_model_path=args.ml_model,
            verbose=args.verbose,
        )
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        if args.image:
            print(f"🩺 Blood Cell Analysis System - Single Image Mode")
            print(f"   Image: {args.image}")
            result = app.analyze_image(
                image_path=args.image,
                output_dir=args.output,
                save_reports=not args.no_reports,
            )
        else:
            print(f"🩺 Blood Cell Analysis System - Batch Processing Mode")
            print(f"   Folder: {args.folder}")
            result = app.analyze_folder(
                folder_path=args.folder,
                output_dir=args.output,
                recursive=args.recursive,
                save_reports=not args.no_reports,
            )
        
        # Print results
        app.print_analysis_report(result)
        
        # Print output locations
        print(f"\n📁 Results saved to: {args.output}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
