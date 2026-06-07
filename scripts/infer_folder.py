"""
Batch processing script for analyzing multiple blood sample images.
Processes all images in a folder and generates consolidated reports.
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline
from src.utils.statistics import StatisticsCalculator, ReportExporter


class BatchProcessor:
    """Process multiple images and generate consolidated reports."""
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    def __init__(self, pipeline: HybridCellPipeline, verbose: bool = False):
        self.pipeline = pipeline
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
    
    def process_folder(
        self,
        input_folder: Path,
        output_folder: Path,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Folder containing images
            output_folder: Folder to save results
            recursive: If True, search subdirectories
        
        Returns:
            Summary statistics for the batch
        """
        # Find all images
        pattern = "**/*" if recursive else "*"
        image_files = [
            f
            for f in input_folder.glob(pattern)
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS and f.is_file()
        ]
        
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return {}
        
        print(f"📁 Found {len(image_files)} image(s)")
        
        # Process each image
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images"), 1):
            self._process_single_image(image_path, output_folder, idx, len(image_files))
        
        # Generate consolidated report
        summary = self._generate_consolidated_report(output_folder)
        
        return summary
    
    def _process_single_image(
        self,
        image_path: Path,
        output_folder: Path,
        current: int,
        total: int,
    ) -> None:
        """Process a single image and store results."""
        
        try:
            # Define output paths
            output_image_path = output_folder / "images" / f"{image_path.stem}_annotated.png"
            output_reports_dir = output_folder / "reports" / image_path.stem
            
            # Run pipeline
            result = self.pipeline.run_on_image_full(
                image_path=image_path,
                output_image_path=output_image_path,
                output_stats_dir=output_reports_dir,
            )
            
            # Store result
            self.results.append({
                "image_name": image_path.name,
                "image_stem": image_path.stem,
                "total_cells": result.stats.total_cells,
                "cell_counts": result.stats.cell_counts,
                "cell_percentages": result.stats.cell_percentages,
                "warnings": len(result.stats.warnings),
                "has_warnings": len(result.stats.warnings) > 0,
            })
        
        except Exception as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def _generate_consolidated_report(self, output_folder: Path) -> Dict[str, Any]:
        """Generate consolidated report for all processed images."""
        
        if not self.results:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate batch statistics
        total_images = len(self.results)
        total_cells = df["total_cells"].sum()
        avg_cells_per_image = total_cells / total_images if total_images > 0 else 0
        images_with_warnings = df["has_warnings"].sum()
        
        # Calculate average cell type distribution
        all_counts = {}
        for counts in df["cell_counts"]:
            for cell_type, count in counts.items():
                all_counts[cell_type] = all_counts.get(cell_type, 0) + count
        
        avg_counts = {ct: count / total_images for ct, count in all_counts.items()}
        
        # Create summary
        summary = {
            "processed_date": datetime.now().isoformat(),
            "total_images": total_images,
            "total_cells_detected": total_cells,
            "avg_cells_per_image": avg_cells_per_image,
            "cell_type_totals": all_counts,
            "avg_cells_per_type": avg_counts,
            "images_with_alerts": images_with_warnings,
            "alert_rate": (images_with_warnings / total_images * 100) if total_images > 0 else 0,
        }
        
        # Save consolidated reports
        reports_dir = output_folder / "consolidated"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV summary
        df.to_csv(reports_dir / "batch_summary.csv", index=False)
        
        # JSON summary
        import json
        with open(reports_dir / "batch_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Text report
        self._save_text_report(summary, df, reports_dir / "batch_report.txt")
        
        # Excel report
        try:
            with pd.ExcelWriter(reports_dir / "batch_report.xlsx", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Add statistics sheet
                stats_data = {
                    "Metric": list(summary.keys()),
                    "Value": [str(v) for v in summary.values()],
                }
                pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)
        except ImportError:
            print("⚠️  openpyxl not installed. Skipping Excel export.")
        
        return summary
    
    @staticmethod
    def _save_text_report(summary: Dict[str, Any], df: pd.DataFrame, output_path: Path) -> None:
        """Save consolidated text report."""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("BATCH PROCESSING REPORT - BLOOD CELL ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Processing Date: {summary['processed_date']}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Images Processed:     {summary['total_images']}\n")
            f.write(f"Total Cells Detected:       {summary['total_cells_detected']}\n")
            f.write(f"Average Cells/Image:        {summary['avg_cells_per_image']:.1f}\n")
            f.write(f"Images with Alerts:         {summary['images_with_alerts']}/{summary['total_images']} ({summary['alert_rate']:.1f}%)\n\n")
            
            # Cell type statistics
            f.write("CELL TYPE STATISTICS (Total Across All Images)\n")
            f.write("-" * 70 + "\n")
            for cell_type, count in summary["cell_type_totals"].items():
                avg = summary["avg_cells_per_type"].get(cell_type, 0)
                f.write(f"{cell_type:15} Total: {count:6d}  |  Average: {avg:6.1f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("DETAILED IMAGE RESULTS\n")
            f.write("-" * 70 + "\n\n")
            
            # Detailed results
            for _, row in df.iterrows():
                f.write(f"Image: {row['image_name']}\n")
                f.write(f"  Total Cells: {row['total_cells']}\n")
                for cell_type, count in row['cell_counts'].items():
                    pct = row['cell_percentages'].get(cell_type, 0)
                    f.write(f"    {cell_type}: {count} ({pct:.1f}%)\n")
                if row['has_warnings']:
                    f.write(f"  ⚠️  Warnings: {row['warnings']}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process multiple blood cell images with consolidated reporting."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PATHS.root / "data" / "test" / "images",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "outputs" / "batch_results",
        help="Output folder for results",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=PATHS.ml_models / ML_MODEL_FILENAME,
        help="Path to ML model",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input folder
    if not args.images_dir.exists():
        print(f"❌ Error: Input folder not found: {args.images_dir}")
        return
    
    print(f"🔄 Batch Processing Mode")
    print(f"📁 Input:  {args.images_dir}")
    print(f"📁 Output: {args.output_dir}")
    
    # Initialize pipeline
    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    
    try:
        pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return
    
    # Process batch
    processor = BatchProcessor(pipeline, verbose=args.verbose)
    summary = processor.process_folder(
        args.images_dir,
        args.output_dir,
        recursive=args.recursive,
    )
    
    print("\n" + "=" * 70)
    print("✅ BATCH PROCESSING COMPLETED")
    print("=" * 70)
    print(f"\nTotal Images:     {summary.get('total_images', 0)}")
    print(f"Total Cells:      {summary.get('total_cells_detected', 0)}")
    print(f"Average/Image:    {summary.get('avg_cells_per_image', 0):.1f}")
    print(f"Images w/Alerts:  {summary.get('images_with_alerts', 0)} ({summary.get('alert_rate', 0):.1f}%)")
    print(f"\n📊 Reports: {args.output_dir / 'consolidated'}")
    print("=" * 70)


if __name__ == "__main__":
    main()


def summarize_cells(df: pd.DataFrame) -> dict:
    counts = df["predicted_label"].value_counts().to_dict() if "predicted_label" in df.columns else {}
    return {
        "total_cells": len(df),
        "platelets": counts.get("Platelets", 0),
        "rbc": counts.get("RBC", 0),
        "wbc": counts.get("WBC", 0),
    }


def main() -> None:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {args.images_dir}")

    image_paths = list_images(args.images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.features_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)

    summary_rows = []
    for img_path in tqdm(image_paths, desc="Inferring"):
        out_image = args.output_dir / f"{img_path.stem}.png"
        out_csv = args.features_dir / f"{img_path.stem}.csv"
        pipeline.run_on_image(img_path, out_image, out_csv)

        if out_csv.exists():
            df = pd.read_csv(out_csv)
            row = {"image": img_path.name, **summarize_cells(df)}
        else:
            row = {"image": img_path.name, "total_cells": 0, "platelets": 0, "rbc": 0, "wbc": 0}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"\nDone. Processed {len(image_paths)} images.")
    print(f"  Annotated images : {args.output_dir}")
    print(f"  Per-image CSV    : {args.features_dir}")
    print(f"  Summary CSV      : {args.summary_csv}")
    print(f"\nTotal cells detected: {summary_df['total_cells'].sum()}")
    print(summary_df[["platelets", "rbc", "wbc"]].sum().to_string())


if __name__ == "__main__":
    main()
