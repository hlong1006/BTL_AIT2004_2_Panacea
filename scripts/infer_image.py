import argparse
from pathlib import Path

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid cell detection-classification pipeline with full statistics reporting."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input microscopic image path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prediction.png"),
        help="Output annotated image path",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory to save CSV/JSON/TXT reports",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=PATHS.ml_models / ML_MODEL_FILENAME,
        help="Path to trained ML classifier",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip saving reports (only save image)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validate input image
    if not args.image.exists():
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Load pipeline
    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    if args.verbose:
        print(f"📦 Loading pipeline...")
        if yolo_path:
            print(f"   ✓ YOLO model: {yolo_path}")
        else:
            print(f"   ⚠️  YOLO model not found (will use fallback)")
        print(f"   ✓ ML model: {args.ml_model}")
    
    try:
        pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return
    
    # Run inference with full statistics
    try:
        if args.verbose:
            print(f"\n🔍 Processing image: {args.image.name}")
        
        output_reports_dir = None if args.no_reports else args.reports
        result = pipeline.run_on_image_full(
            image_path=args.image,
            output_image_path=args.output,
            output_stats_dir=output_reports_dir,
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("✅ INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(result.report_text)
        print("=" * 70)
        
        # Print file locations
        print(f"\n📁 Output Files:")
        print(f"   📷 Image:   {result.image_path}")
        
        if output_reports_dir:
            print(f"   📊 Reports: {output_reports_dir}")
            print(f"      ├─ {args.image.stem}_report.txt")
            print(f"      ├─ {args.image.stem}_summary.csv")
            print(f"      ├─ {args.image.stem}_report.json")
            print(f"      ├─ {args.image.stem}_report.xlsx")
            print(f"      └─ {args.image.stem}_features.csv")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
