import argparse
from pathlib import Path

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run hybrid cell detection-classification pipeline on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Input microscopic image path")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.png"), help="Output annotated image path")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("outputs/predicted_features.csv"),
        help="Optional per-cell features + predictions",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Path to YOLO weights (if missing, fallback will use full image as one object)",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=PATHS.ml_models / ML_MODEL_FILENAME,
        help="Path to trained ML classifier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)
    out_path = pipeline.run_on_image(args.image, args.output, args.feature_csv)
    print(f"Inference done. Saved to: {out_path}")


if __name__ == "__main__":
    main()
