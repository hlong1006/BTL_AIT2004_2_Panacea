"""Run hybrid pipeline on all images in a folder."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect, count and classify cells for every image in a folder.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PATHS.root / "data" / "test" / "images",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "outputs" / "predictions",
        help="Folder for annotated images",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PATHS.root / "outputs" / "features",
        help="Folder for per-image feature CSV files",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PATHS.root / "outputs" / "summary.csv",
        help="Combined count + classification summary for all images",
    )
    parser.add_argument("--yolo-model", type=Path, default=PATHS.yolo_models / "best.pt")
    parser.add_argument("--ml-model", type=Path, default=PATHS.ml_models / ML_MODEL_FILENAME)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


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
