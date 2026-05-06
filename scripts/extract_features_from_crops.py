import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.features.extractor import CellFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract handcrafted features from cropped cell images.")
    parser.add_argument("--crops-dir", type=Path, required=True, help="Directory containing crop images")
    parser.add_argument("--output-csv", type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--label", type=str, default=None, help="Optional fixed label for all rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = CellFeatureExtractor()
    rows = []

    image_paths = sorted([p for p in args.crops_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    for img_path in tqdm(image_paths, desc="Extracting features"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        feat = extractor.extract(img)
        feat["image_name"] = img_path.name
        if args.label is not None:
            feat["label"] = args.label
        rows.append(feat)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
