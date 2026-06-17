import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.features.extractor import CellFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất các đặc trưng thủ công từ ảnh cắt tế bào.")
    parser.add_argument("--crops-dir", type=Path, required=True, help="Thư mục chứa các ảnh cắt tế bào")
    parser.add_argument("--output-csv", type=Path, required=True, help="Đường dẫn đến tệp CSV kết quả")
    parser.add_argument("--label", type=str, default=None, help="Nhãn cố định tùy chọn cho tất cả các hàng")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = CellFeatureExtractor()
    rows = []

    image_paths = sorted([p for p in args.crops_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    for img_path in tqdm(image_paths, desc="Đang trích xuất đặc trưng"):
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
    print(f"Đã lưu {len(df)} hàng vào tệp {args.output_csv}")


if __name__ == "__main__":
    main()