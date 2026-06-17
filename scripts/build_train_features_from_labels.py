"""Xây dựng tệp train_features.csv từ nhãn gốc ground-truth của YOLO (Tiểu cầu / Hồng cầu / Bạch cầu)."""

import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

from src.config.settings import PATHS
from src.features.extractor import CellFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PATHS.root / "data")
    parser.add_argument("--data-yaml", type=Path, default=PATHS.root / "data" / "data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--output-csv", type=Path, default=PATHS.processed_features / "train_features.csv")
    parser.add_argument("--min-crop-size", type=int, default=8)
    return parser.parse_args()


def yolo_line_to_bbox(line: str, img_w: int, img_h: int) -> tuple[int, int, int, int, int]:
    # Chuyển đổi một dòng cấu trúc YOLO (tọa độ chuẩn hóa) sang tọa độ pixel (hộp giới hạn bounding box)
    parts = line.strip().split()
    class_id = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = max(0, int((cx - bw / 2) * img_w))
    y1 = max(0, int((cy - bh / 2) * img_h))
    x2 = min(img_w, int((cx + bw / 2) * img_w))
    y2 = min(img_h, int((cy + bh / 2) * img_h))
    return class_id, x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def main() -> None:
    args = parse_args()
    
    # Tải danh sách tên các lớp từ tệp cấu hình data.yaml
    with open(args.data_yaml, encoding="utf-8") as f:
        class_names = list(yaml.safe_load(f)["names"])

    extractor = CellFeatureExtractor()
    rows = []

    # Duyệt qua các tập phân tách dữ liệu (thông thường là train và valid)
    for split in args.splits:
        images_dir = args.data_root / split / "images"
        labels_dir = args.data_root / split / "labels"
        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        
        for img_path in tqdm(image_paths, desc=f"Đang trích xuất đặc trưng ({split})"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            # Đọc từng dòng nhãn trong tệp txt của ảnh tương ứng
            for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                    
                class_id, x1, y1, x2, y2 = yolo_line_to_bbox(line, w, h)
                
                # Cắt vùng ảnh tế bào dựa trên tọa độ hộp giới hạn ground-truth
                crop = image[y1:y2, x1:x2]
                
                # Bỏ qua các vùng cắt quá nhỏ để tránh lỗi trích xuất hoặc nhiễu dữ liệu
                if crop.shape[0] < args.min_crop_size or crop.shape[1] < args.min_crop_size:
                    continue
                    
                label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                
                # Trích xuất 23 đặc trưng hình thái, màu sắc và kết cấu
                feat = extractor.extract(crop)
                feat["image_name"] = f"{split}_{img_path.stem}_obj{line_idx:03d}.png"
                feat["label"] = label
                rows.append(feat)

    if not rows:
        raise RuntimeError("Không trích xuất được đặc trưng nào. Vui lòng kiểm tra lại đường dẫn tập dữ liệu.")

    # Tạo thư mục và lưu kết quả ra tệp CSV tổng hợp để huấn luyện ML
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    
    print(f"Đã lưu {len(df)} hàng -> {args.output_csv}")
    print("\nThống kê số lượng theo từng nhãn tế bào:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()