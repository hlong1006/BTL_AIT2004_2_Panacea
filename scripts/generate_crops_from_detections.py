"""
Tạo các ảnh cắt (crop) tế bào từ kết quả phát hiện của YOLO trên dữ liệu huấn luyện.
Tập lệnh này chạy suy luận trên tất cả các ảnh trong thư mục data/train/images/
và lưu các tế bào đã được cắt vào data/interim/crops/
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Tạo các ảnh cắt (crop) tế bào từ kết quả phát hiện của YOLO")
    
    # Lấy thư mục gốc của dự án
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "yolo" / "best.pt",
        help="Đường dẫn đến mô hình YOLO (best.pt)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "data" / "train" / "images",
        help="Thư mục chứa các ảnh đầu vào",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "interim" / "crops",
        help="Thư mục để lưu các ảnh cắt",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Ngưỡng độ tin cậy cho YOLO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Đảm bảo thư mục đầu ra tồn tại
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo bộ phát hiện
    detector = YoloDetector(model_path=args.model_path, conf_threshold=args.conf_threshold)

    # Lấy tất cả các đường dẫn ảnh
    image_paths = sorted([p for p in args.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])

    if not image_paths:
        print(f"[LỖI] Không tìm thấy ảnh nào trong thư mục {args.images_dir}")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu quá trình tạo ảnh cắt...")
    total_crops = 0

    for img_path in tqdm(image_paths, desc="Đang xử lý ảnh"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[CẢNH BÁO] Không thể đọc ảnh: {img_path}")
            continue

        # Chạy quy trình phát hiện
        detections = detector.detect(image)

        # Lưu các ảnh cắt
        for idx, det in enumerate(detections):
            crop = detector.crop_detection(image, det)
            crop_name = f"{img_path.stem}_cell_{idx:04d}.png"
            crop_path = args.output_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            total_crops += 1

    print(f"\n✓ Đã tạo thành công {total_crops} ảnh cắt tế bào")
    print(f"  Đã lưu tại: {args.output_dir}")


if __name__ == "__main__":
    main()