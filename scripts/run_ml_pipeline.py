"""Xây dựng tệp CSV đặc trưng và huấn luyện bộ phân loại ML (Mô hình YOLO phải tồn tại trước đó)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    import os

    print(f"\n>>> {' '.join(cmd)}")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    py = sys.executable

    yolo_dst = ROOT / "models" / "yolo" / "best.pt"
    yolo_src = ROOT / "runs" / "cell_detect" / "weights" / "best.pt"
    if not yolo_dst.exists() and yolo_src.exists():
        yolo_dst.parent.mkdir(parents=True, exist_ok=True)
        yolo_dst.write_bytes(yolo_src.read_bytes())
        print(f"Đã sao chép trọng số YOLO -> {yolo_dst}")

    # 1. Trích xuất đặc trưng từ nhãn để xây dựng tập dữ liệu huấn luyện ML
    run([py, "scripts/build_train_features_from_labels.py"])
    
    # 2. Huấn luyện các mô hình phân loại ML cổ điển (SVM, KNN, DT)
    run([py, "scripts/train_ml.py"])
    
    print("\nQuy trình ML đã hoàn thành. Chạy thử nghiệm suy luận bằng lệnh sau:")
    print("  python scripts/infer_image.py --image <đường-dẫn-tới-ảnh>")


if __name__ == "__main__":
    main()