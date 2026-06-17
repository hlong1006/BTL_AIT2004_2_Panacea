#!/usr/bin/env python3
"""Huấn luyện hoặc huấn luyện lại bộ phát hiện tế bào YOLOv8 với các thiết lập được tối ưu cho độ phủ (recall)."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện bộ phát hiện tế bào máu YOLOv8")
    parser.add_argument(
        "--data",
        type=Path,
        default=PATHS.root / "data" / "data.yaml",
        help="Đường dẫn đến tệp data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Mô hình cơ sở (yolov8n/s/m). Sử dụng yolov8s hoặc yolov8m để có độ phủ (recall) tốt hơn.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="", help="ID của thiết bị cuda hoặc 'cpu'")
    parser.add_argument(
        "--project",
        type=Path,
        default=PATHS.root / "runs",
        help="Thư mục dự án của Ultralytics",
    )
    parser.add_argument("--name", type=str, default="cell_detect")
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Sao chép tệp best.pt đến đây sau khi hoàn thành huấn luyện",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp data.yaml: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "save": True,
        "verbose": True,
        "seed": 0,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "iou": 0.6,
        "max_det": 500,
    }
    if args.device:
        train_kwargs["device"] = args.device

    print("Đang huấn luyện YOLO với cấu hình:", train_kwargs)
    model.train(**train_kwargs)

    best_src = args.project / args.name / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"Quá trình huấn luyện đã kết thúc nhưng không tìm thấy tệp best.pt: {best_src}")

    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, args.copy_to)
    print(f"Đã sao chép {best_src} -> {args.copy_to}")


if __name__ == "__main__":
    main()