#!/usr/bin/env python3
"""Train or retrain YOLOv8 cell detector with settings tuned for recall."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 blood cell detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=PATHS.root / "data" / "data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model (yolov8n/s/m). Use yolov8s or yolov8m for better recall.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="", help="cuda device id or 'cpu'")
    parser.add_argument(
        "--project",
        type=Path,
        default=PATHS.root / "runs",
        help="Ultralytics project directory",
    )
    parser.add_argument("--name", type=str, default="cell_detect")
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Copy best.pt here after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

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

    print("Training YOLO with:", train_kwargs)
    model.train(**train_kwargs)

    best_src = args.project / args.name / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best_src}")

    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, args.copy_to)
    print(f"Copied {best_src} -> {args.copy_to}")


if __name__ == "__main__":
    main()
