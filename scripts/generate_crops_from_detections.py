"""
Generate crop images from YOLO detections on training data.
This script runs inference on all images in data/train/images/
and saves cropped cells to data/interim/crops/
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Generate crop images from YOLO detections")
    
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "yolo" / "best.pt",
        help="Path to YOLO model (best.pt)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "data" / "train" / "images",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "interim" / "crops",
        help="Directory to save crop images",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for YOLO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = YoloDetector(model_path=args.model_path, conf_threshold=args.conf_threshold)

    # Get all images
    image_paths = sorted([p for p in args.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])

    if not image_paths:
        print(f"[ERROR] No images found in {args.images_dir}")
        return

    print(f"Found {len(image_paths)} images. Starting crop generation...")
    total_crops = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        # Run detection
        detections = detector.detect(image)

        # Save crops
        for idx, det in enumerate(detections):
            crop = detector.crop_detection(image, det)
            crop_name = f"{img_path.stem}_cell_{idx:04d}.png"
            crop_path = args.output_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            total_crops += 1

    print(f"\n✓ Successfully generated {total_crops} crop images")
    print(f"  Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
