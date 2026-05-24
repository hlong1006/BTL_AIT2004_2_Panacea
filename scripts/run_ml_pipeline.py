"""Build features CSV and train ML classifier (YOLO must already exist)."""

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
        print(f"Copied YOLO weights -> {yolo_dst}")

    run([py, "scripts/build_train_features_from_labels.py"])
    run([py, "scripts/train_ml.py"])
    print("\nML pipeline done. Run inference:")
    print("  python scripts/infer_image.py --image <path-to-image>")


if __name__ == "__main__":
    main()
