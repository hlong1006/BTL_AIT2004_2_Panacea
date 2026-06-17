from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    raw_images: Path = root / "data" / "train" / "images"
    raw_labels: Path = root / "data" / "train" / "labels"
    interim_crops: Path = root / "data" / "interim" / "crops"
    processed_features: Path = root / "data" / "processed" / "features"
    yolo_models: Path = root / "models" / "yolo"
    ml_models: Path = root / "models" / "ml"


PATHS = ProjectPaths()
