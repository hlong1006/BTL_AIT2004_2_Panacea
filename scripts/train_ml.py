from pathlib import Path

import pandas as pd

from src.classification.ml_classifier import ML_MODEL_FILENAME, MLClassifier
from src.config.settings import PATHS


def main() -> None:
    feature_csv = PATHS.processed_features / "train_features.csv"
    model_out = PATHS.ml_models / ML_MODEL_FILENAME

    trainer = MLClassifier()
    reports = trainer.train_and_select(feature_csv, target_col="label")

    df = pd.read_csv(feature_csv)
    classes = sorted(df["label"].dropna().unique().tolist())
    best_name = reports["best_model"]

    MLClassifier.save_model(
        trainer.models["best"],
        model_out,
        feature_columns=trainer.feature_columns,
        model_name=best_name,
        classes=classes,
    )

    print("=== MODEL REPORTS ===")
    for name, report in reports.items():
        print(f"\n[{name}]\n{report}")
    print(f"\nSaved best model to: {model_out}")


if __name__ == "__main__":
    main()
