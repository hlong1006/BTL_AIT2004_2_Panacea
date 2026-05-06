from pathlib import Path

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS


def main() -> None:
    feature_csv = PATHS.processed_features / "train_features.csv"
    model_out = PATHS.ml_models / "best_ml_model.joblib"

    trainer = MLClassifier()
    reports = trainer.train_and_select(feature_csv, target_col="label")
    MLClassifier.save_model(trainer.models["best"], model_out)

    print("=== MODEL REPORTS ===")
    for name, report in reports.items():
        print(f"\n[{name}]\n{report}")
    print(f"\nSaved best model to: {model_out}")


if __name__ == "__main__":
    main()
