from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MLClassifier:
    def __init__(self):
        self.models = {
            "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
            "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
        }

    def train_and_select(self, csv_path: Path, target_col: str = "label") -> Dict[str, str]:
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"Missing target column '{target_col}' in {csv_path}")

        x = df.drop(columns=[target_col, "image_name"])
        y = df[target_col]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

        reports: Dict[str, str] = {}
        best_name = ""
        best_score = -1.0
        best_model = None

        for name, model in self.models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            report = classification_report(y_test, pred, zero_division=0)
            reports[name] = report
            score = model.score(x_test, y_test)
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model

        reports["best_model"] = best_name
        reports["best_score"] = f"{best_score:.4f}"
        self.models["best"] = best_model
        return reports

    @staticmethod
    def save_model(model, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)

    @staticmethod
    def load_model(model_path: Path):
        return joblib.load(model_path)

    @staticmethod
    def predict(model, rows: List[Dict[str, float]]) -> List[str]:
        df = pd.DataFrame(rows)
        return model.predict(df).tolist()

