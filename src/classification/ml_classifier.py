from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


METADATA_COLS = frozenset({"image_name", "label", "split", "predicted_label", "det_confidence"})
ML_MODEL_FILENAME = "best_ml_model.pt"


class MLClassifier:
    def __init__(self):
        # Improved model configurations for better class separation
        self.models = {
            # KNN: Good baseline, fast inference. K=5 often good for 3-class problems
            "knn": Pipeline([
                ("scaler", StandardScaler()), 
                ("clf", KNeighborsClassifier(n_neighbors=5, weights='distance'))
            ]),
            
            # Decision Tree: Interpretable, handles non-linear boundaries
            "decision_tree": DecisionTreeClassifier(
                max_depth=10,  # Increased from 8 for better discrimination
                min_samples_split=5,  # Prevent overfitting
                random_state=42
            ),
            
            # SVM: Best for linear/non-linear separation, especially WBC vs RBC
            # Using RBF kernel with optimized C parameter
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel='rbf',
                    C=1.0,  # Regularization parameter
                    gamma='scale',  # Automatic gamma scaling
                    probability=True,  # Enable probability estimates
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                ))
            ]),
        }
        self.feature_columns: List[str] = []

    def train_and_select(self, csv_path: Path, target_col: str = "label") -> Dict[str, str]:
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"Missing target column '{target_col}' in {csv_path}")

        drop_cols = [c for c in df.columns if c == target_col or c in METADATA_COLS]
        x = df.drop(columns=drop_cols)
        y = df[target_col]
        self.feature_columns = list(x.columns)

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
    def save_model(
        model,
        save_path: Path,
        *,
        feature_columns: List[str],
        model_name: str,
        classes: List[str],
    ) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": model_name,
            "classes": classes,
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load_model(model_path: Path) -> dict:
        if not model_path.exists():
            raise FileNotFoundError(f"ML model not found: {model_path}")
        return torch.load(model_path, map_location="cpu", weights_only=False)

    @staticmethod
    def predict(checkpoint: dict, rows: List[Dict[str, float]]) -> List[str]:
        model = checkpoint["model"]
        feature_columns = checkpoint["feature_columns"]
        df = pd.DataFrame(rows)
        drop_cols = [c for c in df.columns if c in METADATA_COLS]
        x = df.drop(columns=drop_cols, errors="ignore")
        x = x.reindex(columns=feature_columns, fill_value=0.0)
        return model.predict(x).tolist()
    
    @staticmethod
    def predict_with_confidence(checkpoint: dict, rows: List[Dict[str, float]], 
                                confidence_threshold: float = 0.0) -> List[tuple]:
        """
        Predict with confidence scores.
        
        Returns list of (predicted_label, confidence) tuples.
        
        Args:
            checkpoint: Model checkpoint
            rows: List of feature dictionaries
            confidence_threshold: Minimum confidence (0.0-1.0)
        
        Returns:
            List of (label_string, confidence_float) tuples
        """
        model = checkpoint["model"]
        feature_columns = checkpoint["feature_columns"]
        classes = checkpoint.get("classes", [0, 1, 2])
        
        df = pd.DataFrame(rows)
        drop_cols = [c for c in df.columns if c in METADATA_COLS]
        x = df.drop(columns=drop_cols, errors="ignore")
        x = x.reindex(columns=feature_columns, fill_value=0.0)
        
        predictions = []
        
        # Try to get probabilities if model supports it
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(x)
            for prob_row in probs:
                max_prob = float(np.max(prob_row))
                max_idx = int(np.argmax(prob_row))
                label = str(classes[max_idx]) if max_idx < len(classes) else str(max_idx)
                
                # If confidence below threshold, mark as uncertain
                if max_prob < confidence_threshold:
                    label = f"{label}?"
                
                predictions.append((label, max_prob))
        else:
            # Fallback to regular predict
            preds = model.predict(x).tolist()
            for pred in preds:
                predictions.append((str(pred), 1.0))
        
        return predictions
