"""Giải thích quyết định phân loại ML bằng permutation importance."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class FeatureExplainer:
    """Giải thích tại sao mô hình ML phân loại tế bào theo cách đó."""

    FEATURE_LABELS = {
        "area": "Diện tích",
        "perimeter": "Chu vi",
        "circularity": "Độ tròn",
        "eccentricity": "Độ lệch tâm",
        "solidity": "Độ đặc",
        "extent": "Mức lấp đầy",
        "hu_moment_1": "Hu moment 1",
        "hu_moment_2": "Hu moment 2",
        "hu_moment_3": "Hu moment 3",
        "mean_b": "Màu B trung bình",
        "mean_g": "Màu G trung bình",
        "mean_r": "Màu R trung bình",
        "mean_h": "Hue trung bình",
        "mean_s": "Saturation trung bình",
        "mean_v": "Value trung bình",
        "color_std_b": "Độ lệch màu B",
        "color_std_g": "Độ lệch màu G",
        "color_std_r": "Độ lệch màu R",
        "color_std_hsv": "Độ lệch Hue",
        "texture_laplacian_var": "Độ sắc nét kết cấu",
    }

    @staticmethod
    def explain_cell(
        checkpoint: dict,
        feature_row: Dict[str, float],
        top_k: int = 5,
    ) -> Dict:
        """Giải thích một tế bào bằng so sánh đặc trưng với trung bình từng lớp."""
        model = checkpoint["model"]
        feature_columns = checkpoint["feature_columns"]
        classes = checkpoint.get("classes", ["Platelets", "RBC", "WBC"])

        x = pd.DataFrame([feature_row]).reindex(columns=feature_columns, fill_value=0.0)
        pred = model.predict(x)[0]
        pred_label = str(pred)

        # Lấy xác suất nếu có
        confidence = 1.0
        prob_dict = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            for i, cls in enumerate(classes):
                if i < len(probs):
                    prob_dict[str(cls)] = round(float(probs[i]), 4)
            confidence = float(np.max(probs))

        # Đặc trưng nổi bật: so với giá trị điển hình từng lớp (heuristic)
        typical = FeatureExplainer._typical_features()
        typical_for_class = typical.get(pred_label, {})
        contributions = []
        for col in feature_columns:
            val = float(feature_row.get(col, 0))
            ref = typical_for_class.get(col, val)
            diff = abs(val - ref) / (abs(ref) + 1e-6)
            contributions.append({
                "feature": col,
                "label_vi": FeatureExplainer.FEATURE_LABELS.get(col, col),
                "value": round(val, 4),
                "typical": round(ref, 4),
                "deviation": round(diff, 4),
            })
        contributions.sort(key=lambda c: c["deviation"], reverse=True)

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "top_features": contributions[:top_k],
            "explanation_text": FeatureExplainer._build_text(pred_label, confidence, contributions[:top_k]),
        }

    @staticmethod
    def global_importance(checkpoint: dict, sample_df: pd.DataFrame, top_k: int = 8) -> Dict:
        """Tính permutation importance trên mẫu (thay SHAP toàn cục)."""
        if sample_df is None or len(sample_df) < 10:
            return {"features": [], "method": "insufficient_data"}

        model = checkpoint["model"]
        feature_columns = checkpoint["feature_columns"]
        labels = sample_df.get("predicted_label", sample_df.get("label"))
        if labels is None:
            return {"features": [], "method": "no_labels"}

        x = sample_df.reindex(columns=feature_columns, fill_value=0.0)
        y = labels.values

        try:
            result = permutation_importance(model, x, y, n_repeats=5, random_state=42, n_jobs=1)
            importances = list(zip(feature_columns, result.importances_mean))
            importances.sort(key=lambda t: t[1], reverse=True)
            features = [
                {
                    "feature": name,
                    "label_vi": FeatureExplainer.FEATURE_LABELS.get(name, name),
                    "importance": round(float(imp), 4),
                }
                for name, imp in importances[:top_k]
            ]
            return {"features": features, "method": "permutation_importance"}
        except Exception as exc:
            return {"features": [], "method": f"error: {exc}"}

    @staticmethod
    def _typical_features() -> Dict[str, Dict[str, float]]:
        return {
            "RBC": {"area": 800, "circularity": 0.85, "eccentricity": 0.3, "mean_r": 140},
            "WBC": {"area": 2500, "circularity": 0.7, "eccentricity": 0.5, "mean_r": 120},
            "Platelets": {"area": 200, "circularity": 0.6, "eccentricity": 0.6, "mean_r": 100},
        }

    @staticmethod
    def _build_text(label: str, confidence: float, top_features: List[Dict]) -> str:
        lines = [
            f"Mô hình phân loại: {label} (độ tin cậy {confidence*100:.1f}%)",
            "Các đặc trưng quyết định chính:",
        ]
        for i, f in enumerate(top_features, 1):
            lines.append(
                f"  {i}. {f['label_vi']}: {f['value']:.2f} "
                f"(điển hình {label}: ~{f['typical']:.2f})"
            )
        return "\n".join(lines)
