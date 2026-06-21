"""Phát hiện tế bào bất thường dựa trên đặc trưng hình thái."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CellAnomaly:
    cell_index: int
    predicted_label: str
    anomaly_score: float
    reasons: List[str]
    area: float
    circularity: float


class AnomalyDetector:
    """Phát hiện tế bào có hình thái bất thường so với cùng loại."""

    FEATURE_COLS = ["area", "perimeter", "circularity", "eccentricity", "solidity", "texture_laplacian_var"]

    def detect(
        self,
        features_df: pd.DataFrame,
        labels: List[str],
        z_threshold: float = 2.5,
    ) -> List[CellAnomaly]:
        if features_df is None or len(features_df) == 0:
            return []

        df = features_df.copy()
        if "predicted_label" not in df.columns:
            df["predicted_label"] = labels

        anomalies: List[CellAnomaly] = []
        for idx, row in df.iterrows():
            label = row.get("predicted_label", labels[idx] if idx < len(labels) else "unknown")
            same_type = df[df["predicted_label"] == label]
            if len(same_type) < 3:
                continue

            reasons = []
            z_scores = []
            for col in self.FEATURE_COLS:
                if col not in df.columns:
                    continue
                vals = same_type[col].dropna()
                if len(vals) < 2:
                    continue
                mean, std = vals.mean(), vals.std()
                if std < 1e-6:
                    continue
                z = abs((row[col] - mean) / std)
                if z > z_threshold:
                    reasons.append(f"{col} lệch {z:.1f}σ (giá trị {row[col]:.2f}, TB {mean:.2f})")
                    z_scores.append(z)

            if reasons:
                anomalies.append(CellAnomaly(
                    cell_index=int(idx),
                    predicted_label=str(label),
                    anomaly_score=float(np.mean(z_scores)),
                    reasons=reasons,
                    area=float(row.get("area", 0)),
                    circularity=float(row.get("circularity", 0)),
                ))

        anomalies.sort(key=lambda a: a.anomaly_score, reverse=True)
        return anomalies

    def to_dict_list(self, anomalies: List[CellAnomaly], limit: int = 10) -> List[Dict]:
        return [
            {
                "cell_index": a.cell_index,
                "label": a.predicted_label,
                "anomaly_score": round(a.anomaly_score, 2),
                "reasons": a.reasons,
                "area": round(a.area, 1),
                "circularity": round(a.circularity, 4),
            }
            for a in anomalies[:limit]
        ]

    def summary_text(self, anomalies: List[CellAnomaly]) -> str:
        if not anomalies:
            return "Không phát hiện tế bào hình thái bất thường."
        lines = [f"Phát hiện {len(anomalies)} tế bào hình thái bất thường:"]
        for a in anomalies[:5]:
            lines.append(
                f"  • Tế bào #{a.cell_index} ({a.predicted_label}): "
                f"điểm bất thường {a.anomaly_score:.1f} — {'; '.join(a.reasons[:2])}"
            )
        if len(anomalies) > 5:
            lines.append(f"  ... và {len(anomalies) - 5} tế bào khác")
        return "\n".join(lines)
