"""Gợi ý phân loại WBC con dựa trên heuristic hình thái (demo — chưa có model riêng)."""

from typing import Dict, List, Optional

import pandas as pd


class WBCSubtypeEstimator:
    """
    Ước lượng loại bạch cầu con từ đặc trưng hình thái.
    Đây là heuristic demo — cần dataset & model riêng để chính xác lâm sàng.
    """

    SUBTYPES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]

    def estimate_from_features(self, features_df: pd.DataFrame, labels: List[str]) -> List[Dict]:
        results = []
        if features_df is None or len(features_df) == 0:
            return results

        for idx, row in features_df.iterrows():
            label = row.get("predicted_label", labels[idx] if idx < len(labels) else "")
            if str(label) != "WBC":
                continue
            subtype, confidence, reason = self._classify_wbc(row)
            results.append({
                "cell_index": int(idx),
                "subtype": subtype,
                "confidence": round(confidence, 3),
                "reason": reason,
            })
        return results

    def _classify_wbc(self, row: pd.Series) -> tuple:
        area = float(row.get("area", 0))
        circ = float(row.get("circularity", 0.5))
        ecc = float(row.get("eccentricity", 0.5))

        if area > 3500 and circ < 0.75:
            return "Monocyte", 0.55, "Kích thước lớn, hình dạng không tròn"
        if area < 1800 and circ > 0.8:
            return "Lymphocyte", 0.6, "Nhỏ, tròn đều"
        if ecc > 0.65:
            return "Neutrophil", 0.5, "Độ lệch tâm cao — gợi ý đa nhân"
        if area > 2500:
            return "Eosinophil", 0.45, "Kích thước trung bình-lớn"
        return "Neutrophil", 0.4, "Mặc định — cần model chuyên biệt để xác nhận"

    def summary(self, subtypes: List[Dict]) -> str:
        if not subtypes:
            return "Không có bạch cầu để phân loại con."
        counts: Dict[str, int] = {}
        for s in subtypes:
            counts[s["subtype"]] = counts.get(s["subtype"], 0) + 1
        lines = [f"Gợi ý phân loại {len(subtypes)} bạch cầu (heuristic demo):"]
        for st, c in sorted(counts.items(), key=lambda x: -x[1]):
            pct = c / len(subtypes) * 100
            lines.append(f"  • {st}: {c} ({pct:.0f}%)")
        lines.append("Lưu ý: Cần huấn luyện model WBC differential để kết quả lâm sàng.")
        return "\n".join(lines)
