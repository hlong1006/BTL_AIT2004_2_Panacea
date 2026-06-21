"""Ước lượng chỉ số CBC từ số tế bào trong field ảnh vi thể."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CBCEstimate:
    rbc_per_ul: Optional[float]
    wbc_per_ul: Optional[float]
    platelets_per_ul: Optional[float]
    method: str
    disclaimer: str
    calibration: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "rbc_per_ul": self.rbc_per_ul,
            "wbc_per_ul": self.wbc_per_ul,
            "platelets_per_ul": self.platelets_per_ul,
            "method": self.method,
            "disclaimer": self.disclaimer,
            "calibration": self.calibration,
        }

    def to_text(self) -> str:
        lines = [
            "ƯỚC LƯỢNG CBC (tham khảo — KHÔNG thay thế xét nghiệm)",
            f"Phương pháp: {self.method}",
            f"  RBC:       {self._fmt(self.rbc_per_ul)} triệu/μL",
            f"  WBC:       {self._fmt(self.wbc_per_ul)} nghìn/μL",
            f"  Platelets: {self._fmt(self.platelets_per_ul)} nghìn/μL",
            f"\n{self.disclaimer}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt(val: Optional[float]) -> str:
        if val is None:
            return "N/A"
        if val >= 1000:
            return f"{val:,.0f}"
        return f"{val:.2f}"


class CBCEstimator:
    """
    Hiệu chuẩn thô: cells_per_field × hệ số → ước lượng nồng độ.

    Hệ số mặc định dựa trên giả định:
    - Field 100× oil immersion ~ 0.002 mm²
    - RBC bình thường ~ 5 triệu/μL
    """

    DEFAULT_CALIBRATION = {
        "magnification": 1000,
        "field_area_mm2": 0.002,
        "rbc_factor": 11000.0,
        "wbc_factor": 120.0,
        "platelet_factor": 2000.0,
    }

    def estimate(
        self,
        cell_counts: Dict[str, int],
        calibration: Optional[Dict[str, float]] = None,
    ) -> CBCEstimate:
        cal = {**self.DEFAULT_CALIBRATION, **(calibration or {})}
        rbc = cell_counts.get("RBC", 0)
        wbc = cell_counts.get("WBC", 0)
        plt = cell_counts.get("Platelets", 0)

        rbc_est = (rbc / cal["rbc_factor"]) if rbc > 0 else None
        wbc_est = (wbc / cal["wbc_factor"]) if wbc > 0 else None
        plt_est = (plt / cal["platelet_factor"]) if plt > 0 else None

        return CBCEstimate(
            rbc_per_ul=rbc_est,
            wbc_per_ul=wbc_est,
            platelets_per_ul=plt_est,
            method=f"Hiệu chuẩn field ảnh (×{cal['magnification']}, {cal['field_area_mm2']} mm²)",
            disclaimer=(
                "Đây là ước lượng thô từ số tế bào đếm trên ảnh. "
                "Cần hiệu chuẩn với máy đếm tự động hoặc CBC thực tế."
            ),
            calibration=cal,
        )
