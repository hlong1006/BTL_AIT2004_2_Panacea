"""
Module phân tích sức khỏe máu & chẩn đoán dựa trên tỷ lệ tế bào trong ảnh lam máu.

Lưu ý: Số tế bào trong một field ảnh vi thể KHÔNG tương đương nồng độ xét nghiệm máu (CBC).
Module này so sánh tỷ lệ % và tỷ lệ tương đối giữa các loại tế bào với ngưỡng tham chiếu
trên lam máu ngoại vi — phù hợp cho sàng lọc sơ bộ, không thay thế xét nghiệm lâm sàng.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


class BloodHealthStatus:
    NORMAL = "Bình thường"
    WARNING = "Cần chú ý"
    ABNORMAL = "Bất thường"
    CRITICAL = "Nguy hiểm"

    _PRIORITY = {
        "Bình thường": 0,
        "Cần chú ý": 1,
        "Bất thường": 2,
        "Nguy hiểm": 3,
    }

    def __init__(self, value: str):
        self.value = value

    def get_priority(self) -> int:
        return self._PRIORITY.get(self.value, 0)


@dataclass
class DiagnosticFinding:
    condition: str
    severity: BloodHealthStatus
    description: str
    reference: str
    actual: str
    recommendation: str = ""


@dataclass
class BloodHealthReport:
    overall_status: BloodHealthStatus
    findings: List[DiagnosticFinding] = field(default_factory=list)
    summary: str = ""
    details: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "=" * 70,
            "BÁO CÁO CHẨN ĐOÁN SỨC KHỎE MÁU (dựa trên tỷ lệ tế bào ảnh lam)",
            "=" * 70,
            "",
            f"TRẠNG THÁI CHUNG: {self.overall_status.value}",
            "",
            "Lưu ý: Kết quả dựa trên tỷ lệ tế bào trong field ảnh vi thể,",
            "không phải nồng độ máu thực tế (CBC). Cần xét nghiệm bổ sung để xác nhận.",
            "",
        ]
        if self.summary:
            lines.extend(["TÓM TẮT:", self.summary, ""])
        if self.findings:
            lines.append("CÁC PHÁT HIỆN CHI TIẾT:")
            for i, finding in enumerate(self.findings, 1):
                lines.extend([
                    f"\n{i}. {finding.condition}",
                    f"   • Mức độ: {finding.severity.value}",
                    f"   • Giá trị tham chiếu: {finding.reference}",
                    f"   • Giá trị thực tế: {finding.actual}",
                    f"   • Mô tả: {finding.description}",
                ])
                if finding.recommendation:
                    lines.append(f"   • Khuyến nghị: {finding.recommendation}")
            lines.append("")
        if self.details:
            lines.extend(["CHI TIẾT PHÂN TÍCH:", self.details, ""])
        if self.recommendations:
            lines.append("KHUYẾN NGHỊ CHUNG:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "details": self.details,
            "findings": [
                {
                    "condition": f.condition,
                    "severity": f.severity.value,
                    "description": f.description,
                    "reference": f.reference,
                    "actual": f.actual,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
            "recommendations": self.recommendations,
        }


class BloodHealthAnalyzer:
    """Phân tích sức khỏe máu dựa trên tỷ lệ % tế bào trong ảnh lam máu."""

    # Tỷ lệ % bình thường trên lam máu ngoại vi (trong field ảnh)
    NORMAL_PERCENTAGE = {
        "RBC": {"min": 75.0, "max": 98.0, "name": "Hồng cầu"},
        "WBC": {"min": 0.5, "max": 8.0, "name": "Bạch cầu"},
        "Platelets": {"min": 1.0, "max": 15.0, "name": "Tiểu cầu"},
    }

    # Tỷ lệ WBC/RBC bình thường (~1 WBC : 20-100 RBC)
    NORMAL_WBC_RBC_RATIO = (0.005, 0.08)

    # Tỷ lệ Platelets/RBC bình thường
    NORMAL_PLT_RBC_RATIO = (0.01, 0.25)

    def __init__(self):
        self.findings: List[DiagnosticFinding] = []

    def analyze(
        self,
        cell_counts: Dict[str, int],
        cell_percentages: Dict[str, float],
        total_cells: int,
        features_df: Optional[pd.DataFrame] = None,
    ) -> BloodHealthReport:
        self.findings = []
        if total_cells == 0:
            return self._create_no_data_report()

        self._analyze_percentages(cell_percentages)
        self._analyze_ratios(cell_counts)
        if features_df is not None and len(features_df) > 0:
            self._analyze_cell_features(features_df)

        return self._create_report(cell_counts, cell_percentages, total_cells)

    def _analyze_percentages(self, cell_percentages: Dict[str, float]) -> None:
        for cell_type, ref in self.NORMAL_PERCENTAGE.items():
            pct = cell_percentages.get(cell_type, 0.0)
            name = ref["name"]
            ref_str = f"{ref['min']:.1f}% – {ref['max']:.1f}%"
            actual_str = f"{pct:.2f}%"

            if pct < ref["min"]:
                gap = ref["min"] - pct
                if gap > ref["min"] * 0.5:
                    severity = BloodHealthStatus(BloodHealthStatus.CRITICAL)
                    condition = f"Thiếu {name} nghiêm trọng (trên ảnh lam)"
                elif gap > ref["min"] * 0.2:
                    severity = BloodHealthStatus(BloodHealthStatus.ABNORMAL)
                    condition = f"Giảm tỷ lệ {name}"
                else:
                    severity = BloodHealthStatus(BloodHealthStatus.WARNING)
                    condition = f"Tỷ lệ {name} thấp hơn bình thường"

                desc_map = {
                    "RBC": "Tỷ lệ hồng cầu thấp trên lam máu — có thể gợi ý thiếu máu hoặc mẫu không đại diện.",
                    "WBC": "Tỷ lệ bạch cầu thấp — có thể gợi ý giảm bạch cầu hoặc mẫu ít bạch cầu.",
                    "Platelets": "Tỷ lệ tiểu cầu thấp — có thể gợi ý giảm tiểu cầu.",
                }
                self.findings.append(DiagnosticFinding(
                    condition=condition,
                    severity=severity,
                    description=desc_map.get(cell_type, f"Tỷ lệ {name} thấp."),
                    reference=ref_str,
                    actual=actual_str,
                    recommendation="Xét nghiệm CBC đầy đủ và kiểm tra chất lượng lam máu.",
                ))

            elif pct > ref["max"]:
                gap = pct - ref["max"]
                if gap > ref["max"] * 0.5:
                    severity = BloodHealthStatus(BloodHealthStatus.ABNORMAL)
                else:
                    severity = BloodHealthStatus(BloodHealthStatus.WARNING)

                desc_map = {
                    "RBC": "Tỷ lệ hồng cầu cao — có thể do mẫu đặc hoặc tăng hồng cầu.",
                    "WBC": "Tỷ lệ bạch cầu cao — có thể gợi ý nhiễm trùng, viêm hoặc tăng bạch cầu.",
                    "Platelets": "Tỷ lệ tiểu cầu cao — có thể gợi ý tăng tiểu cầu hoặc cụm tiểu cầu.",
                }
                self.findings.append(DiagnosticFinding(
                    condition=f"Tăng tỷ lệ {name}",
                    severity=severity,
                    description=desc_map.get(cell_type, f"Tỷ lệ {name} cao."),
                    reference=ref_str,
                    actual=actual_str,
                    recommendation="Đối chiếu với triệu chứng lâm sàng và xét nghiệm bổ sung.",
                ))

    def _analyze_ratios(self, cell_counts: Dict[str, int]) -> None:
        rbc = cell_counts.get("RBC", 0)
        wbc = cell_counts.get("WBC", 0)
        platelets = cell_counts.get("Platelets", 0)

        if rbc > 0 and wbc > 0:
            ratio = wbc / rbc
            lo, hi = self.NORMAL_WBC_RBC_RATIO
            if ratio > hi:
                self.findings.append(DiagnosticFinding(
                    condition="Tỷ lệ WBC/RBC cao",
                    severity=BloodHealthStatus(BloodHealthStatus.WARNING),
                    description="Bạch cầu chiếm tỷ lệ cao so với hồng cầu — có thể gợi ý nhiễm trùng.",
                    reference=f"{lo:.3f} – {hi:.3f}",
                    actual=f"{ratio:.4f} (1 WBC : {rbc/wbc:.0f} RBC)",
                    recommendation="Kiểm tra dấu hiệu nhiễm trùng, công thức máu toàn phần.",
                ))
            elif ratio < lo and wbc > 0:
                self.findings.append(DiagnosticFinding(
                    condition="Tỷ lệ WBC/RBC thấp",
                    severity=BloodHealthStatus(BloodHealthStatus.WARNING),
                    description="Rất ít bạch cầu so với hồng cầu trong field ảnh.",
                    reference=f"{lo:.3f} – {hi:.3f}",
                    actual=f"{ratio:.4f}",
                    recommendation="Kiểm tra vùng quét ảnh có đủ đại diện không.",
                ))

        if rbc > 0 and platelets > 0:
            plt_ratio = platelets / rbc
            lo, hi = self.NORMAL_PLT_RBC_RATIO
            if plt_ratio < lo:
                self.findings.append(DiagnosticFinding(
                    condition="Tỷ lệ tiểu cầu/hồng cầu thấp",
                    severity=BloodHealthStatus(BloodHealthStatus.WARNING),
                    description="Ít tiểu cầu so với hồng cầu trong field ảnh.",
                    reference=f"{lo:.3f} – {hi:.3f}",
                    actual=f"{plt_ratio:.4f}",
                    recommendation="Đếm tiểu cầu trên nhiều field ảnh hoặc xét nghiệm CBC.",
                ))

    def _analyze_cell_features(self, features_df: pd.DataFrame) -> None:
        if "area" not in features_df.columns:
            return
        areas = features_df["area"].dropna()
        if len(areas) == 0:
            return
        median_area = areas.median()
        huge_cells = int((areas > median_area * 3).sum())
        if huge_cells > 0:
            huge_ratio = (huge_cells / len(areas)) * 100
            if huge_ratio > 10:
                self.findings.append(DiagnosticFinding(
                    condition="Tế bào kích thước bất thường",
                    severity=BloodHealthStatus(BloodHealthStatus.WARNING),
                    description=f"{huge_cells} tế bào ({huge_ratio:.1f}%) lớn hơn 3× trung vị.",
                    reference="Kích thước đồng đều",
                    actual=f"{huge_cells} tế bào bất thường",
                    recommendation="Kiểm tra cụm tế bào chưa tách hoặc tế bào bất thường.",
                ))

    def _create_no_data_report(self) -> BloodHealthReport:
        return BloodHealthReport(
            overall_status=BloodHealthStatus(BloodHealthStatus.CRITICAL),
            summary="Không phát hiện tế bào — không thể phân tích.",
            recommendations=[
                "Kiểm tra chất lượng ảnh và độ phóng đại kính hiển vi.",
                "Giảm ngưỡng confidence YOLO hoặc chụp lại mẫu.",
            ],
        )

    def _create_report(
        self,
        cell_counts: Dict[str, int],
        cell_percentages: Dict[str, float],
        total_cells: int,
    ) -> BloodHealthReport:
        if not self.findings:
            overall = BloodHealthStatus(BloodHealthStatus.NORMAL)
            summary = (
                f"Tỷ lệ tế bào trong field ảnh nằm trong khoảng bình thường. "
                f"Tổng {total_cells} tế bào: RBC {cell_percentages.get('RBC', 0):.1f}%, "
                f"WBC {cell_percentages.get('WBC', 0):.1f}%, "
                f"Platelets {cell_percentages.get('Platelets', 0):.1f}%."
            )
        else:
            max_priority = max(f.severity.get_priority() for f in self.findings)
            status_map = {
                3: BloodHealthStatus.CRITICAL,
                2: BloodHealthStatus.ABNORMAL,
                1: BloodHealthStatus.WARNING,
                0: BloodHealthStatus.NORMAL,
            }
            overall = BloodHealthStatus(status_map[max_priority])
            summaries = {
                BloodHealthStatus.CRITICAL: "Phát hiện dấu hiệu nghiêm trọng trên ảnh lam — cần xét nghiệm bổ sung ngay.",
                BloodHealthStatus.ABNORMAL: "Tỷ lệ tế bào bất thường — cần theo dõi và xét nghiệm thêm.",
                BloodHealthStatus.WARNING: "Một số chỉ số cần chú ý — nên đối chiếu lâm sàng.",
            }
            summary = summaries.get(overall.value, "Có phát hiện cần lưu ý.")

        recommendations = self._generate_recommendations()
        details = self._generate_details(cell_counts, cell_percentages)

        return BloodHealthReport(
            overall_status=overall,
            findings=self.findings,
            summary=summary,
            details=details,
            recommendations=recommendations,
        )

    def _generate_recommendations(self) -> List[str]:
        recs = []
        if any(f.severity.value == BloodHealthStatus.CRITICAL for f in self.findings):
            recs.append("CẢNH BÁO: Cần tham vấn bác sĩ và xét nghiệm CBC đầy đủ.")
        if any(f.severity.value == BloodHealthStatus.ABNORMAL for f in self.findings):
            recs.append("Thực hiện xét nghiệm máu chi tiết (CBC, công thức máu).")
        if any("Thiếu" in f.condition or "Giảm" in f.condition for f in self.findings):
            recs.append("Xét nguyên nhân thiếu (dinh dưỡng, tủy xương, mất máu...).")
        if any("Tăng" in f.condition for f in self.findings):
            recs.append("Tìm dấu hiệu nhiễm trùng, viêm hoặc bệnh nền.")
        if not recs:
            recs.append("Tiếp tục theo dõi sức khỏe định kỳ.")
        recs.append("Kết quả chỉ mang tính hỗ trợ sàng lọc — không thay thế chẩn đoán y khoa.")
        return recs

    def _generate_details(
        self, cell_counts: Dict[str, int], cell_percentages: Dict[str, float]
    ) -> str:
        lines = [
            f"Tổng tế bào phát hiện: {sum(cell_counts.values())}",
            f"  RBC: {cell_counts.get('RBC', 0)} ({cell_percentages.get('RBC', 0):.2f}%)",
            f"  WBC: {cell_counts.get('WBC', 0)} ({cell_percentages.get('WBC', 0):.2f}%)",
            f"  Platelets: {cell_counts.get('Platelets', 0)} ({cell_percentages.get('Platelets', 0):.2f}%)",
        ]
        if self.findings:
            lines.append(f"\nPhát hiện {len(self.findings)} dấu hiệu:")
            for i, f in enumerate(self.findings, 1):
                lines.append(f"  {i}. {f.condition} ({f.severity.value})")
        else:
            lines.append("\nTất cả tỷ lệ nằm trong khoảng tham chiếu lam máu ngoại vi.")
        return "\n".join(lines)
