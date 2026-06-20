from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
from collections import Counter

import pandas as pd
import json


@dataclass
class CellStatistics:
    """Lưu trữ số liệu thống kê tế bào tổng hợp cho phân tích mẫu máu."""
    
    total_cells: int = 0
    cell_counts: Dict[str, int] = field(default_factory=dict)
    cell_percentages: Dict[str, float] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        """Thêm thông báo cảnh báo vào phần thống kê."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thống kê sang dạng từ điển."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Chuyển đổi thống kê sang chuỗi JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi tóm tắt thống kê sang DataFrame."""
        data = []
        for cell_type, count in self.cell_counts.items():
            percentage = self.cell_percentages.get(cell_type, 0.0)
            data.append({
                "Cell Type": cell_type,
                "Count": count,
                "Percentage": f"{percentage:.2f}%"
            })
        return pd.DataFrame(data)


class StatisticsCalculator:
    """Tính toán thống kê từ các dự đoán về tế bào."""
    
    # Các khoảng giá trị bình thường cho số lượng tế bào máu (giá trị tham chiếu)
    NORMAL_RANGES = {
        "RBC": {"min": 4.5, "max": 5.5, "unit": "triệu/μL"},       # Hồng cầu
        "WBC": {"min": 4.5, "max": 11.0, "unit": "nghìn/μL"},      # Bạch cầu
        "Platelets": {"min": 150, "max": 400, "unit": "nghìn/μL"}, # Tiểu cầu
    }
    
    @staticmethod
    def calculate_statistics(
        labels: List[str],
        features_df: pd.DataFrame
    ) -> CellStatistics:
        """Tính toán các số liệu thống kê toàn diện từ các dự đoán."""
        
        stats = CellStatistics()
        stats.total_cells = len(labels)
        
        # Đếm tế bào theo loại
        counter = Counter(labels)
        stats.cell_counts = dict(counter)
        
        # Tính toán tỷ lệ phần trăm
        if stats.total_cells > 0:
            for cell_type, count in stats.cell_counts.items():
                percentage = (count / stats.total_cells) * 100
                stats.cell_percentages[cell_type] = percentage
        
        # Tính toán thống kê đặc trưng cho từng loại tế bào
        if features_df is not None and len(features_df) > 0:
            features_df["predicted_label"] = labels
            for cell_type in stats.cell_counts.keys():
                mask = features_df["predicted_label"] == cell_type
                cell_features = features_df[mask]
                
                if len(cell_features) > 0:
                    stats.feature_stats[cell_type] = {
                        "avg_area": float(cell_features.get("area", pd.Series(0)).mean()),
                        "avg_perimeter": float(cell_features.get("perimeter", pd.Series(0)).mean()),
                        "avg_circularity": float(cell_features.get("circularity", pd.Series(0)).mean()),
                        "avg_texture": float(cell_features.get("texture_laplacian_var", pd.Series(0)).mean()),
                    }
        
        # Tạo cảnh báo cho các giá trị bất thường
        if features_df is not None and len(features_df) > 0 and "area" in features_df.columns:
            areas = features_df["area"].dropna()
            if len(areas) > 0:
                median_area = float(areas.median())
                huge = areas[areas > median_area * 8]
                if len(huge) > 0:
                    stats.add_warning(
                        f"Phát hiện {len(huge)} vùng có kích thước bất thường lớn — "
                        "có thể là cụm tế bào chưa tách được, kết quả phân loại có thể không chính xác"
                    )

        StatisticsCalculator._generate_warnings(stats)
        
        return stats
    
    @staticmethod
    def _generate_warnings(stats: CellStatistics) -> None:
        """Tạo cảnh báo cho các giá trị bất thường."""
        
        total = stats.total_cells
        
        # Kiểm tra số lượng RBC thấp (thiếu máu)
        rbc_count = stats.cell_counts.get("RBC", 0)
        if total > 0:
            rbc_percentage = (rbc_count / total) * 100
            if rbc_percentage < 40:
                stats.add_warning(
                    f"RBC thấp: {rbc_count} tế bào ({rbc_percentage:.1f}%) - "
                    "Có thể là dấu hiệu thiếu máu hoặc rối loạn máu"
                )
        
        # Kiểm tra số lượng WBC cao (tăng bạch cầu)
        wbc_count = stats.cell_counts.get("WBC", 0)
        if total > 0:
            wbc_percentage = (wbc_count / total) * 100
            if wbc_percentage > 20:
                stats.add_warning(
                    f"WBC cao: {wbc_count} tế bào ({wbc_percentage:.1f}%) - "
                    "Có thể là dấu hiệu nhiễm trùng hoặc rối loạn miễn dịch"
                )
        
        # Kiểm tra số lượng WBC thấp (giảm bạch cầu)
        if total > 0 and wbc_percentage < 2:
            stats.add_warning(
                f"WBC cực thấp: {wbc_count} tế bào ({wbc_percentage:.1f}%) - "
                "Có thể là dấu hiệu suy giảm miễn dịch nghiêm trọng"
            )
        
        # Kiểm tra số lượng tiểu cầu bất thường
        platelet_count = stats.cell_counts.get("Platelets", 0)
        if total > 0:
            platelet_percentage = (platelet_count / total) * 100
            if platelet_percentage < 5:
                stats.add_warning(
                    f"Tiểu cầu cực thấp: {platelet_count} tế bào ({platelet_percentage:.1f}%) - "
                    "Có thể là dấu hiệu giảm tiểu cầu"
                )
            elif platelet_percentage > 25:
                stats.add_warning(
                    f"Tiểu cầu cao: {platelet_count} tế bào ({platelet_percentage:.1f}%) - "
                    "Có thể là dấu hiệu tăng tiểu cầu"
                )
        
        # Cảnh báo nếu phát hiện rất ít tế bào
        if total < 5:
            stats.add_warning(
                f"Phát hiện rất ít tế bào: {total} - "
                "Ảnh có thể quá mờ/nhạt hoặc cần tăng độ phóng đại kính hiển vi"
            )
        
        # Cảnh báo nếu số lượng tế bào quá nhiều (có thể do lỗi phân đoạn)
        if total > 500:
            stats.add_warning(
                f"Số lượng tế bào rất cao: {total} - "
                "Có thể cho thấy phân đoạn quá mức hoặc các tế bào bị kết tụ"
            )
    
    @staticmethod
    def generate_report_text(stats: CellStatistics, image_name: str = "Blood Sample") -> str:
        """Tạo báo cáo dạng văn bản."""
        
        report = []
        report.append("=" * 60)
        report.append("BÁO CÁO PHÂN TÍCH TẾ BÀO MÁU")
        report.append("=" * 60)
        report.append(f"\nMẫu: {image_name}")
        report.append(f"\nTổng số tế bào được phát hiện: {stats.total_cells}")
        report.append("\n" + "-" * 60)
        report.append("TÓM TẮT SỐ LƯỢNG TẾ BÀO")
        report.append("-" * 60)
        
        # Sắp xếp theo số lượng (giảm dần)
        sorted_cells = sorted(stats.cell_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cell_type, count in sorted_cells:
            percentage = stats.cell_percentages.get(cell_type, 0.0)
            bar_length = int(percentage / 2)  # Tỷ lệ hiển thị tối đa 50 ký tự
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report.append(f"\n{cell_type:12} | {count:4d} | {percentage:6.2f}% | {bar}")
        
        # Thống kê đặc trưng
        if stats.feature_stats:
            report.append("\n" + "-" * 60)
            report.append("ĐẶC TRƯNG HÌNH THÁI (Trung bình theo loại tế bào)")
            report.append("-" * 60)
            
            for cell_type, features in stats.feature_stats.items():
                report.append(f"\n{cell_type}:")
                report.append(f"   • Diện tích trung bình:     {features.get('avg_area', 0):.2f} px²")
                report.append(f"   • Chu vi trung bình:       {features.get('avg_perimeter', 0):.2f} px")
                report.append(f"   • Độ tròn trung bình:      {features.get('avg_circularity', 0):.4f}")
                report.append(f"   • Kết cấu trung bình:      {features.get('avg_texture', 0):.4f}")
        
        # Cảnh báo
        if stats.warnings:
            report.append("\n" + "!" * 60)
            report.append("CẢNH BÁO LÂM SÀNG")
            report.append("!" * 60)
            for i, warning in enumerate(stats.warnings, 1):
                report.append(f"\n{i}. {warning}")
        else:
            report.append("\n" + "✓" * 60)
            report.append("Không phát hiện kết quả bất thường.")
            report.append("✓" * 60)
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


class ReportExporter:
    """Xuất số liệu thống kê sang nhiều định dạng tệp khác nhau."""
    
    @staticmethod
    def to_csv(stats: CellStatistics, output_path: str) -> None:
        """Xuất thống kê sang CSV."""
        df = stats.to_dataframe()
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def to_json(stats: CellStatistics, output_path: str) -> None:
        """Xuất thống kê sang JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(stats.to_json(indent=2))
    
    @staticmethod
    def to_txt(report_text: str, output_path: str) -> None:
        """Xuất báo cáo văn bản sang tệp."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    @staticmethod
    def to_excel(stats: CellStatistics, features_df: pd.DataFrame, output_path: str) -> None:
        """Xuất sang Excel với nhiều trang tính (sheet)."""
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Trang tóm tắt
                summary_df = stats.to_dataframe()
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Trang chi tiết
                if features_df is not None and len(features_df) > 0:
                    features_df.to_excel(writer, sheet_name="Cell Details", index=False)
                
                # Trang cảnh báo
                if stats.warnings:
                    warnings_df = pd.DataFrame({
                        "Alert": stats.warnings
                    })
                    warnings_df.to_excel(writer, sheet_name="Warnings", index=False)
        except ImportError:
            print("Chưa cài đặt openpyxl. Bỏ qua xuất Excel.")