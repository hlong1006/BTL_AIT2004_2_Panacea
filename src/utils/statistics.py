from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
from collections import Counter
import pandas as pd
import json

@dataclass
class CellStatistics:
    """Thực thể lưu trữ và quản lý dữ liệu thống kê hình thái tế bào máu.
    
    Cấu trúc dữ liệu này phục vụ tầng phân tích dữ liệu tổng hợp từ Pipeline lai,
    hỗ trợ lưu trữ số lượng phân bố, phân tách phân hệ cảnh báo kỹ thuật và lâm sàng.
    """
    
    total_cells: int = 0
    cell_counts: Dict[str, int] = field(default_factory=dict)
    cell_percentages: Dict[str, float] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Phân hệ lưu trữ cảnh báo tách biệt theo kiến trúc phần mềm y tế
    system_warnings: List[str] = field(default_factory=list)
    clinical_warnings: List[str] = field(default_factory=list)
    
    @property
    def warnings(self) -> List[str]:
        """Thuộc tính ảo (Property) gộp hai danh sách cảnh báo.
        
        Đảm bảo tính tương thích ngược (Backward Compatibility) đối với các module
        giao diện (UI) hoặc API cũ đang gọi trường dữ liệu .warnings.
        """
        return self.clinical_warnings + self.system_warnings
    
    def add_system_warning(self, warning: str) -> None:
        """Ghi nhận cảnh báo liên quan đến chất lượng dữ liệu đầu vào hoặc lỗi thuật toán."""
        if warning not in self.system_warnings:
            self.system_warnings.append(warning)
            
    def add_clinical_warning(self, warning: str) -> None:
        """Ghi nhận cảnh báo bất thường về phân bố mật độ tế bào theo tiêu chuẩn lâm sàng."""
        if warning not in self.clinical_warnings:
            self.clinical_warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thực thể sang cấu trúc từ điển (Dictionary).
        
        Đã được sửa lỗi logic (bug) để đảm bảo trường dữ liệu 'warnings' mở rộng
        được tích hợp chính xác vào dữ liệu trả về cho API Web.
        """
        data = asdict(self)
        data["warnings"] = self.warnings
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Chuỗi hóa cấu trúc thống kê sang định dạng JSON phục vụ truyền tải qua API."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi dữ liệu thống kê phân bố sang cấu trúc bảng Pandas DataFrame."""
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
    """Hệ chuyên gia toán học tính toán chỉ số và phân tích logic lâm sàng.
    
    Chịu trách nhiệm xử lý dữ liệu sau thực nghiệm (Post-processing) từ kết quả
    dự đoán của mô hình học máy và đặc trưng trích xuất từ thị trường (Field of View).
    """
    
    @staticmethod
    def calculate_statistics(labels: List[str], features_df: pd.DataFrame) -> CellStatistics:
        """Tính toán tổng hợp các chỉ số hình thái học và kích hoạt bộ lọc kiểm tra.
        
        Args:
            labels (List[str]): Danh sách nhãn phân loại thu được từ mô hình dự đoán.
            features_df (pd.DataFrame): Bảng chứa các thuộc tính hình thái trích xuất qua OpenCV.
            
        Returns:
            CellStatistics: Đối tượng chứa toàn bộ kết quả phân tích hệ thống.
        """
        stats = CellStatistics()
        stats.total_cells = len(labels)
        
        # Thống kê tần suất xuất hiện của từng loại tế bào
        counter = Counter(labels)
        stats.cell_counts = dict(counter)
        
        # Tính toán tỷ lệ phân bố phần trăm của các tập hợp tế bào
        if stats.total_cells > 0:
            for cell_type, count in stats.cell_counts.items():
                stats.cell_percentages[cell_type] = (count / stats.total_cells) * 100
        
        # Xử lý và tính toán các giá trị đặc trưng hình thái học trung bình
        if features_df is not None and not features_df.empty:
            features_df["predicted_label"] = labels
            for cell_type in stats.cell_counts.keys():
                mask = features_df["predicted_label"] == cell_type
                cell_features = features_df[mask]
                
                if not cell_features.empty:
                    stats.feature_stats[cell_type] = {
                        "avg_area": float(cell_features.get("area", pd.Series([0])).mean()),
                        "avg_perimeter": float(cell_features.get("perimeter", pd.Series([0])).mean()),
                        "avg_circularity": float(cell_features.get("circularity", pd.Series([0])).mean()),
                        "avg_texture": float(cell_features.get("texture_laplacian_var", pd.Series([0])).mean()),
                    }
        
        # Module lọc nhiễu kỹ thuật: Phát hiện sự bất thường về diện tích mặt phẳng (tế bào dính chùm)
        if features_df is not None and not features_df.empty and "area" in features_df.columns:
            areas = features_df["area"].dropna()
            if len(areas) > 0:
                median_area = float(areas.median())
                huge_cells = areas[areas > median_area * 8]
                if len(huge_cells) > 0:
                    percentage_huge = (len(huge_cells) / stats.total_cells) * 100
                    stats.add_system_warning(
                        f"Chất lượng mẫu thấp: Phát hiện {len(huge_cells)} cụm tế bào bị vón cục "
                        f"(chiếm {percentage_huge:.1f}% vi trường). Thuật toán phân đoạn có thể đếm thiếu số lượng thực tế. "
                        f"Khuyến nghị: Chuẩn bị lại tiêu bản máu mỏng hơn hoặc cấu hình lại tham số NMS."
                    )

        # Kích hoạt hệ thống luật để duyệt tìm các dấu hiệu lâm sàng bất thường
        StatisticsCalculator._generate_warnings(stats)
        return stats
    
    @staticmethod
    def _generate_warnings(stats: CellStatistics) -> None:
        """Hệ thống luật (Rule-based System) đánh giá tương quan tỷ lệ giữa các dòng tế bào.
        
        Áp dụng các ngưỡng toán học sinh học dựa trên mối tương quan với dòng Hồng cầu (RBC)
        để đưa ra các định hướng suy luận y khoa hỗ trợ người phân tích.
        """
        total = stats.total_cells
        
        # Kiểm tra ngưỡng giới hạn biên của thị trường (Field of View - FOV)
        if total < 5:
            stats.add_system_warning(f"Thị trường rỗng: Chỉ phát hiện {total} tế bào. Tiêu bản quá thưa hoặc ảnh bị mất nét.")
            return  # Đình chỉ phân tích lâm sàng nếu mật độ tế bào không đạt mức tối thiểu
            
        if total > 800:
            stats.add_system_warning(f"Mật độ tế bào cực cao ({total}): Cảnh báo rủi ro trùng lặp hoặc lỗi phân đoạn từ mô hình mạng.")

        # Lấy số liệu phân bố của 3 dòng tế bào cốt lõi
        rbc_count = stats.cell_counts.get("RBC", 0)
        plt_count = stats.cell_counts.get("Platelets", 0)
        wbc_count = stats.cell_counts.get("WBC", 0)
        
        if rbc_count > 0:
            plt_to_rbc_ratio = plt_count / rbc_count
            wbc_to_rbc_ratio = wbc_count / rbc_count
            
            # --- Khối logic đánh giá dòng Tiểu cầu (Platelets) ---
            if plt_count == 0:
                stats.add_clinical_warning(
                    f"Giảm tiểu cầu nghiêm trọng (Mức độ Nặng): Tỷ lệ PLT:RBC = 0:{rbc_count} (0.0%). "
                    f"Ngưỡng sinh lý tham chiếu tiêu chuẩn ~ 1:20. Nguy cơ xuất huyết cao. "
                    f"Khuyến nghị: Chỉ định kiểm tra tế bào học dòng tiểu cầu bằng máy đếm laser (CBC)."
                )
            elif plt_to_rbc_ratio < 0.025:
                stats.add_clinical_warning(
                    f"Giảm tiểu cầu (Theo dõi sinh lý): Tỷ lệ PLT:RBC hiện tại là {plt_count}:{rbc_count} ({plt_to_rbc_ratio*100:.1f}%). "
                    f"Dưới ngưỡng phân bố sinh lý trung bình (~ 5%). Cần đối chiếu thêm trên các vi trường lân cận."
                )
            elif plt_to_rbc_ratio > 0.2:
                stats.add_clinical_warning(
                    f"Tăng tiểu cầu bất thường: Tỷ lệ PLT:RBC tăng cao ({plt_to_rbc_ratio*100:.1f}%). "
                    f"Dấu hiệu cảnh báo phản ứng viêm cấp tính hoặc hội chứng tăng sinh tủy xương."
                )
                
            # --- Khối logic đánh giá dòng Bạch cầu (WBC) ---
            if wbc_to_rbc_ratio > 0.05:
                stats.add_clinical_warning(
                    f"Tăng bạch cầu bệnh lý (Leukocytosis): Tỷ lệ WBC:RBC đạt mức báo động {wbc_count}:{rbc_count}. "
                    f"Mật độ bạch cầu quá dày đặc trên một trường hiển vi, phản ánh phản ứng nhiễm trùng cấp hoặc bệnh lý ác tính hệ tạo máu."
                )
            elif wbc_count > 0 and wbc_to_rbc_ratio > 0.02:
                stats.add_clinical_warning(
                    f"Theo dõi động học Bạch cầu: Phát hiện mật độ WBC cao hơn phân bố sinh lý thông thường "
                    f"trên vi trường khảo sát (Tỷ lệ {wbc_count}:{rbc_count})."
                )
        else:
            stats.add_system_warning("Không tìm thấy Hồng cầu (RBC). Hệ thống mất điểm quy chiếu nền để tính toán chỉ số lâm sàng.")

    @staticmethod
    def generate_report_text(stats: CellStatistics, image_name: str = "Blood Sample") -> str:
        """Sinh tài liệu báo cáo phân tích định dạng văn bản chuẩn hóa.
        
        Tích hợp đầy đủ các phần biểu đồ phân bố, số liệu hình thái học trung bình,
        phân hệ kết luận lâm sàng và thông báo miễn trừ trách nhiệm y khoa.
        """
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO PHÂN TÍCH HÌNH THÁI TẾ BÀO MÁU (HYBRID AI PIPELINE)")
        report.append("=" * 70)
        report.append(f"\nTên tệp mẫu: {image_name}")
        report.append(f"Tổng số phần tử nhận diện: {stats.total_cells}")
        
        report.append("\n" + "-" * 70)
        report.append("TẦN SUẤT PHÂN BỐ CÁC DÒNG TẾ BÀO")
        report.append("-" * 70)
        sorted_cells = sorted(stats.cell_counts.items(), key=lambda x: x[1], reverse=True)
        for cell_type, count in sorted_cells:
            percentage = stats.cell_percentages.get(cell_type, 0.0)
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report.append(f"{cell_type:12} | {count:4d} | {percentage:6.2f}% | {bar}")
            
        # Trực quan hóa số liệu trích xuất đặc trưng hình thái
        if stats.feature_stats:
            report.append("\n" + "-" * 70)
            report.append("ĐẶC TRƯNG HÌNH THÁI HỌC TRUNG BÌNH (OpenCV Extraction)")
            report.append("-" * 70)
            for cell_type, features in stats.feature_stats.items():
                report.append(f"\n[{cell_type}]")
                report.append(f"  • Diện tích trung bình (Area):         {features.get('avg_area', 0):.2f} px²")
                report.append(f"  • Độ tròn trung bình (Circularity):   {features.get('avg_circularity', 0):.4f}")
        
        # Hiển thị phân hệ chẩn đoán logic lâm sàng
        report.append("\n" + "!" * 70)
        report.append("KẾT LUẬN LÂM SÀNG SƠ BỘ (CLINICAL FINDINGS)")
        report.append("!" * 70)
        if stats.clinical_warnings:
            for i, warning in enumerate(stats.clinical_warnings, 1):
                report.append(f" 🩺 {i}. {warning}")
        else:
            report.append(" ✓ Các chỉ số phân bố dòng tế bào trên vi trường này nằm trong giới hạn sinh lý bình thường.")

        # Hiển thị phân hệ cảnh báo chất lượng thuật toán và kỹ thuật ảnh
        if stats.system_warnings:
            report.append("\n" + "⚙️" * 35)
            report.append("ĐÁNH GIÁ CHẤT LƯỢNG KỸ THUẬT TIÊU BẢN (SYSTEM ALERTS)")
            report.append("⚙️" * 35)
            for i, warning in enumerate(stats.system_warnings, 1):
                report.append(f" ⚠️ {i}. {warning}")
                
        # THÊM ĐOẠN NÀY: Khối thông báo cấu trúc và miễn trừ trách nhiệm pháp lý y tế
        report.append("\n" + "=" * 70)
        report.append("LƯU Ý HỆ THỐNG / DISCLAIMER:")
        report.append("Hệ thống hiện đang hoạt động ở giai đoạn thử nghiệm thuật toán (Proof of Concept).")
        report.append("Các biểu hiện lâm sàng được suy luận tự động thông qua hệ chuyên gia dựa trên")
        report.append("quy tắc tương quan mật độ hình ảnh, chỉ có ý nghĩa hỗ trợ nghiên cứu viên.")
        report.append("Kết quả này KHÔNG THAY THẾ cho các quyết định chẩn đoán lâm sàng của bác sĩ.")
        report.append("Vui lòng đối chiếu với dữ liệu tổng phân tích tế bào máu ngoại vi thu được từ")
        report.append("hệ thống máy đếm tổng trở hoặc máy đếm laser chuyên dụng.")
        report.append("=" * 70)
        
        return "\n".join(report)


class ReportExporter:
    """Module phụ trợ chịu trách nhiệm kết xuất và lưu trữ tài liệu đa định dạng."""
    
    @staticmethod
    def to_csv(stats: CellStatistics, output_path: str) -> None:
        """Lưu trữ dữ liệu phân bố tóm tắt dưới định dạng phẳng CSV."""
        df = stats.to_dataframe()
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def to_json(stats: CellStatistics, output_path: str) -> None:
        """Lưu trữ toàn bộ thuộc tính cấu trúc đối tượng sang tệp JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(stats.to_json(indent=2))
    
    @staticmethod
    def to_txt(report_text: str, output_path: str) -> None:
        """Xuất tệp văn bản thuần báo cáo kết luận phân tích."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    @staticmethod
    def to_excel(stats: CellStatistics, features_df: pd.DataFrame, output_path: str) -> None:
        """Đóng gói dữ liệu thực nghiệm đa tầng vào bảng tính Excel (Multi-sheets)."""
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Sheet 1: Tổng quan phân bố tần suất
                summary_df = stats.to_dataframe()
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Sheet 2: Chi tiết 20 thông số hình thái học của từng phần tử tế bào
                if features_df is not None and not features_df.empty:
                    features_df.to_excel(writer, sheet_name="Cell Details", index=False)
                
                # Sheet 3: Nhật ký phân loại cảnh báo hệ thống và lâm sàng
                all_alerts = []
                for w in stats.clinical_warnings:
                    all_alerts.append({"Phân hệ": "Lâm sàng (Medical)", "Nội dung chỉ định": w})
                for w in stats.system_warnings:
                    all_alerts.append({"Phân hệ": "Hệ thống (System)", "Nội dung chỉ định": w})
                    
                if all_alerts:
                    warnings_df = pd.DataFrame(all_alerts)
                    warnings_df.to_excel(writer, sheet_name="Warnings", index=False)
        except ImportError:
            print("[Hệ thống] Thư viện openpyxl chưa được cài đặt. Tiến trình xuất Excel bị bỏ qua.")