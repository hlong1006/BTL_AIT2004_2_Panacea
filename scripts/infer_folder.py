"""
Tập lệnh xử lý hàng loạt để phân tích nhiều ảnh mẫu máu.
Xử lý tất cả các ảnh trong một thư mục và tạo các báo cáo tổng hợp.
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline
from src.utils.statistics import StatisticsCalculator, ReportExporter


class BatchProcessor:
    """Xử lý nhiều ảnh và tạo báo cáo tổng hợp."""
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    def __init__(self, pipeline: HybridCellPipeline, verbose: bool = False):
        self.pipeline = pipeline
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
    
    def process_folder(
        self,
        input_folder: Path,
        output_folder: Path,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        Xử lý tất cả các ảnh trong một thư mục.
        
        Tham số:
            input_folder: Thư mục chứa ảnh
            output_folder: Thư mục để lưu kết quả
            recursive: Nếu True, tìm kiếm cả trong các thư mục con
        
        Trả về:
            Thống kê tóm tắt cho toàn bộ đợt xử lý
        """
        # Tìm tất cả các ảnh
        pattern = "**/*" if recursive else "*"
        image_files = [
            f
            for f in input_folder.glob(pattern)
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS and f.is_file()
        ]
        
        if not image_files:
            print(f" Không tìm thấy ảnh nào trong {input_folder}")
            return {}
        
        print(f" Đã tìm thấy {len(image_files)} ảnh")
        
        # Xử lý từng ảnh
        for idx, image_path in enumerate(tqdm(image_files, desc="Đang xử lý ảnh"), 1):
            self._process_single_image(image_path, output_folder, idx, len(image_files))
        
        # Tạo báo cáo tổng hợp
        summary = self._generate_consolidated_report(output_folder)
        
        return summary
    
    def _process_single_image(
        self,
        image_path: Path,
        output_folder: Path,
        current: int,
        total: int,
    ) -> None:
        """Xử lý một ảnh đơn và lưu kết quả."""
        
        try:
            # Xác định đường dẫn đầu ra
            output_image_path = output_folder / "images" / f"{image_path.stem}_annotated.png"
            output_reports_dir = output_folder / "reports" / image_path.stem
            
            # Chạy quy trình (pipeline)
            result = self.pipeline.run_on_image_full(
                image_path=image_path,
                output_image_path=output_image_path,
                output_stats_dir=output_reports_dir,
            )
            
            # Lưu kết quả
            self.results.append({
                "image_name": image_path.name,
                "image_stem": image_path.stem,
                "total_cells": result.stats.total_cells,
                "cell_counts": result.stats.cell_counts,
                "cell_percentages": result.stats.cell_percentages,
                "warnings": len(result.stats.warnings),
                "has_warnings": len(result.stats.warnings) > 0,
            })
        
        except Exception as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def _generate_consolidated_report(self, output_folder: Path) -> Dict[str, Any]:
        """Tạo báo cáo tổng hợp cho tất cả các ảnh đã xử lý."""
        
        if not self.results:
            return {}
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(self.results)
        
        # Tính toán thống kê hàng loạt
        total_images = len(self.results)
        total_cells = df["total_cells"].sum()
        avg_cells_per_image = total_cells / total_images if total_images > 0 else 0
        images_with_warnings = df["has_warnings"].sum()
        
        # Tính trung bình phân bổ các loại tế bào
        all_counts = {}
        for counts in df["cell_counts"]:
            for cell_type, count in counts.items():
                all_counts[cell_type] = all_counts.get(cell_type, 0) + count
        
        avg_counts = {ct: count / total_images for ct, count in all_counts.items()}
        
        # Tạo bản tóm tắt
        summary = {
            "processed_date": datetime.now().isoformat(),
            "total_images": total_images,
            "total_cells_detected": total_cells,
            "avg_cells_per_image": avg_cells_per_image,
            "cell_type_totals": all_counts,
            "avg_cells_per_type": avg_counts,
            "images_with_alerts": images_with_warnings,
            "alert_rate": (images_with_warnings / total_images * 100) if total_images > 0 else 0,
        }
        
        # Lưu các báo cáo tổng hợp
        reports_dir = output_folder / "consolidated"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Tóm tắt CSV
        df.to_csv(reports_dir / "batch_summary.csv", index=False)
        
        # Tóm tắt JSON
        import json
        with open(reports_dir / "batch_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Báo cáo văn bản
        self._save_text_report(summary, df, reports_dir / "batch_report.txt")
        
        # Báo cáo Excel
        try:
            with pd.ExcelWriter(reports_dir / "batch_report.xlsx", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Thêm trang tính thống kê
                stats_data = {
                    "Metric": list(summary.keys()),
                    "Value": [str(v) for v in summary.values()],
                }
                pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)
        except ImportError:
            print(" openpyxl chưa được cài đặt. Bỏ qua xuất Excel.")
        
        return summary
    
    @staticmethod
    def _save_text_report(summary: Dict[str, Any], df: pd.DataFrame, output_path: Path) -> None:
        """Lưu báo cáo văn bản tổng hợp."""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("BÁO CÁO XỬ LÝ HÀNG LOẠT - PHÂN TÍCH TẾ BÀO MÁU\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Ngày xử lý: {summary['processed_date']}\n\n")
            
            # Thống kê tóm tắt
            f.write("THỐNG KÊ TÓM TẮT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Tổng số ảnh đã xử lý:       {summary['total_images']}\n")
            f.write(f"Tổng số tế bào phát hiện:   {summary['total_cells_detected']}\n")
            f.write(f"Trung bình tế bào/Ảnh:      {summary['avg_cells_per_image']:.1f}\n")
            f.write(f"Ảnh có cảnh báo:            {summary['images_with_alerts']}/{summary['total_images']} ({summary['alert_rate']:.1f}%)\n\n")
            
            # Thống kê loại tế bào
            f.write("THỐNG KÊ LOẠI TẾ BÀO (Tổng trên tất cả các ảnh)\n")
            f.write("-" * 70 + "\n")
            for cell_type, count in summary["cell_type_totals"].items():
                avg = summary["avg_cells_per_type"].get(cell_type, 0)
                f.write(f"{cell_type:15} Tổng: {count:6d}  |  Trung bình: {avg:6.1f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("KẾT QUẢ CHI TIẾT TỪNG ẢNH\n")
            f.write("-" * 70 + "\n\n")
            
            # Kết quả chi tiết
            for _, row in df.iterrows():
                f.write(f"Ảnh: {row['image_name']}\n")
                f.write(f"  Tổng số tế bào: {row['total_cells']}\n")
                for cell_type, count in row['cell_counts'].items():
                    pct = row['cell_percentages'].get(cell_type, 0)
                    f.write(f"    {cell_type}: {count} ({pct:.1f}%)\n")
                if row['has_warnings']:
                    f.write(f"    Cảnh báo: {row['warnings']}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý hàng loạt nhiều ảnh tế bào máu kèm theo báo cáo tổng hợp."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PATHS.root / "data" / "test" / "images",
        help="Thư mục chứa ảnh đầu vào",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "outputs" / "batch_results",
        help="Thư mục đầu ra cho kết quả",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Đường dẫn đến mô hình YOLO",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=PATHS.ml_models / ML_MODEL_FILENAME,
        help="Đường dẫn đến mô hình ML",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Tìm kiếm đệ quy trong các thư mục con",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết đầu ra",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Xác thực thư mục đầu vào
    if not args.images_dir.exists():
        print(f" Lỗi: Không tìm thấy thư mục đầu vào: {args.images_dir}")
        return
    
    print(f" Chế độ xử lý hàng loạt")
    print(f" Đầu vào:  {args.images_dir}")
    print(f" Đầu ra: {args.output_dir}")
    
    # Khởi tạo quy trình
    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    
    try:
        pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)
    except Exception as e:
        print(f" Lỗi khi tải quy trình: {e}")
        return
    
    # Xử lý hàng loạt
    processor = BatchProcessor(pipeline, verbose=args.verbose)
    summary = processor.process_folder(
        args.images_dir,
        args.output_dir,
        recursive=args.recursive,
    )
    
    if summary:
        print("\n" + "=" * 70)
        print(" ĐÃ HOÀN THÀNH XỬ LÝ HÀNG LOẠT")
        print("=" * 70)
        print(f"\nTổng số ảnh:      {summary.get('total_images', 0)}")
        print(f"Tổng số tế bào:   {summary.get('total_cells_detected', 0)}")
        print(f"Trung bình/Ảnh:   {summary.get('avg_cells_per_image', 0):.1f}")
        print(f"Ảnh có cảnh báo:  {summary.get('images_with_alerts', 0)} ({summary.get('alert_rate', 0):.1f}%)")
        print(f"\n Báo cáo: {args.output_dir / 'consolidated'}")
        print("=" * 70)


if __name__ == "__main__":
    main()