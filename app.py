#!/usr/bin/env python3
"""
Hệ thống Phát hiện & Phân loại Tế bào Máu
Giao diện Ứng dụng Chính

Ứng dụng này cung cấp một giao diện dễ sử dụng để phân tích tế bào máu,
kết hợp việc phát hiện bằng YOLOv8 với phân loại bằng Học máy (Machine Learning).

Cách sử dụng:
    python app.py --image sample.png
    python app.py --folder ./images
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Windows console: tránh lỗi Unicode khi in tiếng Việt
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo src có thể import được
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline
from src.utils.statistics import StatisticsCalculator, ReportExporter


class BloodCellAnalysisApp:
    """Ứng dụng chính để phân tích tế bào máu."""
    
    def __init__(
        self,
        yolo_model_path: Optional[Path] = None,
        ml_model_path: Optional[Path] = None,
        yolo_conf: float = 0.06,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 640,
        yolo_max_det: int = 1000,
        verbose: bool = False,
    ):
        """Khởi tạo ứng dụng."""
        self.verbose = verbose
        
        # Đặt các giá trị mặc định
        if yolo_model_path is None:
            yolo_model_path = PATHS.yolo_models / "best.pt"
        if ml_model_path is None:
            ml_model_path = PATHS.ml_models / ML_MODEL_FILENAME
        
        # Xác thực các đường dẫn
        if not ml_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình ML: {ml_model_path}")
        if not yolo_model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình YOLO: {yolo_model_path}\n"
                "Hãy đặt file best.pt vào models/yolo/best.pt (tải từ huấn luyện hoặc copy từ runs/cell_detect/weights/)."
            )
        
        if self.verbose:
            print(" Đang khởi tạo Hệ thống Phân tích Tế bào Máu...")
            print(f"   Mô hình YOLO: {yolo_model_path}")
            print(f"   Mô hình ML: {ml_model_path}")
        
        # Khởi tạo quy trình (pipeline)
        self.pipeline = HybridCellPipeline(
            yolo_model_path=yolo_model_path,
            ml_model_path=ml_model_path,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_imgsz=yolo_imgsz,
            yolo_max_det=yolo_max_det,
        )
    
    def analyze_image(
        self,
        image_path: Path,
        output_dir: Path,
        save_reports: bool = True,
    ) -> dict:
        """
        Phân tích một ảnh mẫu máu đơn lẻ.
        
        Tham số:
            image_path: Đường dẫn đến tệp ảnh
            output_dir: Thư mục để lưu kết quả
            save_reports: Có lưu các báo cáo chi tiết hay không
        
        Trả về:
            Dictionary chứa kết quả phân tích
        """
        # Xác thực đầu vào
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        
        if self.verbose:
            print(f"\n Đang phân tích: {image_path.name}")
        
        # Tạo đường dẫn đầu ra
        output_image_path = output_dir / "images" / f"{image_path.stem}_annotated.png"
        output_reports_dir = (output_dir / "reports" / image_path.stem) if save_reports else None
        
        # Chạy quy trình
        result = self.pipeline.run_on_image_full(
            image_path=image_path,
            output_image_path=output_image_path,
            output_stats_dir=output_reports_dir,
        )
        
        # Chuẩn bị đầu ra
        output = {
            "success": True,
            "image_name": image_path.name,
            "annotated_image": output_image_path,
            "total_cells": result.stats.total_cells,
            "cell_counts": result.stats.cell_counts,
            "cell_percentages": result.stats.cell_percentages,
            "warnings": result.stats.warnings,
            "report_text": result.report_text,
        }
        
        if save_reports and output_reports_dir:
            output["reports_dir"] = output_reports_dir
        
        return output
    
    def analyze_folder(
        self,
        folder_path: Path,
        output_dir: Path,
        recursive: bool = False,
        save_reports: bool = True,
    ) -> dict:
        """
        Phân tích nhiều ảnh trong một thư mục.
        
        Tham số:
            folder_path: Đường dẫn đến thư mục chứa ảnh
            output_dir: Thư mục để lưu kết quả
            recursive: Có tìm kiếm đệ quy trong các thư mục con hay không
            save_reports: Có lưu các báo cáo chi tiết hay không
        
        Trả về:
            Dictionary chứa kết quả phân tích hàng loạt
        """
        # Tìm ảnh
        pattern = "**/*" if recursive else "*"
        image_files = [
            f for f in folder_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        
        if not image_files:
            raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {folder_path}")
        
        if self.verbose:
            print(f"\n Đã tìm thấy {len(image_files)} ảnh")
        
        # Phân tích từng ảnh
        results = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "images": [],
            "total_cells": 0,
            "cell_counts": {},
        }
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                if self.verbose:
                    print(f"  [{idx}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)
                
                result = self.analyze_image(image_path, output_dir, save_reports=save_reports)
                result["index"] = idx
                results["images"].append(result)
                results["successful"] += 1
                results["total_cells"] += result["total_cells"]
                
                # Tổng hợp số lượng tế bào
                for cell_type, count in result["cell_counts"].items():
                    results["cell_counts"][cell_type] = results["cell_counts"].get(cell_type, 0) + count
                
                if self.verbose:
                    print(f" ({result['total_cells']} tế bào)")
                else:
                    print(".", end="", flush=True)
            
            except Exception as e:
                results["failed"] += 1
                if self.verbose:
                    print(f"Error {e}")
                else:
                    print("✗", end="", flush=True)
        
        # Tính số liệu trung bình
        if results["successful"] > 0:
            results["avg_cells_per_image"] = results["total_cells"] / results["successful"]
        else:
            results["avg_cells_per_image"] = 0
        
        return results
    
    def print_analysis_report(self, analysis_result: dict) -> None:
        """In kết quả phân tích ra bảng điều khiển (console)."""
        print("\n" + "=" * 70)
        print(" BÁO CÁO PHÂN TÍCH")
        print("=" * 70)
        
        if "report_text" in analysis_result:
            # Kết quả ảnh đơn
            print(analysis_result["report_text"])
        else:
            # Kết quả hàng loạt
            print(f"\nTổng số ảnh:           {analysis_result['total_images']}")
            print(f"Phân tích thành công:  {analysis_result['successful']}")
            print(f"Thất bại:              {analysis_result['failed']}")
            print(f"Tổng số tế bào phát hiện: {analysis_result['total_cells']}")
            print(f"Trung bình mỗi ảnh:    {analysis_result['avg_cells_per_image']:.1f}")
            
            print("\nPhân bổ loại tế bào:")
            for cell_type, count in analysis_result["cell_counts"].items():
                pct = (count / analysis_result["total_cells"] * 100) if analysis_result["total_cells"] > 0 else 0
                print(f"  {cell_type:15} {count:5d} ({pct:6.2f}%)")
        
        print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống Phát hiện & Phân loại Tế bào Máu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Phân tích một ảnh
  python app.py --image sample.png --output results/

  # Phân tích cả thư mục
  python app.py --folder images/ --output results/ --recursive

  # Với các mô hình tùy chỉnh
  python app.py --image sample.png --yolo models/custom_yolo.pt --ml models/custom_ml.pt

  # In chi tiết đầu ra
  python app.py --image sample.png --output results/ --verbose
        """
    )
    
    parser.add_argument(
        "--image",
        type=Path,
        help="Đường dẫn đến tệp ảnh đầu vào (chế độ ảnh đơn)",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Đường dẫn đến thư mục chứa ảnh (chế độ hàng loạt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis_results"),
        help="Thư mục đầu ra cho kết quả (mặc định: outputs/analysis_results)",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        help="Đường dẫn đến mô hình YOLO (mặc định: models/yolo/best.pt)",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        help="Đường dẫn đến mô hình ML (mặc định: models/ml/best_ml_model.pt)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Tìm kiếm đệ quy trong các thư mục con (cho chế độ thư mục)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Bỏ qua việc lưu các báo cáo chi tiết (chỉ lưu các ảnh đã chú thích)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết đầu ra",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.06,
        help="Ngưỡng độ tin cậy YOLO (mặc định: 0.06, thấp hơn = bắt tế bào nhỏ tốt hơn)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Ngưỡng IoU của YOLO NMS (mặc định: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước suy luận YOLO (mặc định: 640, cao hơn giúp bắt tế bào nhỏ)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Xác thực các đối số
    if not args.image and not args.folder:
        print(" Lỗi: Vui lòng chỉ định --image hoặc --folder")
        sys.exit(1)
    
    if args.image and args.folder:
        print(" Lỗi: Không thể chỉ định cả --image và --folder cùng lúc")
        sys.exit(1)
    
    try:
        # Khởi tạo ứng dụng
        app = BloodCellAnalysisApp(
            yolo_model_path=args.yolo_model,
            ml_model_path=args.ml_model,
            yolo_conf=args.conf,
            yolo_iou=args.iou,
            yolo_imgsz=args.imgsz,
            verbose=args.verbose,
        )
        
        # Tạo thư mục đầu ra
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Chạy phân tích
        if args.image:
            print(f" Hệ thống Phân tích Tế bào Máu - Chế độ Ảnh Đơn")
            print(f"   Ảnh: {args.image}")
            result = app.analyze_image(
                image_path=args.image,
                output_dir=args.output,
                save_reports=not args.no_reports,
            )
        else:
            print(f" Hệ thống Phân tích Tế bào Máu - Chế độ Xử lý Hàng loạt")
            print(f"   Thư mục: {args.folder}")
            result = app.analyze_folder(
                folder_path=args.folder,
                output_dir=args.output,
                recursive=args.recursive,
                save_reports=not args.no_reports,
            )
        
        # In kết quả
        app.print_analysis_report(result)
        
        # In vị trí đầu ra
        print(f"\n Các kết quả đã được lưu tại: {args.output}")
        
    except FileNotFoundError as e:
        print(f" Lỗi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f" Lỗi không xác định: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()