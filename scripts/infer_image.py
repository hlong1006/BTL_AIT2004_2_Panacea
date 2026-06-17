import argparse
from pathlib import Path

from src.classification.ml_classifier import ML_MODEL_FILENAME
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quy trình lai phát hiện - phân loại tế bào với tính năng báo cáo thống kê đầy đủ."
    )
    parser.add_argument("--image", required=True, type=Path, help="Đường dẫn đến ảnh hiển vi đầu vào")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prediction.png"),
        help="Đường dẫn lưu ảnh đầu ra đã chú thích",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("outputs/reports"),
        help="Thư mục để lưu các báo cáo CSV/JSON/TXT",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PATHS.yolo_models / "best.pt",
        help="Đường dẫn đến trọng số mô hình YOLO",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=PATHS.ml_models / ML_MODEL_FILENAME,
        help="Đường dẫn đến bộ phân loại ML đã huấn luyện",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Bỏ qua việc lưu báo cáo (chỉ lưu ảnh kết quả)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In thông tin đầu ra chi tiết",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Xác thực ảnh đầu vào
    if not args.image.exists():
        print(f" Lỗi: Không tìm thấy ảnh: {args.image}")
        return
    
    # Tải quy trình (pipeline)
    yolo_path = args.yolo_model if args.yolo_model.exists() else None
    if args.verbose:
        print(f" Đang tải quy trình phân tích...")
        if yolo_path:
            print(f"    Mô hình YOLO: {yolo_path}")
        else:
            print(f"     Không tìm thấy mô hình YOLO (sẽ sử dụng phương án dự phòng)")
        print(f"    Mô hình ML: {args.ml_model}")
    
    try:
        pipeline = HybridCellPipeline(yolo_model_path=yolo_path, ml_model_path=args.ml_model)
    except Exception as e:
        print(f" Lỗi khi tải quy trình: {e}")
        return
    
    # Chạy suy luận với đầy đủ thống kê
    try:
        if args.verbose:
            print(f"\n Đang xử lý ảnh: {args.image.name}")
        
        output_reports_dir = None if args.no_reports else args.reports
        result = pipeline.run_on_image_full(
            image_path=args.image,
            output_image_path=args.output,
            output_stats_dir=output_reports_dir,
        )
        
        # In kết quả ra màn hình
        print("\n" + "=" * 70)
        print(" QUÁ TRÌNH SUY LUẬN ĐÃ HOÀN THÀNH THÀNH CÔNG")
        print("=" * 70)
        print(result.report_text)
        print("=" * 70)
        
        # In vị trí lưu các tệp đầu ra
        print(f"\n Các tệp kết quả đầu ra:")
        print(f"    Ảnh chú thích:   {result.image_path}")
        
        if output_reports_dir:
            print(f"    Thư mục báo cáo: {output_reports_dir}")
            print(f"      ├─ {args.image.stem}_report.txt")
            print(f"      ├─ {args.image.stem}_summary.csv")
            print(f"      ├─ {args.image.stem}_report.json")
            print(f"      ├─ {args.image.stem}_report.xlsx")
            print(f"      └─ {args.image.stem}_features.csv")
        
    except Exception as e:
        print(f" Gặp lỗi trong quá trình suy luận: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()