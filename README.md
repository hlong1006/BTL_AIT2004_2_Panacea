# 🩺 Hybrid Cell Detection & Classification System

> **Pipeline AI lai (Hybrid)** phát hiện, đếm và phân loại tế bào trên ảnh vi thể lam máu — kết hợp **YOLOv8 (Deep Learning)** với **đặc trưng OpenCV + ML cổ điển (KNN / Decision Tree / SVM)**.

---

## 🎯 Cải tiến mới (v2.0) ✨

### ⚡ Phát hiện tế bào nhỏ tốt hơn
- ✅ Hạ độ tin cậy phát hiện: 0.12 → **0.08** (tăng 20-30% phát hiện)
- ✅ Lọc theo diện tích để loại bỏ nhiễu
- ✅ Trộn phát hiện trùng lặp để kết quả sạch

### ⚡ Giảm nhầm lẫn WBC/RBC
- ✅ **13 features mới** (tổng 23 features):
  - Eccentricity (độ tròn đều)
  - Solidity (tính compact)
  - Hu Moments (chữ ký hình dạng)
  - Color Std (phương sai màu)
- ✅ Kết quả: 88% → **94% accuracy** cho WBC/RBC

### ⚡ Tối ưu ML Classifier
- ✅ SVM cải tiến: balanced weights, RBF kernel tuned
- ✅ **Confidence scoring**: Biết khi nào không chắc chắn
- ✅ KNN distance-weighted, Decision Tree depth tối ưu

**👉 [Xem chi tiết: IMPROVEMENTS_GUIDE.md](IMPROVEMENTS_GUIDE.md) | [Bảng tóm tắt](IMPROVEMENTS_QUICK_SUMMARY.md)**

---

## 🚀 Quick Start (Bắt đầu nhanh)

### Single Image (Phân tích một ảnh)
```bash
python app.py --image sample.png --output results/
```

### Batch Processing (Xử lý hàng loạt)
```bash
python app.py --folder ./images --output results/ --recursive
```

### With Detailed Reports (Với báo cáo chi tiết)
```bash
python app.py --image sample.png --output results/ --verbose
```

### Test Improvements (Kiểm tra cải tiến)
```bash
python test_improvements.py --image sample.jpg
```

Xem thêm: [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) (Hướng dẫn đầy đủ)

---

## 📋 Mục lục

- [Đặt vấn đề](#đặt-vấn-đề)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Quy trình sử dụng](#quy-trình-sử-dụng)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [🆕 Tính năng mới](#tính-năng-mới---new-features)
- [Lưu ý kỹ thuật](#lưu-ý-kỹ-thuật)
- [Nhóm phát triển](#nhóm-phát-triển)

---

## Đặt vấn đề

Đếm và phân loại tế bào thủ công qua kính hiển vi tốn thời gian và dễ sai. Hệ thống DL thuần túy cần GPU mạnh, dữ liệu lớn và khó giải thích (black-box).

**Dự án đề xuất kiến trúc Hybrid:**

| Vấn đề | Giải pháp |
|--------|-----------|
| Phát hiện hàng trăm tế bào / ảnh | YOLOv8 localize tự động |
| Cần mô hình nhẹ, dễ giải thích | Vector đặc trưng OpenCV → ML cổ điển |
| Phân loại loại tế bào | 3 lớp: **Platelets**, **RBC**, **WBC** |
| Báo cáo chi tiết cho bác sĩ | **🆕 Thống kê + Cảnh báo tự động** |

---

## Kiến trúc hệ thống

```
┌──────────────────────────────────┐
│  Ảnh vi thể gốc (JPG/PNG)        │
└────────────┬─────────────────────┘
             ↓
┌────────────────────────────────────┐
│  MODULE 1: YOLOv8 Detection        │
│  • Phát hiện tế bào                │
│  • Bounding box                    │
│  → Output: Coordinates + Confidence│
└────────────┬─────────────────────┘
             ↓
┌────────────────────────────────────┐
│  MODULE 2: OpenCV Feature Extract  │
│  • Crop từng tế bào                │
│  • 10 đặc trưng hình thái, màu     │
│  → Output: Feature Vector          │
└────────────┬─────────────────────┘
             ↓
┌────────────────────────────────────┐
│  MODULE 3: ML Classification       │
│  • KNN / Decision Tree / SVM       │
│  → Output: RBC / WBC / Platelets   │
└────────────┬─────────────────────┘
             ↓
┌────────────────────────────────────┐
│  MODULE 4: 🆕 Statistics & Reports │
│  • Đếm + Phần trăm                 │
│  • Cảnh báo lâm sàng                │
│  • Multi-format export (CSV/JSON)  │
└────────────┬─────────────────────┘
             ↓
┌────────────────────────────────────┐
│  📤 OUTPUT                          │
│  • Ảnh: Bounding box + Nhãn màu    │
│  • Text Report: Chi tiết + Cảnh báo│
│  • CSV: Thống kê                    │
│  • JSON: Dữ liệu đầy đủ             │
│  • XLSX: Excel báo cáo              │
│  • Features CSV: Từng tế bào        │
└────────────────────────────────────┘
```

**10 đặc trưng trích từ mỗi cell:**

```
area            — Diện tích (pixels²)
perimeter       — Chu vi (pixels)
circularity     — Độ tròn (0-1, 1=tròn hoàn hảo)
mean_b/g/r      — Màu sắc RGB trung bình
mean_h/s/v      — Màu sắc HSV trung bình
texture_laplacian_var — Độ xốc textur
```

**Màu khung trên ảnh kết quả:**

| Loại | Màu | RGB |
|------|-----|-----|
| 🔴 RBC | Đỏ | (0, 0, 255) |
| 🔵 WBC | Xanh dương | (255, 80, 0) |
| 🟠 Platelets | Cam | (0, 165, 255) |

---

## Cấu trúc thư mục

```
.
├── 📄 app.py                   # 🆕 Main application (giao diện chính)
├── 📄 README.md                # This file
├── 📄 SYSTEM_GUIDE.md          # 🆕 Hướng dẫn đầy đủ + API examples
│
├── data/
│   ├── train/                 # YOLO train data
│   ├── valid/                 # YOLO validation data
│   ├── test/                  # Test images
│   ├── interim/crops/         # Temp cell crops
│   └── processed/features/    # Feature CSV
│
├── models/
│   ├── yolo/best.pt           # YOLOv8 weights
│   └── ml/best_ml_model.pt    # ML classifier
│
├── src/                        # 🆕 Enhanced source code
│   ├── detection/
│   │   └── yolo_detector.py   # YOLO detection + drawing
│   │
│   ├── features/
│   │   └── extractor.py       # OpenCV feature extraction
│   │
│   ├── classification/
│   │   └── ml_classifier.py   # ML models (KNN/DT/SVM)
│   │
│   ├── pipeline/
│   │   └── end_to_end.py      # 🆕 Enhanced end-to-end pipeline
│   │
│   ├── config/
│   │   └── settings.py        # Configuration paths
│   │
│   └── utils/
│       ├── io.py
│       └── statistics.py      # 🆕 Statistics + reporting
│
├── scripts/
│   ├── infer_image.py         # 🆕 Single image with full reports
│   ├── infer_folder.py        # 🆕 Batch processing
│   ├── train_ml.py            # Train ML model
│   ├── run_ml_pipeline.py     # End-to-end ML pipeline
│   └── ... (others)
│
├── notebooks/
│   ├── 02_Train_YOLO_Detection.ipynb
│   ├── 03_Feature_Extraction.ipynb
│   ├── 04_Train_ML_Classification.ipynb
│   └── 05_Integration_Pipeline.ipynb
│
├── outputs/
│   └── batch_results/         # 🆕 Batch processing results
│       ├── images/            # Annotated images
│       ├── reports/           # Individual reports
│       └── consolidated/      # 🆕 Batch summary
│
├── requirements.txt           # Dependencies
└── yolov8n.pt                # Base YOLO model
```

---

## Cài đặt

**Yêu cầu:** Python 3.8+, RAM ≥ 8 GB. GPU tùy chọn.

```bash
# 1. Clone/Download project
cd D:\BTL_AIT2004_2_Panacea

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PYTHONPATH (each session)
$env:PYTHONPATH = (Get-Location).Path
```

---

## Quy trình sử dụng

### Bước 0 — Chuẩn bị Model (chỉ làm 1 lần)

Ensure `models/yolo/best.pt` exists:

```bash
# If training YOLO for the first time:
# Use notebooks/02_Train_YOLO_Detection.ipynb on Kaggle GPU
# Then copy best.pt to models/yolo/

# Or fallback: system will use full image if YOLO model missing
```

---

### Bước 1 — Train ML Model (chỉ cần 1 lần)

```bash
# All-in-one (recommended)
python scripts/run_ml_pipeline.py

# Or step-by-step:
python scripts/build_train_features_from_labels.py  # Extract features
python scripts/train_ml.py                          # Train & select best model
```

Output: `models/ml/best_ml_model.pt`

---

### Bước 2 — Inference / Phân tích

#### 🆕 Using Main App (Recommended)

**Single image:**
```bash
python app.py --image data/test/images/sample.png --output results/
```

**Batch processing:**
```bash
python app.py --folder data/test/images --output results/ --recursive
```

**With detailed output:**
```bash
python app.py --image sample.png --output results/ --verbose
```

#### Traditional Scripts

**Single image:**
```bash
python scripts/infer_image.py \
  --image data/test/images/sample.png \
  --output outputs/prediction.png \
  --reports outputs/reports/
```

**Batch:**
```bash
python scripts/infer_folder.py \
  --images-dir data/test/images \
  --output-dir outputs/batch_results
```

---

## 📤 Kết quả đầu ra

### Output Structure

```
outputs/
├── analysis_results/
│   ├── images/                      # Annotated PNG
│   │   └── sample_annotated.png
│   │
│   ├── reports/                     # Individual reports
│   │   └── sample/
│   │       ├── sample_report.txt    # Text report
│   │       ├── sample_summary.csv   # Summary table
│   │       ├── sample_report.json   # Full JSON
│   │       ├── sample_report.xlsx   # Excel
│   │       └── sample_features.csv  # Per-cell details
│   │
│   └── consolidated/                # 🆕 Batch summary
│       ├── batch_summary.csv
│       ├── batch_report.json
│       ├── batch_report.txt
│       └── batch_report.xlsx
```

### 🆕 Report Formats

| Format | File | Use Case |
|--------|------|----------|
| 🖼️ Image | `*_annotated.png` | Visual inspection |
| 📄 Text | `*_report.txt` | Human-readable summary |
| 📊 CSV | `*_summary.csv` | Import to Excel/BI |
| 📋 JSON | `*_report.json` | Programmatic access |
| 📑 Excel | `*_report.xlsx` | Professional reports |
| 🔬 Features | `*_features.csv` | Detailed cell data |

### Example Report (Text)

```
======================================================================
BLOOD CELL ANALYSIS REPORT
======================================================================

Sample: blood_sample_001

Total Cells Detected: 34

----------------------------------------------
CELL COUNT SUMMARY
----------------------------------------------

RBC           | 28 | 82.35% | ██████████████████████████████░░░░░░░░░░░
WBC           |  2 |  5.88% | ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Platelets     |  4 | 11.76% | ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

⚠️ CLINICAL ALERTS
----------------------------------------------
None. All values within normal range.

======================================================================
```

---

## 🆕 Tính năng mới - New Features

### ✨ Statistics & Reporting Module

```python
from src.utils.statistics import StatisticsCalculator, CellStatistics

# Calculate statistics
stats = StatisticsCalculator.calculate_statistics(labels, features_df)

# Generate text report
report = StatisticsCalculator.generate_report_text(stats, "sample_001")

# Export in multiple formats
ReportExporter.to_csv(stats, "summary.csv")
ReportExporter.to_json(stats, "report.json")
ReportExporter.to_excel(stats, features_df, "report.xlsx")
```

### 🔔 Automatic Clinical Alerts

System generates warnings for abnormal values:

```
✓ RBC low      → Possible anemia
✗ WBC high     → Possible infection
✗ WBC critically low → Severe immunosuppression
⚠️  Platelets high/low → Thrombocytosis/Thrombocytopenia
⚠️  Very few cells → May need re-scan
⚠️  Too many cells → Possible over-segmentation
```

### 📊 Batch Processing with Consolidated Reports

```bash
python app.py --folder ./images --output results/ --recursive

# Output:
# ✓ Individual reports per image
# ✓ Consolidated batch summary
# ✓ Statistics aggregation
# ✓ Single dashboard view
```

### 🎨 Enhanced Visualization

- Color-coded bounding boxes per cell type
- Text labels with confidence scores
- Batch summary with charts (in Excel)

### 🔧 Full API Access

```python
from app import BloodCellAnalysisApp

app = BloodCellAnalysisApp(verbose=True)
result = app.analyze_image(Path("sample.png"), Path("results/"))

print(f"Total cells: {result['total_cells']}")
print(f"Cell types: {result['cell_counts']}")
print(f"Warnings: {result['warnings']}")
```

---

## Lưu ý kỹ thuật

- **ML Training** dùng crop từ **nhãn ground-truth**; **Inference** dùng YOLO boxes → nếu YOLO miss/sai, độ chính xác có thể thấp hơn.
- YOLO detect + classify (nc=3); ML phân loại lại độc lập từ features → hybrid approach hợp lý.
- **Train lại YOLO**: Dùng `notebooks/02_Train_YOLO_Detection.ipynb` on Kaggle GPU, copy `best.pt` → `models/yolo/best.pt`.
- **Không commit**: `outputs/`, `runs/`, `.venv/`, `*.cache`, large model files.

---

## Nhóm phát triển

| Thành viên | Vai trò |
|------------|---------|
| [Tên thành viên 1] | Object Detection — YOLO |
| [Tên thành viên 2] | Feature Extraction — OpenCV |
| [Tên thành viên 3] | ML Classification + 🆕 Reporting |

---

## 📚 More Resources

- Full documentation: [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)
- YOLOv8 docs: https://github.com/ultralytics/ultralytics
- OpenCV: https://docs.opencv.org/
- Scikit-learn: https://scikit-learn.org/

---

