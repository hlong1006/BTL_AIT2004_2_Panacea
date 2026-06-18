# Hệ thống Phát hiện và Phân loại Tế bào Máu

> Hệ thống phát hiện và phân loại tế bào máu tự động sử dụng mô hình YOLOv8 kết hợp với các thuật toán Machine Learning nhằm hỗ trợ phân tích ảnh lam máu.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Mục tiêu](#mục-tiêu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Phương pháp thực hiện](#phương-pháp-thực-hiện)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Trạng thái dự án](#trạng-thái-dự-án)
- [Lưu ý kỹ thuật](#lưu-ý-kỹ-thuật)
- [Hướng phát triển](#hướng-phát-triển)
- [Thành viên nhóm](#thành-viên-nhóm)

---

## Giới thiệu

Trong xét nghiệm huyết học, việc đếm và phân loại tế bào máu dưới kính hiển vi là một công việc quan trọng nhưng tốn nhiều thời gian và dễ xảy ra sai sót do yếu tố con người.

Dự án này xây dựng một hệ thống AI có khả năng:

- Phát hiện tế bào máu trong ảnh lam máu.
- Phân loại tế bào thành ba loại: Hồng cầu (RBC), Bạch cầu (WBC), Tiểu cầu (Platelets).
- Thống kê số lượng và tỷ lệ từng loại tế bào.
- Sinh báo cáo kết quả tự động ở nhiều định dạng.
- Cảnh báo lâm sàng khi phát hiện giá trị bất thường.

Hệ thống được phát triển theo hướng Hybrid AI Pipeline, kết hợp giữa Deep Learning và Machine Learning nhằm tận dụng ưu điểm của cả hai phương pháp: tốc độ phát hiện của YOLO và tính giải thích được của các mô hình ML cổ điển.

---

## Mục tiêu

- Xây dựng mô hình phát hiện tế bào máu từ ảnh hiển vi.
- Trích xuất các đặc trưng hình thái học của tế bào.
- Huấn luyện mô hình phân loại tế bào máu.
- Tự động hóa quy trình phân tích ảnh.
- Cung cấp giao diện trực quan cho người dùng.

---

## Kiến trúc hệ thống

```text
  Ảnh đầu vào (.jpg / .png)
          |
          v
  YOLOv8 Detection
  (Phát hiện & Bounding Box)
          |
          v
  Feature Extraction (OpenCV)
  (Cắt crop & trích 20 đặc trưng)
          |
          v
  ML Classification
  (KNN / Decision Tree / SVM)
          |
          v
  Statistics & Reporting
  (Đếm, tỷ lệ, cảnh báo lâm sàng)
          |
          v
  Kết quả đầu ra
  (Ảnh có nhãn + Báo cáo CSV/JSON/TXT/Excel)
```

---

## Phương pháp thực hiện

### 1. Phát hiện tế bào bằng YOLOv8

Mô hình YOLOv8 được sử dụng để xác định vị trí các tế bào máu trong ảnh. Đầu ra gồm tọa độ Bounding Box và Confidence Score cho từng tế bào được phát hiện.

Cấu hình mặc định:

| Tham số         | Giá trị | Mô tả                               |
|-----------------|---------|-------------------------------------|
| conf_threshold  | 0.08    | Ngưỡng độ tin cậy                   |
| iou_threshold   | 0.40    | Ngưỡng NMS                          |
| max_det         | 800     | Số tế bào tối đa phát hiện mỗi ảnh  |
| imgsz           | 416     | Kích thước ảnh đầu vào              |

### 2. Trích xuất đặc trưng

Sau khi phát hiện tế bào, từng vùng ảnh được cắt ra và xử lý bằng OpenCV để trích xuất **20 đặc trưng** hình thái học.

**Nhóm đặc trưng hình dạng (9 đặc trưng)**

| Đặc trưng     | Mô tả                                          |
|---------------|------------------------------------------------|
| area          | Diện tích vùng tế bào (pixel²)                 |
| perimeter     | Chu vi đường viền (pixel)                      |
| circularity   | Độ tròn (0–1, giá trị 1 là tròn hoàn hảo)     |
| eccentricity  | Độ lệch tâm, đo mức độ kéo dài của hình dạng  |
| solidity      | Độ đặc = diện tích / diện tích bao lồi        |
| extent        | Tỷ lệ diện tích / hình chữ nhật bao quanh     |
| hu_moment_1   | Mô-men Hu bất biến thứ nhất                   |
| hu_moment_2   | Mô-men Hu bất biến thứ hai                    |
| hu_moment_3   | Mô-men Hu bất biến thứ ba                     |

**Nhóm đặc trưng màu sắc (7 đặc trưng)**

| Đặc trưng   | Mô tả                                |
|-------------|--------------------------------------|
| mean_b/g/r  | Giá trị màu trung bình theo kênh BGR |
| mean_h/s/v  | Giá trị màu trung bình theo kênh HSV |

**Nhóm đặc trưng thống kê màu (4 đặc trưng)**

| Đặc trưng        | Mô tả                                 |
|------------------|---------------------------------------|
| color_std_b/g/r  | Độ lệch chuẩn màu sắc theo kênh BGR  |
| color_std_hsv    | Độ lệch chuẩn kênh Hue trong HSV     |

**Nhóm đặc trưng kết cấu (1 đặc trưng)**

| Đặc trưng             | Mô tả                          |
|-----------------------|--------------------------------|
| texture_laplacian_var | Phương sai Laplacian (độ sắc)  |

### 3. Phân loại bằng Machine Learning

Các thuật toán được huấn luyện và so sánh:

- Support Vector Machine (SVM) với kernel RBF — thường cho kết quả tốt nhất
- K-Nearest Neighbors (KNN) với distance weighting
- Decision Tree với max_depth=10

Mô hình tốt nhất được chọn tự động theo Accuracy trên tập kiểm thử và lưu vào `models/ml/best_ml_model.pt`.

---

## Kết quả thực nghiệm

### Bộ dữ liệu

- 765 ảnh lam máu đã gán nhãn.
- 3 lớp tế bào: RBC, WBC, Platelets.
- Tập train / valid / test được chia theo cấu trúc thư mục `data/`.

### Hiệu năng mô hình ML

Kết quả được đo trên tập kiểm thử (20% dữ liệu, stratified split):

| Chỉ số              | Giá trị                              |
|---------------------|--------------------------------------|
| Accuracy tổng thể   | Xem output của `scripts/train_ml.py` |
| F1-Score (weighted) | Xem output của `scripts/train_ml.py` |
| Mô hình được chọn   | SVM (RBF)                            |

> Lưu ý: Accuracy chính xác phụ thuộc vào chất lượng YOLO model và bộ dữ liệu. Chạy `python scripts/train_ml.py` để xem kết quả thực tế kèm confusion matrix.

### Tốc độ xử lý

| Thành phần         | Thời gian           |
|--------------------|---------------------|
| YOLO Detection     | 50 – 100 ms / ảnh   |
| Feature Extraction | 10 – 20 ms / tế bào |
| ML Classification  | 1 – 2 ms / tế bào   |

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- RAM tối thiểu 8 GB
- GPU tùy chọn (chỉ cần khi huấn luyện lại YOLO trên Kaggle)

### Cài đặt môi trường

```bash
# 1. Clone repository
git clone <repository-url>
cd BTL_AIT2004_2_Panacea

# 2. Tạo môi trường ảo
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

### Thiết lập PYTHONPATH (bắt buộc mỗi lần mở terminal mới)

```powershell
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path

# macOS / Linux
export PYTHONPATH=$(pwd)
```

### Chuẩn bị dữ liệu

Đảm bảo file `data/data.yaml` tồn tại với nội dung:

```yaml
path: ./data
train: train/images
val:   valid/images
test:  test/images

nc: 3
names: ['Platelets', 'RBC', 'WBC']
```

### Chuẩn bị model

Đặt các file model vào đúng vị trí:

```text
models/yolo/best.pt          <- YOLO weights (train trên Kaggle GPU)
models/ml/best_ml_model.pt   <- ML model (sinh ra khi chạy train_ml.py)
```

---

## Hướng dẫn sử dụng

### Bước 1: Huấn luyện ML model (chỉ cần làm 1 lần)

```bash
# Gộp 2 bước: trích đặc trưng + huấn luyện
python scripts/run_ml_pipeline.py

# Hoặc từng bước thủ công:
python scripts/build_train_features_from_labels.py
python scripts/train_ml.py
```

### Bước 2: Chạy phân tích

**Giao diện Web**

```bash
python web_app.py
# Truy cập http://localhost:5000
```

Chức năng: tải ảnh từ trình duyệt, hiển thị kết quả trực quan, tải báo cáo.

**Command Line — một ảnh**

```bash
python app.py --image data/test/images/sample.jpg --output outputs/ --verbose
```

**Command Line — cả thư mục**

```bash
python app.py --folder data/test/images --output outputs/
```

**Python API**

```python
from src.pipeline.end_to_end import HybridCellPipeline
from pathlib import Path

pipeline = HybridCellPipeline(
    yolo_model_path=Path("models/yolo/best.pt"),
    ml_model_path=Path("models/ml/best_ml_model.pt")
)

result = pipeline.run_on_image_full(
    image_path=Path("sample.jpg"),
    output_image_path=Path("output.png"),
    output_stats_dir=Path("outputs/reports/")
)

print(result.report_text)
```

---

## Kết quả đầu ra

Sau khi phân tích, hệ thống sinh ra các file sau:

| File              | Mô tả                                          |
|-------------------|------------------------------------------------|
| *_annotated.png   | Ảnh gốc với bbox màu và nhãn từng tế bào      |
| *_report.txt      | Báo cáo văn bản dễ đọc                        |
| *_report.json     | Báo cáo cấu trúc JSON                         |
| *_summary.csv     | Bảng thống kê tóm tắt                         |
| *_features.csv    | Đặc trưng chi tiết từng tế bào                |
| *_report.xlsx     | Báo cáo Excel nhiều sheet                     |

Màu bbox trên ảnh kết quả:

| Loại tế bào | Màu        |
|-------------|------------|
| RBC         | Đỏ         |
| WBC         | Xanh dương |
| Platelets   | Cam        |

---

## Cấu trúc dự án

```text
BTL_AIT2004_2_Panacea/
|
|-- app.py                          # Giao diện CLI chính
|-- web_app.py                      # Giao diện Web (Flask)
|
|-- src/
|   |-- detection/
|   |   `-- yolo_detector.py        # Wrapper YOLOv8: detect, crop, draw
|   |-- features/
|   |   `-- extractor.py            # Trích xuất 20 đặc trưng bằng OpenCV
|   |-- classification/
|   |   `-- ml_classifier.py        # Train / load / predict KNN, DT, SVM
|   |-- pipeline/
|   |   `-- end_to_end.py           # Nối 3 module thành pipeline hoàn chỉnh
|   |-- config/
|   |   `-- settings.py             # Quản lý đường dẫn toàn dự án
|   `-- utils/
|       |-- statistics.py           # Tính thống kê, sinh báo cáo, cảnh báo
|       `-- io.py                   # Tiện ích tạo thư mục
|
|-- scripts/
|   |-- run_ml_pipeline.py          # Gộp build features + train ML
|   |-- build_train_features_from_labels.py  # Trích đặc trưng từ GT labels
|   |-- train_ml.py                 # Huấn luyện và lưu ML model
|   |-- train_yolo.py               # Huấn luyện YOLO (thay thế notebook 02)
|   |-- infer_image.py              # Inference một ảnh
|   |-- infer_folder.py             # Inference cả thư mục (batch)
|   |-- generate_crops_from_detections.py
|   `-- extract_features_from_crops.py
|
|-- notebooks/
|   |-- 02_Train_YOLO_Detection.ipynb     # Train YOLO trên Kaggle GPU
|   |-- 03_Feature_Extraction.ipynb       # Demo trích xuất đặc trưng
|   |-- 04_Train_ML_Classification.ipynb  # Train ML, so sánh, confusion matrix
|   `-- 05_Integration_Pipeline.ipynb     # Demo pipeline end-to-end
|
|-- models/
|   |-- yolo/best.pt                # YOLO weights (train từ Kaggle)
|   `-- ml/best_ml_model.pt         # ML model tốt nhất
|
|-- data/
|   |-- train/images + labels/
|   |-- valid/images + labels/
|   |-- test/images + labels/
|   |-- interim/crops/              # Crops tạm thời
|   |-- processed/features/         # train_features.csv
|   `-- data.yaml                   # Cấu hình dataset cho YOLO
|
|-- outputs/                        # Kết quả inference (tự sinh)
|-- templates/index.html            # Giao diện Web HTML
|-- requirements.txt
`-- README.md
```

---

## Công nghệ sử dụng

| Thư viện / Công cụ    | Mục đích                              |
|-----------------------|---------------------------------------|
| Python 3.8+           | Ngôn ngữ lập trình chính              |
| YOLOv8 (Ultralytics)  | Phát hiện tế bào                      |
| OpenCV                | Xử lý ảnh, trích xuất đặc trưng      |
| Scikit-Learn          | Huấn luyện và đánh giá mô hình ML    |
| PyTorch               | Backend cho YOLO và lưu model         |
| NumPy / Pandas        | Xử lý dữ liệu số và bảng             |
| Flask + Flask-CORS    | Giao diện Web                         |
| openpyxl              | Xuất báo cáo Excel                    |
| Docker                | Triển khai container                  |

---

## Trạng thái dự án

- Hoàn thành Pipeline End-to-End (YOLO + OpenCV + ML)
- Hoàn thành giao diện Web (Flask)
- Hỗ trợ CLI và Python API
- Hỗ trợ xuất báo cáo đa định dạng (TXT, JSON, CSV, Excel)
- Đã xử lý thành công 765 ảnh dữ liệu

---

## Lưu ý kỹ thuật

- ML model được huấn luyện từ crop lấy theo **nhãn ground-truth**. Khi inference, crop được lấy từ **YOLO detection** — nếu YOLO phát hiện sai vị trí, độ chính xác phân loại có thể thấp hơn kết quả trên tập train.
- YOLO được cấu hình `nc=3` để phát hiện 3 lớp tế bào; ML phân loại lại độc lập từ đặc trưng hình thái — đây là điểm mấu chốt của kiến trúc Hybrid.
- Khi huấn luyện lại YOLO, sử dụng `notebooks/02_Train_YOLO_Detection.ipynb` trên Kaggle GPU, sau đó copy `best.pt` về `models/yolo/best.pt`.
- Không commit vào git: `outputs/`, `runs/`, `.venv/`, `*.cache`, file model lớn.

---

## Hướng phát triển

- Mở rộng tập dữ liệu huấn luyện để tăng độ chính xác.
- Triển khai trên nền tảng Cloud (AWS / Azure).
- Xây dựng ứng dụng di động.
- Tích hợp Explainable AI (SHAP, LIME) để giải thích quyết định phân loại.
- Tích hợp với hệ thống quản lý bệnh viện (HIS).

---

## Thành viên nhóm

| STT | Họ tên         | MSSV     | 
|-----|----------------|----------|
| 1   | Hoa Văn Long   | 24022390 | 
| 2   | Lê Việt Phú    | 24022426 | 
| 3   | Nguyễn Hải Nam | 24022414 | 