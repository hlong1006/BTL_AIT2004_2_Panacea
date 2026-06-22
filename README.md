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

### 1. Phát hiện tế bào với YOLOv8

Mô hình YOLOv8 được dùng để phát hiện từng tế bào máu trong ảnh lam. Quá trình xử lý gồm:

- Tiền xử lý ảnh đầu vào: resize, chuẩn hoá và chuyển đổi định dạng cho phù hợp với mô hình.
- Chạy YOLOv8 để nhận diện các vùng chứa tế bào.
- Lọc kết quả bằng confidence threshold và Non-Maximum Suppression (NMS).
- Sinh ra danh sách bounding box kèm nhãn, độ tin cậy và tọa độ.

Cấu hình mặc định của pipeline:

| Tham số         | Giá trị | Mô tả                                       |
|-----------------|---------|---------------------------------------------|
| conf_threshold  | 0.06    | Ngưỡng chấp nhận phát hiện                   |
| iou_threshold   | 0.45    | Ngưỡng NMS để loại bỏ các hộp chồng lấp     |
| max_det         | 1000     | Số lượng tế bào tối đa xử lý mỗi ảnh        |
| imgsz           | 640    | Kích thước ảnh đầu vào sau khi resize       |

Sau bước này, mỗi tế bào được biểu diễn bằng một vùng ảnh (ROI) sẵn sàng cho bước trích xuất đặc trưng.

### 2. Trích xuất đặc trưng tế bào

Mỗi ROI được xử lý bằng OpenCV và các phép toán hình ảnh để trích lựa các đặc trưng mô tả hình dạng, màu sắc và kết cấu.

Quá trình trích xuất bao gồm:

- Chuyển ROI về không gian màu BGR và HSV.
- Làm mịn ảnh và áp dụng threshold để tách nền khỏi đối tượng.
- Xác định contour và tính đa giác bao quanh tế bào.
- Tạo mask tế bào để tính đặc trưng chỉ trên vùng đối tượng.
- Tính toán các chỉ số hình học và thống kê màu sắc.

Các đặc trưng được sử dụng bao gồm:

**Nhóm đặc trưng hình dạng (9 đặc trưng)**

| Đặc trưng     | Mô tả                                          |
|---------------|------------------------------------------------|
| area          | Diện tích vùng tế bào (pixel²)                 |
| perimeter     | Chu vi đường viền tế bào (pixel)               |
| circularity   | Độ tròn = 4π × area / perimeter²                |
| eccentricity  | Mức độ kéo dài của hình dạng tế bào             |
| solidity      | Diện tích / diện tích bao lồi                   |
| extent        | Tỷ lệ diện tích so với hình chữ nhật bao quanh  |
| hu_moment_1   | Mô-men Hu bất biến thứ nhất                     |
| hu_moment_2   | Mô-men Hu bất biến thứ hai                      |
| hu_moment_3   | Mô-men Hu bất biến thứ ba                       |

**Nhóm đặc trưng màu sắc (6 đặc trưng)**

| Đặc trưng   | Mô tả                                      |
|-------------|--------------------------------------------|
| mean_b      | Trung bình kênh B trên vùng tế bào          |
| mean_g      | Trung bình kênh G trên vùng tế bào          |
| mean_r      | Trung bình kênh R trên vùng tế bào          |
| mean_h      | Trung bình kênh H trên không gian HSV       |
| mean_s      | Trung bình kênh S trên không gian HSV       |
| mean_v      | Trung bình kênh V trên không gian HSV       |

**Nhóm đặc trưng thống kê màu (4 đặc trưng)**

| Đặc trưng        | Mô tả                                           |
|------------------|-------------------------------------------------|
| color_std_b      | Độ lệch chuẩn kênh B                            |
| color_std_g      | Độ lệch chuẩn kênh G                            |
| color_std_r      | Độ lệch chuẩn kênh R                            |
| color_std_hsv    | Độ lệch chuẩn kênh Hue (HSV)                          |

**Nhóm đặc trưng kết cấu (1 đặc trưng)**

| Đặc trưng             | Mô tả                                  |
|-----------------------|----------------------------------------|
| texture_laplacian_var | Phương sai Laplacian, đo độ sắc nét   |

Tất cả đặc trưng được ghép thành feature vector cho mỗi tế bào, làm đầu vào cho bước phân loại.

### 3. Huấn luyện và phân loại bằng Machine Learning

Bộ dữ liệu đặc trưng được chia thành tập huấn luyện và kiểm thử giữ nguyên tỷ lệ các lớp (stratified split).

Các mô hình ML được so sánh gồm:

- Support Vector Machine (SVM) với kernel RBF
- K-Nearest Neighbors (KNN) với trọng số khoảng cách
- Decision Tree với max_depth=10

Quy trình huấn luyện:

- Chuẩn hoá đặc trưng nếu cần.
- Lựa chọn các đặc trưng quan trọng.
- Huấn luyện mỗi mô hình trên tập train.
- Đánh giá bằng accuracy, F1-score và confusion matrix trên tập test.
- Lưu mô hình tốt nhất vào `models/ml/best_ml_model.pt`.

Các script chính thực thi quy trình này:

- `scripts/build_train_features_from_labels.py` — tạo bảng đặc trưng từ dữ liệu ảnh và nhãn.
- `scripts/train_ml.py` — huấn luyện và lưu mô hình ML.
- `scripts/run_ml_pipeline.py` — chạy toàn bộ pipeline trích đặc trưng và huấn luyện.

### 4. Pipeline end-to-end

Pipeline cuối cùng kết hợp cả hai giai đoạn:

1. Phát hiện tế bào bằng YOLOv8.
2. Trích xuất đặc trưng hình ảnh cho từng ROI.
3. Dự đoán loại tế bào bằng mô hình ML đã huấn luyện.
4. Tính toán số lượng và tỷ lệ mỗi loại tế bào.
5. Xuất báo cáo và ảnh kết quả.

Luồng này giúp tận dụng khả năng phát hiện nhanh của Deep Learning và độ chính xác giải thích được của các mô hình ML cổ điển.

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
- Docker & Docker Compose (Tùy chọn, nếu muốn chạy qua Container)

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

### Bước 3: Chạy hệ thống bằng Docker (Khuyến nghị)

**Khởi động nhanh Web App bằng Docker Compose (Bản CPU)**

```bash
docker-compose up -d
# Truy cập http://localhost:5000
# Để tắt hệ thống: docker-compose down
```

**Khởi động bản GPU (Yêu cầu có Card Nvidia & NVIDIA Container Toolkit)**

```bash
docker-compose up -d blood-cell-analyzer-gpu
```

**Chạy giao diện dòng lệnh (CLI) qua Docker**
```bash
# Phân tích một ảnh
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  blood-cell-analyzer:latest python app.py --image data/test/images/sample.jpg --output outputs/analysis_results/

# Phân tích hàng loạt thư mục
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  blood-cell-analyzer:latest python app.py --folder data/test/images --output outputs/analysis_results/ --recursive
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
|-- app.py                          # Giao diện CLI chính
|-- web_app.py                      # Giao diện Web API (Flask)
|
|-- src/                            # LÕI HỆ THỐNG
|   |-- calibration/                # Ước lượng chỉ số máu (CBC Estimator)
|   |-- classification/             # Phân loại ML, gợi ý WBC Subtype
|   |-- config/                     # Cấu hình dự án, đường dẫn
|   |-- detection/                  # Phát hiện YOLOv8, NMS tùy chỉnh
|   |-- diagnosis/                  # Chẩn đoán sức khỏe y khoa, phát hiện bất thường
|   |-- evaluation/                 # Đánh giá Ground Truth, Benchmark
|   |-- explainability/             # Giải thích quyết định mô hình (Feature Explainer)
|   |-- features/                   # Rút trích 20 đặc trưng OpenCV (Extractor)
|   |-- llm/                        # Giải thích ngôn ngữ tự nhiên (Local/OpenAI)
|   |-- pipeline/                   # Khối điều phối Pipeline End-to-End
|   |-- reporting/                  # Trình xuất báo cáo PDF (ReportLab)
|   `-- utils/                      # Tính toán thống kê lâm sàng, I/O
|
|-- scripts/                        # CÁC KỊCH BẢN THỰC THI
|   |-- run_ml_pipeline.py
|   |-- build_train_features_from_labels.py
|   |-- train_ml.py
|   |-- train_yolo.py
|   |-- infer_image.py
|   |-- infer_folder.py
|   |-- generate_crops_from_detections.py
|   |-- extract_features_from_crops.py
|   `-- generate_benchmark_report.py
|
|-- notebooks/                      # Jupyter Notebooks nghiên cứu
|-- models/                         # Lưu trữ trọng số (YOLO & ML)
|-- data/                           # Dữ liệu ảnh, nhãn, features
|-- outputs/                        # Kết quả đầu ra tự sinh
|-- templates/                      # Chứa giao diện index.html
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
- Hoàn thành giao diện Web SPA với Flask & Chart.js
- Chẩn đoán sức khỏe máu dựa trên **tỷ lệ %** tế bào
- Phát hiện tế bào bất thường (Anomaly Detection)
- Giải thích AI (Permutation Importance / Feature Explanation)
- So sánh YOLO-only vs Hybrid pipeline
- Đánh giá Ground Truth (ảnh test set)
- Ước lượng CBC, WBC subtype heuristic
- Batch upload (phân tích hàng loạt) và xem Lịch sử kết quả
- Xuất báo cáo PDF
- Giải thích dễ hiểu (local + tùy chọn OpenAI API)
- Docker, CLI, Python API, Unit tests

---

## Tính năng nổi bật

| Tính năng | Mô tả |
|-----------|-------|
| **Chẩn đoán sức khỏe** | Tự động so sánh tỷ lệ RBC/WBC/Platelets với ngưỡng tham chiếu sinh lý |
| **Giao diện Web Dashboard** | Kéo thả upload, biểu đồ trực quan (Chart.js), điều chỉnh ngưỡng confidence |
| **Phân tích hàng loạt** | Upload nhiều ảnh cùng lúc, sinh báo cáo tổng hợp (Batch processing) |
| **Xuất báo cáo PDF** | Báo cáo chuyên nghiệp đầy đủ thông tin, cảnh báo, hình ảnh |
| **Giải thích AI (XAI)** | Đánh giá tầm quan trọng của từng đặc trưng bằng Permutation Importance |
| **Phát hiện bất thường** | Phát hiện tế bào có hình thái lệch chuẩn (Z-score > 2.5) |
| **Kiểm chứng Ground Truth** | So sánh trực tiếp với nhãn thủ công để ra điểm Precision/Recall/F1 |
| **Ước lượng CBC** | Ước lượng thô nồng độ tế bào máu (cells/µL) dựa trên thông số vi trường |
| **Gợi ý WBC Subtype** | Sử dụng Heuristic để dự đoán loại bạch cầu con |
| **Giải thích bằng LLM** | Dịch kết quả chuyên ngành sang ngôn ngữ bình dân (tích hợp OpenAI API) |
| **So sánh Hybrid** | Báo cáo tỷ lệ đồng thuận giữa YOLO-only và kiến trúc Hybrid |


Chạy benchmark cho slide bảo vệ:

```bash
python scripts/generate_benchmark_report.py
```

Chạy unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Lưu ý kỹ thuật

- ML model được huấn luyện từ crop lấy theo **nhãn ground-truth**. Khi inference, crop được lấy từ **YOLO detection** — nếu YOLO phát hiện sai vị trí, độ chính xác phân loại có thể thấp hơn kết quả trên tập train.
- YOLO được cấu hình `nc=3` để phát hiện 3 lớp tế bào; ML phân loại lại độc lập từ đặc trưng hình thái — đây là điểm mấu chốt của kiến trúc Hybrid.
- Khi huấn luyện lại YOLO, sử dụng `notebooks/02_Train_YOLO_Detection.ipynb` trên Kaggle GPU, sau đó copy `best.pt` về `models/yolo/best.pt`.
- Không commit vào git: `outputs/`, `runs/`, `.venv/`, `*.cache`, file model lớn.

---

## Hướng phát triển

- Huấn luyện model WBC differential (Neutrophil, Lymphocyte...) với dataset chuyên biệt
- Hiệu chuẩn CBC chính xác với máy đếm tự động
- Triển khai Cloud (AWS / Azure)
- Ứng dụng di động (React Native / Flutter)
- Tích hợp SHAP đầy đủ khi có GPU
- Tích hợp HIS bệnh viện

---

## Thành viên nhóm

| STT | Họ tên         | MSSV     | 
|-----|----------------|----------|
| 1   | Hoa Văn Long   | 24022390 | 
| 2   | Lê Việt Phú    | 24022426 | 
| 3   | Nguyễn Hải Nam | 24022414 | 