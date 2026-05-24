#  Hybrid Cell Detection & Classification Pipeline


> **Pipeline AI Lai (Hybrid)** tự động phát hiện, đếm và phân loại tế bào trong ảnh vi thể — kết hợp tốc độ của Deep Learning với tính giải thích được của Machine Learning cổ điển.

---

##  Mục lục

- [Đặt vấn đề](#-đặt-vấn-đề)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt môi trường](#-cài-đặt-môi-trường)
- [Quy trình sử dụng](#-quy-trình-sử-dụng)
- [Kết quả đầu ra](#-kết-quả-đầu-ra)
- [Lưu ý kỹ thuật](#-lưu-ý-kỹ-thuật)
- [Nhóm phát triển](#-nhóm-phát-triển)

---

##  Đặt vấn đề

Việc đếm và phân loại tế bào thủ công qua kính hiển vi (tế bào máu, tế bào khối u) tốn nhiều thời gian và dễ sai sót. Các hệ thống Deep Learning thuần túy đòi hỏi phần cứng mạnh mẽ, dữ liệu khổng lồ và hoạt động như "hộp đen" — thiếu tính giải thích lâm sàng.

**Dự án này đề xuất kiến trúc Hybrid** giải quyết đồng thời cả hai vấn đề:

| Vấn đề | Giải pháp |
|--------|-----------|
| Phát hiện hàng nghìn tế bào thủ công | YOLO tự động localize toàn bộ tế bào |
| Mô hình black-box, khó giải thích | Feature truyền thống → ML có thể interpret |
| Yêu cầu GPU cao để inference | Module ML nhẹ, chạy mượt trên CPU local |

---

##  Kiến trúc hệ thống

```
Ảnh vi thể gốc
      │
      ▼
┌─────────────────────────────┐
│  MODULE 1: Object Detection │  ← YOLOv8 (best.pt)
│  Phát hiện & Bounding Box   │
└─────────────┬───────────────┘
              │  crop từng tế bào
              ▼
┌─────────────────────────────┐
│  MODULE 2: Feature Extract  │  ← OpenCV
│  Hình thái · Màu sắc · LBP  │
└─────────────┬───────────────┘
              │  vector đặc trưng số học
              ▼
┌─────────────────────────────┐
│  MODULE 3: ML Classifier    │  ← KNN / Decision Tree / SVM
│  Phân loại bệnh lý          │
└─────────────┬───────────────┘
              │
              ▼
   Ảnh kết quả có nhãn + CSV báo cáo
```

---

##  Cấu trúc thư mục

```
.
├── data/
│   ├── raw/
│   │   ├── images/            # Ảnh vi thể gốc đầu vào
│   │   └── labels/            # Annotation YOLO (nếu cần huấn luyện lại)
│   ├── interim/
│   │   └── crops/             # Vùng ảnh tế bào sau khi cắt từ bounding box
│   └── processed/
│       └── features/          # File CSV đặc trưng dùng để train/predict
│
├── models/
│   ├── yolo/
│   │   └── best.pt            # Trọng số YOLO đã huấn luyện (từ Kaggle)
│   └── ml/
│       └── best_ml_model.joblib  # Model ML tốt nhất sau khi train
│
├── src/
│   ├── detection/
│   │   └── yolo_detector.py   # Wrapper gọi YOLO, trả về bounding boxes
│   ├── features/
│   │   └── extractor.py       # Trích xuất đặc trưng từ crop bằng OpenCV
│   ├── classification/
│   │   └── ml_classifier.py   # Load model ML, predict nhãn
│   └── pipeline/
│       └── end_to_end.py      # Kết nối 3 module thành pipeline hoàn chỉnh
│
├── scripts/
│   ├── extract_features_from_crops.py  # [Script] Trích đặc trưng từ thư mục crop
│   ├── train_ml.py                     # [Script] Huấn luyện & lưu model ML
│   └── infer_image.py                  # [Script] Chạy inference end-to-end
│
├── outputs/                   # Ảnh kết quả và CSV dự đoán (tự sinh ra)
├── notebooks/                 # Jupyter notebook EDA, thực nghiệm
├── requirements.txt
└── README.md
```

---

##  Cài đặt môi trường

### Yêu cầu hệ thống

- Python **3.10+**
- RAM ≥ 8 GB (khuyến nghị 16 GB cho ảnh độ phân giải cao)
- GPU tùy chọn (chỉ cần cho việc huấn luyện YOLO trên Kaggle)

### Các bước cài đặt

```bash
# 1. Clone repository
git clone <repo-url>
cd hybrid-cell-classification

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

##  Quy trình sử dụng

### Bước 1 — Chuẩn bị dữ liệu

Thêm ảnh vi thể vào thư mục đầu vào:

```
data/raw/images/     ← đặt ảnh gốc (.png, .jpg, ...) vào đây
```

**Nếu cần huấn luyện lại YOLO** (khuyến nghị chạy trên Kaggle GPU):

```
data/raw/labels/     ← đặt annotation YOLO (.txt) tương ứng vào đây
```

Sau khi huấn luyện xong, copy file trọng số về:

```
models/yolo/best.pt
```

>  Nếu chưa có `best.pt`, hệ thống sẽ **tự động fallback**: coi toàn bộ ảnh là một object duy nhất để test các bước còn lại của pipeline.

---

### Bước 2 — Trích xuất đặc trưng để huấn luyện ML

Chạy script sau **cho từng nhãn** (ví dụ: `normal`, `cancer`):

```bash
# Trích đặc trưng tế bào bình thường
python scripts/extract_features_from_crops.py \
    --crops-dir data/interim/crops/normal \
    --output-csv data/processed/features/normal_features.csv \
    --label normal

# Trích đặc trưng tế bào ung thư
python scripts/extract_features_from_crops.py \
    --crops-dir data/interim/crops/cancer \
    --output-csv data/processed/features/cancer_features.csv \
    --label cancer
```

Sau đó merge các file CSV lại:

```bash
# Gộp thành 1 file dùng để train
python scripts/merge_features.py \
    --input-dir data/processed/features/ \
    --output-csv data/processed/features/train_features.csv
```

> File `train_features.csv` cần có cột `label` để train.

---

### Bước 3 — Huấn luyện mô hình ML

```bash
python scripts/train_ml.py \
    --input-csv data/processed/features/train_features.csv \
    --output-model models/ml/best_ml_model.joblib
```

Script sẽ tự động:
- Thử nghiệm các mô hình: **KNN**, **Decision Tree**, **SVM**
- Đánh giá bằng **cross-validation**
- Lưu mô hình tốt nhất theo F1-score
- In ra **Confusion Matrix** và **Classification Report**

---

### Bước 4 — Chạy inference end-to-end

```bash
python scripts/infer_image.py \
    --image data/raw/images/sample.png \
    --output outputs/prediction.png \
    --feature-csv outputs/predicted_features.csv
```

---

##  Kết quả đầu ra

Sau khi inference, thư mục `outputs/` sẽ chứa:

| File | Mô tả |
|------|-------|
| `prediction.png` | Ảnh gốc được vẽ bounding box + nhãn dự đoán từng tế bào |
| `predicted_features.csv` | Bảng đặc trưng số học kèm nhãn dự đoán của từng tế bào |

**Ví dụ nội dung `predicted_features.csv`:**

```
cell_id | area | perimeter | mean_r | mean_g | mean_b | lbp_hist | predicted_label
--------|------|-----------|--------|--------|--------|----------|----------------
cell_01 | 1523 | 142.3     | 210    | 180    | 175    | [...]    | normal
cell_02 | 2841 | 198.7     | 145    | 110    | 160    | [...]    | cancer
```

---

##  Lưu ý kỹ thuật

- **Huấn luyện YOLO** nên được thực hiện trên **Kaggle GPU** để tiết kiệm thời gian; sau đó copy `best.pt` về local.
- **Trích đặc trưng và huấn luyện ML** có thể chạy hoàn toàn trên máy local (CPU).
- Để đạt hiệu năng tốt trong ứng dụng y tế thực tế, nên tinh chỉnh (fine-tune) bộ đặc trưng OpenCV và hyperparameter của mô hình ML theo từng bộ dữ liệu cụ thể.
- Module phân loại ML hoạt động dựa trên **đặc trưng có ý nghĩa y khoa** (diện tích nhân, màu sắc, kết cấu màng), giúp tăng tính **giải thích được (Explainable AI)** — phù hợp với yêu cầu minh bạch trong chẩn đoán lâm sàng.

---

##  Nhóm phát triển

| Thành viên | Vai trò |
|------------|---------|
| [Tên thành viên 1] | Object Detection — Huấn luyện & tích hợp YOLO |
| [Tên thành viên 2] | Feature Extraction — Pipeline OpenCV |
| [Tên thành viên 3] | ML Classifier — Train, Evaluate, End-to-End |

---

