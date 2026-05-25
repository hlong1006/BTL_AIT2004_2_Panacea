# Hybrid Cell Detection & Classification

> Pipeline **AI lai (Hybrid)** phát hiện, đếm và phân loại tế bào trên ảnh vi thể lam máu — kết hợp **YOLO (Deep Learning)** với **đặc trưng OpenCV + ML cổ điển (KNN / Decision Tree / SVM)**.

---

## Mục lục

- [Đặt vấn đề](#đặt-vấn-đề)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Quy trình sử dụng](#quy-trình-sử-dụng)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
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

> **Lưu ý:** ML trong project phân loại **loại tế bào** (RBC / WBC / Platelets), **không** phải chẩn đoán ung thư máu (normal/cancer). Nếu cần bài toán ung thư, phải bổ sung dataset và nhãn lớp mới.

---

## Kiến trúc hệ thống

```
Ảnh vi thể gốc
      │
      ▼
┌─────────────────────────────┐
│  MODULE 1: Object Detection │  ← YOLOv8 (models/yolo/best.pt)
│  Phát hiện & bounding box   │
└─────────────┬───────────────┘
              │  crop từng tế bào
              ▼
┌─────────────────────────────┐
│  MODULE 2: Feature Extract  │  ← OpenCV
│  Hình thái · Màu sắc · Texture│
└─────────────┬───────────────┘
              │  vector 10 đặc trưng
              ▼
┌─────────────────────────────┐
│  MODULE 3: ML Classifier    │  ← KNN / DT / SVM (best_ml_model.pt)
│  Platelets / RBC / WBC      │
└─────────────┬───────────────┘
              │
              ▼
   Ảnh có nhãn màu + CSV báo cáo
```

**10 đặc trưng trích từ mỗi crop:**

`area`, `perimeter`, `circularity`, `mean_b/g/r`, `mean_h/s/v`, `texture_laplacian_var`

**Màu khung trên ảnh kết quả:**

| Loại | Màu |
|------|-----|
| RBC | Đỏ |
| WBC | Xanh dương |
| Platelets | Cam |

---

## Cấu trúc thư mục

```
.
├── data/
│   ├── train/                 # Ảnh + nhãn YOLO (train)
│   ├── valid/                 # Ảnh + nhãn YOLO (valid)
│   ├── test/                  # Ảnh + nhãn YOLO (test)
│   ├── data.yaml              # Cấu hình lớp: Platelets, RBC, WBC
│   └── processed/features/
│       └── train_features.csv # CSV đặc trưng (sau build, có thể gitignore)
│
├── models/
│   ├── yolo/best.pt           # Trọng số YOLO (đã train)
│   └── ml/best_ml_model.pt    # Model ML (PyTorch checkpoint)
│
├── src/
│   ├── detection/yolo_detector.py
│   ├── features/extractor.py
│   ├── classification/ml_classifier.py
│   └── pipeline/end_to_end.py
│
├── scripts/
│   ├── build_train_features_from_labels.py  # Tạo train_features.csv từ nhãn GT
│   ├── run_ml_pipeline.py                   # Build features + train ML
│   ├── train_ml.py
│   ├── infer_image.py                       # Suy luận 1 ảnh
│   ├── infer_folder.py                      # Suy luận cả folder
│   ├── extract_features_from_crops.py       # (tùy chọn) từ thư mục crop có sẵn
│   └── generate_crops_from_detections.py
│
├── outputs/                   # Kết quả inference (tự sinh)
├── notebooks/                 # Notebook 02–05 (YOLO, feature, ML, tích hợp)
├── runs/                      # Log train YOLO (không cần commit)
└── requirements.txt
```

---

## Cài đặt

**Yêu cầu:** Python 3.10+, RAM ≥ 8 GB. GPU chỉ cần khi train lại YOLO (vd. Kaggle).

```powershell
cd D:\BTL_AIT2004_2_Panacea

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Bắt buộc mỗi lần chạy script
$env:PYTHONPATH = (Get-Location).Path
```

---

## Quy trình sử dụng

### Bước 0 — Model YOLO (đã có sẵn)

Nếu chưa có `models/yolo/best.pt`, copy từ thư mục train:

```powershell
New-Item -ItemType Directory -Force -Path models\yolo | Out-Null
Copy-Item runs\cell_detect\weights\best.pt models\yolo\best.pt
```

> Thiếu `best.pt` → pipeline fallback: coi **cả ảnh** là 1 object (chỉ để test).

---

### Bước 1 — Train ML (chỉ cần làm 1 lần, hoặc khi đổi data)

**Cách 1 — Gộp 2 bước:**

```powershell
python scripts/run_ml_pipeline.py
```

**Cách 2 — Từng bước:**

```powershell
# 1) Crop từ nhãn GT (train + valid) -> train_features.csv
python scripts/build_train_features_from_labels.py

# 2) Train KNN / DT / SVM, lưu best_ml_model.pt
python scripts/train_ml.py
```

Script train in classification report và chọn model tốt nhất (thường **SVM**).

---

### Bước 2 — Suy luận (đã có model thì chỉ cần bước này)

**Cả folder test (mặc định `data/test/images`):**

```powershell
python scripts/infer_folder.py
```

**Một ảnh:**

```powershell
python scripts/infer_image.py `
  --image data/test/images/BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg `
  --output outputs/prediction.png `
  --feature-csv outputs/predicted_features.csv
```

**Folder khác (vd. valid):**

```powershell
python scripts/infer_folder.py --images-dir data/valid/images
```

---

## Kết quả đầu ra

### Suy luận 1 ảnh (`infer_image.py`)

| File | Mô tả |
|------|-------|
| `outputs/prediction.png` | Ảnh có bbox màu + nhãn từng tế bào |
| `outputs/predicted_features.csv` | Đặc trưng + `predicted_label` mỗi cell |

### Suy luận folder (`infer_folder.py`)

| File / thư mục | Mô tả |
|----------------|-------|
| `outputs/predictions/*.png` | Ảnh kết quả từng file |
| `outputs/features/*.csv` | Chi tiết từng tế bào mỗi ảnh |
| `outputs/summary.csv` | Tổng hợp đếm: `total_cells`, `platelets`, `rbc`, `wbc` / ảnh |

**Ví dụ `summary.csv`:**

```
image,total_cells,platelets,rbc,wbc
BloodImage_00038_....jpg,12,1,10,1
```

---

## Lưu ý kỹ thuật

- **Train ML** dùng crop từ **nhãn ground-truth**; **inference** dùng crop từ **bbox YOLO** — nếu YOLO miss/sai box, độ chính xác pipeline có thể thấp hơn số trên CSV train.
- YOLO đã train `nc=3` (vừa detect vừa dự đoán lớp); ML phân loại lại độc lập từ đặc trưng → phù hợp mô tả hybrid (DL định vị + ML giải thích từ feature).
- Train lại YOLO: dùng notebook `notebooks/02_Train_YOLO_Detection.ipynb` trên Kaggle GPU, sau đó copy `best.pt` về `models/yolo/`.
- Không commit: `outputs/`, `runs/`, `yolov8n.pt`, file cache `*.cache`.

---

## Nhóm phát triển

| Thành viên | Vai trò |
|------------|---------|
| [Tên thành viên 1] | Object Detection — YOLO |
| [Tên thành viên 2] | Feature Extraction — OpenCV |
| [Tên thành viên 3] | ML Classifier — Train, infer, pipeline |

---
