# Hệ thống phát hiện và phân loại tế bào máu

> Đồ án xây dựng hệ thống AI lai (Hybrid AI Pipeline) kết hợp **YOLOv8**, **OpenCV** và **Machine Learning** nhằm tự động phát hiện, đếm và phân loại tế bào máu từ ảnh lam máu.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

---

#  Giới thiệu

Phân tích tế bào máu dưới kính hiển vi là một công việc quan trọng trong xét nghiệm huyết học. Tuy nhiên, quá trình đếm và phân loại thủ công thường:

* Tốn nhiều thời gian.
* Dễ xảy ra sai sót do yếu tố con người.
* Yêu cầu kỹ thuật viên có chuyên môn.

Để giải quyết vấn đề này, nhóm xây dựng hệ thống AI có khả năng:

* Tự động phát hiện tế bào máu trong ảnh.
* Phân loại tế bào thành:

  * 🔴 Hồng cầu (RBC)
  * 🔵 Bạch cầu (WBC)
  * 🟠 Tiểu cầu (Platelets)
* Sinh báo cáo thống kê tự động.
* Hỗ trợ nghiên cứu và ứng dụng AI trong lĩnh vực y sinh.

---

#  Mục tiêu

* Xây dựng mô hình phát hiện tế bào máu bằng YOLOv8.
* Trích xuất đặc trưng hình thái học bằng OpenCV.
* Huấn luyện mô hình Machine Learning để phân loại tế bào.
* Xây dựng pipeline hoàn chỉnh từ ảnh đầu vào đến báo cáo đầu ra.
* Cung cấp giao diện Web và Command Line để sử dụng dễ dàng.

---

#  Kiến trúc hệ thống

```text
Ảnh đầu vào
      │
      ▼
┌───────────────────────┐
│   YOLOv8 Detection    │
│   Phát hiện tế bào    │
└───────────┬───────────┘
            ▼
      Bounding Boxes
            │
            ▼
┌───────────────────────┐
│ Feature Extraction    │
│      OpenCV           │
└───────────┬───────────┘
            ▼
      Vector đặc trưng
            │
            ▼
┌───────────────────────┐
│ ML Classification     │
│ SVM / KNN / DT        │
└───────────┬───────────┘
            ▼
      Nhãn tế bào
            │
            ▼
┌───────────────────────┐
│ Statistics & Reports  │
└───────────────────────┘
```

---

#  Bộ dữ liệu

Bộ dữ liệu gồm ảnh lam máu đã được gán nhãn với 3 lớp:

* RBC (Red Blood Cell)
* WBC (White Blood Cell)
* Platelets

Dữ liệu được chia thành:

| Tập dữ liệu | Mục đích           |
| ----------- | ------------------ |
| Train       | Huấn luyện mô hình |
| Validation  | Tinh chỉnh tham số |
| Test        | Đánh giá cuối cùng |

Tổng số ảnh đã xử lý:

```text
765 ảnh
```

---

#  Cải tiến phiên bản 2.0

## 1. Cải thiện phát hiện tế bào

| Tham số              | Phiên bản cũ | Phiên bản mới |
| -------------------- | ------------ | ------------- |
| Confidence Threshold | 0.12         | 0.08          |
| IoU Threshold        | 0.45         | 0.40          |
| Max Detection        | 500          | 800           |

### Kết quả

 - Tỷ lệ phát hiện tế bào nhỏ tăng từ khoảng 70% lên 90%

 - Giảm số lượng phát hiện trùng lặp

 - Hoạt động tốt hơn với ảnh có mật độ tế bào cao

---

## 2. Mở rộng bộ đặc trưng

Số lượng đặc trưng tăng từ:

```text
10 → 23 đặc trưng
```

### Đặc trưng hình dạng

* Area
* Perimeter
* Circularity
* Eccentricity
* Solidity
* Extent
* Hu Moments

### Đặc trưng màu sắc

* Mean BGR
* Mean HSV
* Standard Deviation BGR
* Standard Deviation HSV

### Đặc trưng kết cấu

* Laplacian Variance

---

## 3. Cải thiện mô hình phân loại

### SVM

* Kernel RBF
* Cân bằng dữ liệu giữa các lớp
* Hỗ trợ Confidence Score

### KNN

* K = 5
* Distance Weighting

### Decision Tree

* Max Depth = 10
* Min Samples Split = 5

---

#  Hiệu năng hệ thống

## Độ chính xác

| Loại tế bào | Accuracy |
| ----------- | -------- |
| RBC         | 95%      |
| WBC         | 94%      |
| Platelets   | 93%      |
| Tổng thể    | 94%      |

## Tốc độ xử lý

| Thành phần           | Thời gian           |
| -------------------- | ------------------- |
| YOLO Detection       | 50 - 100 ms         |
| Trích xuất đặc trưng | 10 - 20 ms / tế bào |
| Phân loại            | 1 - 2 ms / tế bào   |

---

#  Cài đặt

## Yêu cầu

* Python 3.8+
* OpenCV
* PyTorch
* Ultralytics YOLOv8
* Scikit-Learn

## Cài đặt môi trường

```bash
git clone <repository-url>

cd Blood-Cell-Detection-System

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

#  Hướng dẫn sử dụng

## 1. Giao diện Web

Khởi chạy:

```bash
python web_app.py
```

Truy cập:

```text
http://localhost:5000
```

Tính năng:

* Tải ảnh trực tiếp từ trình duyệt
* Phân tích tự động
* Hiển thị kết quả trực quan
* Tải báo cáo

---

## 2. Command Line

### Phân tích một ảnh

```bash
python app.py --image sample.jpg
```

### Phân tích thư mục ảnh

```bash
python app.py --folder ./images
```

### Tăng độ nhạy phát hiện

```bash
python app.py --image sample.jpg --conf 0.06
```

### Giảm phát hiện nhầm

```bash
python app.py --image sample.jpg --conf 0.12
```

---

## 3. Python API

```python
from src.pipeline.end_to_end import HybridCellPipeline

pipeline = HybridCellPipeline()

result = pipeline.run_on_image_full(
    image_path="sample.jpg",
    output_image_path="output.png"
)
```

---

#  Kết quả đầu ra

Sau khi xử lý, hệ thống sinh ra:

```text
Ảnh đã gắn nhãn (.png)
Báo cáo văn bản (.txt)
Báo cáo JSON (.json)
Bảng thống kê CSV (.csv)
File đặc trưng (.csv)
Báo cáo Excel (.xlsx)
```

Ví dụ:

```text
Tổng số tế bào: 15

RBC: 11
WBC: 3
Platelets: 1
```

---

#  Cấu trúc dự án

```text
project/
│
├── app.py
├── web_app.py
│
├── src/
│   ├── detection/
│   ├── features/
│   ├── classification/
│   ├── pipeline/
│   └── utils/
│
├── models/
│   ├── yolo/
│   └── ml/
│
├── notebooks/
├── outputs/
├── templates/
│
├── requirements.txt
└── README.md
```

---


#  Hướng phát triển

* Triển khai lên Cloud (AWS/Azure)
* Ứng dụng di động
* Hệ thống đăng nhập người dùng
* Lưu trữ kết quả bằng Database
* Explainable AI (SHAP/LIME)
* Tích hợp với hệ thống quản lý bệnh viện

---

#  Thành viên nhóm

| Thành viên   | Vai trò                    |
| ------------ | -------------------------- |
| Thành viên 1 | YOLO Detection             |
| Thành viên 2 | Feature Extraction         |
| Thành viên 3 | Classification & Reporting |

---

#  Công nghệ sử dụng

* Python
* YOLOv8
* OpenCV
* Scikit-Learn
* NumPy
* Pandas
* Flask
* PyTorch
* Docker

---

#  Kết luận

Dự án đã xây dựng thành công hệ thống AI lai kết hợp Deep Learning và Machine Learning để phát hiện và phân loại tế bào máu tự động.

### Kết quả nổi bật

- Xử lý thành công 765 ảnh dữ liệu

- Phát hiện tế bào nhỏ đạt ~90%

- 23 đặc trưng hình thái học

- Độ chính xác phân loại khoảng 94%

- Hỗ trợ Web UI, CLI và Python API

- Sẵn sàng triển khai và mở rộng trong thực tế
