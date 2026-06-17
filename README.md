#  Hệ thống phát hiện và phân loại tế bào máu

> Hệ thống phát hiện và phân loại tế bào máu tự động sử dụng mô hình YOLOv8 kết hợp với các thuật toán Machine Learning nhằm hỗ trợ phân tích ảnh lam máu.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

---

#  Giới thiệu

Trong xét nghiệm huyết học, việc đếm và phân loại tế bào máu dưới kính hiển vi là một công việc quan trọng nhưng tốn nhiều thời gian và dễ xảy ra sai sót do yếu tố con người.

Dự án này xây dựng một hệ thống AI có khả năng:

* Phát hiện tế bào máu trong ảnh lam máu.
* Phân loại tế bào thành:

  -  Hồng cầu (RBC)
  -  Bạch cầu (WBC)
  -  Tiểu cầu (Platelets)
* Thống kê số lượng và tỷ lệ từng loại tế bào.
* Sinh báo cáo kết quả tự động.

Hệ thống được phát triển theo hướng Hybrid AI Pipeline, kết hợp giữa Deep Learning và Machine Learning nhằm tận dụng ưu điểm của cả hai phương pháp.

---

#  Mục tiêu

* Xây dựng mô hình phát hiện tế bào máu từ ảnh hiển vi.
* Trích xuất các đặc trưng hình thái học của tế bào.
* Huấn luyện mô hình phân loại tế bào máu.
* Tự động hóa quy trình phân tích ảnh.
* Cung cấp giao diện trực quan cho người dùng.

---

#  Kiến trúc hệ thống

```text
Ảnh đầu vào
      │
      ▼
YOLOv8 Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Feature Extraction (OpenCV)
      │
      ▼
Machine Learning Classification
      │
      ▼
Statistics & Reporting
      │
      ▼
Kết quả đầu ra
```

---

#  Phương pháp thực hiện

## 1. Phát hiện tế bào bằng YOLOv8

Mô hình YOLOv8 được sử dụng để xác định vị trí các tế bào máu trong ảnh.

Đầu ra của mô hình gồm:

* Bounding Box
* Confidence Score

Ưu điểm:

* Tốc độ xử lý nhanh.
* Độ chính xác cao.
* Khả năng phát hiện nhiều tế bào trong cùng một ảnh.

---

## 2. Trích xuất đặc trưng

Sau khi phát hiện tế bào, từng vùng ảnh được cắt ra và xử lý bằng OpenCV để trích xuất các đặc trưng hình thái học.

Các nhóm đặc trưng bao gồm:

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

Tổng cộng hệ thống sử dụng 23 đặc trưng cho mỗi tế bào.

---

## 3. Phân loại bằng Machine Learning

Các thuật toán được sử dụng:

* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree

Trong đó SVM cho kết quả tốt nhất trên tập kiểm thử và được sử dụng trong hệ thống cuối cùng.

---

#  Kết quả thực nghiệm

## Bộ dữ liệu

* 765 ảnh lam máu đã gán nhãn.
* 3 lớp tế bào:

  * RBC
  * WBC
  * Platelets

## Hiệu năng hệ thống

| Chỉ số                | Giá trị |
| --------------------- | ------- |
| Độ chính xác tổng thể | ~94%    |
| Accuracy RBC          | ~95%    |
| Accuracy WBC          | ~94%    |
| Accuracy Platelets    | ~93%    |

## Tốc độ xử lý

| Thành phần         | Thời gian           |
| ------------------ | ------------------- |
| YOLO Detection     | 50 – 100 ms         |
| Feature Extraction | 10 – 20 ms / tế bào |
| Classification     | 1 – 2 ms / tế bào   |

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

cd BTL_AIT2004_2_Panacea

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶ Hướng dẫn sử dụng

## Giao diện Web

Khởi chạy:

```bash
python web_app.py
```

Truy cập:

```text
http://localhost:5000
```

Chức năng:

* Tải ảnh trực tiếp từ trình duyệt.
* Hiển thị kết quả phân tích.
* Tải báo cáo kết quả.

---

## Command Line

Phân tích một ảnh:

```bash
python app.py --image sample.jpg
```

Phân tích thư mục:

```bash
python app.py --folder ./images
```

---

## Python API

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

Hệ thống sinh ra:

* Ảnh đã gắn nhãn (PNG)
* Báo cáo TXT
* Báo cáo JSON
* Bảng thống kê CSV
* File đặc trưng CSV
* Báo cáo Excel

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

#  Thành viên nhóm

| Thành viên   | Vai trò                    |
| ------------ | -------------------------- |
| Thành viên 1 | Object Detection           |
| Thành viên 2 | Feature Extraction         |
| Thành viên 3 | Classification & Reporting |

---

#  Hướng phát triển

* Mở rộng tập dữ liệu huấn luyện.
* Triển khai trên nền tảng Cloud.
* Xây dựng ứng dụng di động.
* Tích hợp Explainable AI (SHAP, LIME).
* Tích hợp với hệ thống quản lý bệnh viện.

---

#  Trạng thái dự án

- Hoàn thành Pipeline End-to-End

- Hoàn thành Web Interface

- Hỗ trợ CLI và Python API

- Xử lý thành công 765 ảnh dữ liệu

- Sẵn sàng cho mục đích nghiên cứu và học tập
