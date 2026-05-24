# Hybrid Cell Detection & Classification

Pipeline lai (Hybrid AI) gom 3 buoc:

1. **YOLO** — phat hien va khoanh vung te bao
2. **OpenCV** — trich xuat dac trung hinh thai / mau / ket cau
3. **ML (KNN / Decision Tree / SVM)** — phan loai Platelets, RBC, WBC

## Cau truc thu muc

```text
.
|-- data/
|   |-- train|valid|test/     # Anh + nhan YOLO
|   `-- processed/features/   # train_features.csv
|-- models/
|   |-- yolo/best.pt          # Model detection (YOLO)
|   `-- ml/best_ml_model.pt   # Model classification (PyTorch .pt)
|-- scripts/                  # Script chay pipeline
|-- outputs/                  # Ket qua suy luan
`-- src/                      # Source code chinh
```

## 1. Cai dat

Mo terminal tai thu muc project:

```powershell
cd D:\BTL_AIT2004_2_Panacea

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Moi lan mo terminal moi, chay lai:

```powershell
cd D:\BTL_AIT2004_2_Panacea
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
```

## 2. Chuan bi model YOLO (da train xong)

Copy `best.pt` vao `models/yolo/`:

```powershell
New-Item -ItemType Directory -Force -Path models\yolo | Out-Null
Copy-Item runs\cell_detect\weights\best.pt models\yolo\best.pt
```

> Neu file da co san trong `models/yolo/best.pt` thi bo qua buoc nay.

## 3. Train model ML (luu file `.pt`)

### Buoc 3a — Tao file dac trung `train_features.csv`

```powershell
python scripts/build_train_features_from_labels.py
```

Hoac gop buoc 3a + 3b:

```powershell
python scripts/run_ml_pipeline.py
```

### Buoc 3b — Train va luu `models/ml/best_ml_model.pt`

```powershell
python scripts/train_ml.py
```

Ket qua: so sanh KNN / Decision Tree / SVM, chon model tot nhat, luu dang **PyTorch checkpoint** (`.pt`) kem danh sach cot feature.

## 4. Chay suy luan (end-to-end)

### Mot anh

```powershell
python scripts/infer_image.py `
  --image data/test/images/BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg `
  --output outputs/prediction.png `
  --feature-csv outputs/predicted_features.csv
```

### Full folder (tat ca anh — khong can go tung anh)

```powershell
python scripts/infer_folder.py
```

Mac dinh doc `data/test/images/`, ket qua:

| Output | Noi dung |
|--------|----------|
| `outputs/predictions/*.png` | Anh co bbox + nhan tung te bao |
| `outputs/features/*.csv` | Dac trung + nhan tung cell moi anh |
| `outputs/summary.csv` | Tong hop: so luong Platelets / RBC / WBC moi anh |

Chay folder khac (vd. valid):

```powershell
python scripts/infer_folder.py --images-dir data/valid/images
```

Tham so tuy chon:

| Tham so | Mac dinh | Mo ta |
|---------|----------|-------|
| `--image` | (bat buoc) | Duong dan anh dau vao |
| `--output` | `outputs/prediction.png` | Anh co bbox + nhan |
| `--feature-csv` | `outputs/predicted_features.csv` | CSV dac trung + du doan |
| `--yolo-model` | `models/yolo/best.pt` | Model YOLO |
| `--ml-model` | `models/ml/best_ml_model.pt` | Model ML |

**Ket qua:**

- `outputs/prediction.png` — anh da ve khung + nhan (Platelets / RBC / WBC)
- `outputs/predicted_features.csv` — bang dac trung va nhan du doan tung te bao

## 5. Tom tat lenh (copy nhanh)

```powershell
cd D:\BTL_AIT2004_2_Panacea
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# Train ML (khong train YOLO)
python scripts/run_ml_pipeline.py

# Suy luan TAT CA anh trong folder test
python scripts/infer_folder.py

# Hoac chi 1 anh
python scripts/infer_image.py --image data/test/images/<ten-anh>.jpg
```

## Luu y

- Model ML luu dang `.pt` (PyTorch), khong dung `.joblib`.
- Neu thieu `models/yolo/best.pt`, pipeline van chay nhung chi detect 1 vung (fallback toan anh).
- Dataset da co san trong `data/train`, `data/valid`, `data/test` (din dang YOLO).
