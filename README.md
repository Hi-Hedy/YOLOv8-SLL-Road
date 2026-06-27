# YOLOv8-SLL-Road: Enhanced Lightweight GPR Road Defect Detection

This repository contains the official implementation, model weights, and dataset for the paper:

**"YOLOv8-SLL-Road: An Enhanced Lightweight Network for Efficient and Accurate Detection of Road Defects Using GPR Data"**

**Author:** [Hi-Hedy](https://github.com/Hi-Hedy)

[![GitHub stars](https://img.shields.io/github/stars/Hi-Hedy/ultralytics-main-SLL)](https://github.com/Hi-Hedy/ultralytics-main-SLL)

---

## 1. Overview

YOLOv8-SLL-Road integrates four key enhancements into YOLOv8n for Ground Penetrating Radar (GPR) B-scan road defect detection:

| Module | Description |
|--------|-------------|
| **SCConv** | Spatial-Channel Reconstruction Convolution for lightweight feature extraction |
| **LSKA** | Large Separable Kernel Attention for long-range context modelling |
| **WIoU v3** | Wise-IoU loss with dynamic outlier-aware focus adjustment |
| **VFL** | Varifocal Loss with IoU-aware classification weighting |


---

## 2. Dataset

### GPR-Road Dataset

To enhance model generalization and mitigate overfitting, we performed data augmentation on the original GPR images, including random brightness adjustment, contrast adjustment, and noise injection. Specifically, we applied the above augmentation strategies to 3,200 FDTD simulated samples and 1,000 field-measured samples, generating 543 additional augmented samples.

The GPR-Road dataset comprises a total of **5,270 samples**:

| Split | Source | Count |
|-------|--------|-------|
| **Train/Val (Internal)** | FDTD simulated | 3,200 |
| | Field-measured | 1,000 |
| | Augmented | 543 |
| | *Subtotal* | *4,743* (split 7:2) |
| **Test (External)** | Public samples | 527 |

> The 527 public test samples serve as an independent external test set, not involved in training/validation or hyperparameter tuning.

### Download

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Dataset-blue?logo=googledrive)](https://drive.google.com/drive/folders/1lxtU2vtmPg_OdfmG4Y3P1aA4cQHEANMw?usp=drive_link)

| Defect Class | Description |
|-------------|-------------|
| Void | Internal cavity / air void |
| Delamination | Debonding between pavement layers |
| Looseness | Loose / uncompacted zone |
| Water-rich | Water-saturated zone |

---

## 3. Installation

```bash
# Clone this repository
git clone https://github.com/Hi-Hedy/ultralytics-main-SLL.git
cd ultralytics-main-SLL

# Install dependencies
pip install -r requirements.txt
```

> **Note:** This repository is based on Ultralytics YOLOv8 v8.0.228. All custom modules are self-contained in `ultralytics/nn/extra_modules/` and registered in `ultralytics/nn/tasks.py`. No additional manual integration is required.

---

## 4. Training

### Quick Start

```bash
python datasets/defects/train_defects.py
```

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `imgsz` | 640 | Input image size |
| `epochs` | 200 | Total training epochs |
| `batch` | 32 | Batch size |
| `optimizer` | SGD | Optimizer |
| `lr0` | 0.01 | Initial learning rate |
| `cos_lr` | True | Cosine annealing schedule |
| `momentum` | 0.937 | SGD momentum |
| `weight_decay` | 0.0005 | L2 regularization |
| `close_mosaic` | 10 | Disable mosaic in last 10 epochs |
| `label_smoothing` | 0.0 (VFL) / 0.1 (BCE) | Label smoothing factor |

### Reproducibility

The training script includes a seed configuration block for reproducible results:

```python
SEED = None     # No fixed seed
# SEED = 42     # Toggle for reproducibility
# SEED = 1234
# SEED = 2025
```

---

## 5. Evaluation

```bash
# Validate trained model
python datasets/defects/val.py

# Or via YOLO CLI
yolo val model=weights/best.pt data=datasets/defects/defects.yaml
```

---

## 6. Project Structure

```
ultralytics-main-SLL/
├── ultralytics/
│   ├── nn/
│   │   ├── extra_modules/
│   │   │   ├── ScConv.py                # SCConv, C2f_SCConv, Bottleneck_SCConv
│   │   │   └── LSKAttention.py          # LSKA attention module
│   │   ├── modules/                     # Standard YOLOv8 modules
│   │   └── tasks.py                     # Model parser
│   ├── utils/
│   │   ├── loss.py                      # Varifocal Loss + WIoU-integrated BboxLoss
│   │   └── metrics.py                   # WIoU v3 bbox_iou implementation
├── datasets/defects/
│   ├── defects.yaml                     # Dataset configuration
│   ├── train_defects.py                 # Training script
│   ├── val.py                           # Validation script
│   └── predict.py                       # Inference script
├── requirements.txt
├── setup.py
└── README.md
```

---

## 7. License

This project is released under the [AGPL-3.0](LICENSE) license, following Ultralytics YOLOv8.

## 8. Citation

If you use this work in your research, please cite:

```bibtex
@article{hedy2025yolov8sllroad,
  title={YOLOv8-SLL-Road: An Enhanced Lightweight Network for Efficient
         and Accurate Detection of Road Defects Using GPR Data},
  author={Hedy},
  year={2025}
}
```
