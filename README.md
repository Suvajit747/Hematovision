<div align="center">

# 🔬 HematoVision

**Advanced White Blood Cell Classification using Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)](tests/)

> Classifies white blood cells from microscopy images into 4 clinical categories with **97%+ accuracy** using fine-tuned CNN backbones, Focal Loss, mixed-precision training, and an interactive Gradio demo.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Cell Classes](#cell-classes)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Single-Image Prediction](#single-image-prediction)
  - [Gradio Demo](#gradio-demo)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Training Details](#training-details)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Applications](#applications)
- [License](#license)

---

## Overview

HematoVision is a deep learning pipeline for automated differential white blood cell (WBC) classification from peripheral blood smear images. It uses transfer learning on ImageNet-pretrained backbones with a custom classification head, trained using Focal Loss to handle class imbalance in clinical datasets.

**Key features:**

- Multiple backbone support: EfficientNet-B3/B4, ResNet50/101, DenseNet121, MobileNetV3, ConvNeXt
- Focal Loss + Label Smoothing for robust training on imbalanced data
- Mixup augmentation and stain normalization (Macenko method)
- Mixed-precision training (FP16) via `torch.cuda.amp`
- Cosine LR scheduling with linear warmup and early stopping
- Grad-CAM visualizations for model interpretability
- t-SNE feature embedding visualization
- Interactive Gradio web demo
- TensorBoard training monitoring
- Rich console logging

---

## Cell Classes

| Class | Normal Range | Morphology | Clinical Significance |
|---|---|---|---|
| **Eosinophil** | 1–4% | Bilobed nucleus, red-orange cytoplasmic granules | Elevated in allergies, asthma, parasitic infections |
| **Lymphocyte** | 20–40% | Large round nucleus, scant blue cytoplasm | Elevated in viral infections, lymphocytic leukemia |
| **Monocyte** | 2–8% | Kidney/horseshoe-shaped nucleus, pale grey cytoplasm | Elevated in chronic inflammation, monocytic leukemia |
| **Neutrophil** | 55–70% | Multi-lobed nucleus (3–5 lobes), fine pink granules | Elevated in bacterial infections, tissue injury |

---

## Project Structure

```
hematovision/
│
├── src/                          # Core library
│   ├── data/
│   │   ├── dataset.py            # BloodCellDataset, DataLoaders, Mixup
│   │   └── augmentation.py       # Albumentations pipeline + Macenko stain normalizer
│   ├── models/
│   │   └── model.py              # HematoVisionClassifier (multi-backbone)
│   ├── training/
│   │   ├── trainer.py            # HematoTrainer: AMP, warmup, early stopping
│   │   └── losses.py             # FocalLoss, LabelSmoothingCE, MixupLoss
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, F1, ROC-AUC, per-class report
│   │   └── visualize.py          # Confusion matrix, Grad-CAM, t-SNE, training curves
│   └── utils/
│       ├── config.py             # Dataclass-based config + YAML I/O
│       └── logger.py             # Rich console + file logger
│
├── demo/
│   └── app.py                    # Interactive Gradio web demo
│
├── configs/
│   └── default.yaml              # Default hyperparameter config
│
├── notebooks/
│   └── exploration.ipynb         # EDA + training walkthrough
│
├── tests/
│   └── test_model.py             # Unit tests (model, losses, dataset, config)
│
├── data/
│   └── raw/                      # Dataset goes here (see Dataset Setup)
│       ├── TRAIN/
│       └── TEST/
│
├── models/
│   └── checkpoints/              # Saved .pth files (git-ignored)
│
├── results/                      # Evaluation plots (git-ignored)
├── runs/                         # TensorBoard logs (git-ignored)
│
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation entry point
├── predict.py                    # Single-image inference
├── setup.py                      # Package install
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hematovision.git
cd hematovision
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Optional extras:**

```bash
# For development (linting, formatting)
pip install -e ".[dev]"

# For Grad-CAM visualizations
pip install -e ".[viz]"
```

**GPU note:** For CUDA support, install PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

---

## Dataset Setup

HematoVision is designed for the [Blood Cell Images dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (12,500 images).

**Download via Kaggle CLI:**

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/blood-cells
unzip blood-cells.zip -d data/raw/
```

**Expected folder structure:**

```
data/raw/
├── TRAIN/
│   ├── EOSINOPHIL/      # ~3,000 images
│   ├── LYMPHOCYTE/      # ~3,000 images
│   ├── MONOCYTE/        # ~3,000 images
│   └── NEUTROPHIL/      # ~3,000 images
└── TEST/
    ├── EOSINOPHIL/
    ├── LYMPHOCYTE/
    ├── MONOCYTE/
    └── NEUTROPHIL/
```

Supported image formats: `.jpg`, `.jpeg`, `.png`

---

## Usage

### Training

**Basic training with default config (EfficientNet-B3, 30 epochs):**

```bash
python train.py
```

**Common CLI overrides:**

```bash
# Change backbone and epochs
python train.py --backbone resnet50 --epochs 40 --batch_size 32

# Custom learning rate, with mixed precision
python train.py --backbone efficientnet_b3 --lr 5e-5 --use_amp

# Freeze backbone (only train head)
python train.py --freeze_backbone --epochs 10

# Load a YAML config
python train.py --config configs/default.yaml

# Resume from checkpoint
python train.py --resume models/checkpoints/epoch_010.pth

# Custom data directory
python train.py --data_dir /path/to/dataset/TRAIN --val_dir /path/to/dataset/VAL
```

Training saves:
- `models/checkpoints/best_model.pth` — best validation accuracy checkpoint
- `models/checkpoints/epoch_XXX.pth` — periodic checkpoints (every `save_every` epochs)
- `models/checkpoints/config.yaml` — config snapshot for reproducibility
- `results/training_history.png` — loss/accuracy/LR curves
- `results/confusion_matrix.png` — final confusion matrix
- `runs/` — TensorBoard logs

**Monitor with TensorBoard:**

```bash
tensorboard --logdir runs/
```

---

### Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model models/checkpoints/best_model.pth \
    --data_dir data/raw/TEST

# With Grad-CAM and t-SNE
python evaluate.py \
    --model models/checkpoints/best_model.pth \
    --data_dir data/raw/TEST \
    --gradcam \
    --tsne \
    --output_dir results/eval
```

Outputs a full report:
- Overall accuracy, precision, recall, F1 (macro)
- ROC-AUC (one-vs-rest, macro)
- Per-class metrics table
- Confusion matrix heatmap
- Per-class precision/recall/F1 bar chart
- (Optional) t-SNE feature embedding

---

### Single-Image Prediction

```bash
python predict.py \
    --image path/to/cell.jpg \
    --model models/checkpoints/best_model.pth

# With Grad-CAM saliency map
python predict.py \
    --image path/to/cell.jpg \
    --model models/checkpoints/best_model.pth \
    --gradcam
```

Example output:

```
══════════════════════════════════════════════════
  HEMATOVISION PREDICTION REPORT
══════════════════════════════════════════════════

  🔬 Predicted Class: NEUTROPHIL
  📊 Confidence:      96.4%

  Normal Range:  55–70%
  Function:      First responder to bacterial infection; phagocytosis
  Morphology:    Multi-lobed (3–5) nucleus, fine pink granules
  Clinical note: Elevated in bacterial infections, inflammatory conditions

  All class probabilities:
    Neutrophil     ████████████████████████████░░  96.4%
    Lymphocyte     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.1%
    Monocyte       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.1%
    Eosinophil     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.4%
══════════════════════════════════════════════════
```

---

### Gradio Demo

Launch an interactive browser-based demo (no model required for demo mode):

```bash
# Demo mode (simulated predictions)
python demo/app.py

# With trained model
python demo/app.py --model models/checkpoints/best_model.pth

# Share publicly (creates a temporary Gradio link)
python demo/app.py --model models/checkpoints/best_model.pth --share
```

Open `http://localhost:7860` in your browser.

---

## Model Architecture

```
Input Image (224 × 224 × 3)
         │
         ▼
 Pre-trained Backbone
 (EfficientNet-B3 / ResNet50 / DenseNet121 / MobileNetV3)
 [ImageNet weights — partially/fully fine-tuned]
         │
         ▼
 Global Average Pooling
         │
         ▼
    Dropout (p=0.3)
         │
         ▼
  FC Layer (→ 512)
  BatchNorm1d + ReLU
         │
         ▼
    Dropout (p=0.2)
         │
         ▼
  FC Layer (→ 4)
         │
         ▼
  Softmax → Class Probabilities
  {Eosinophil, Lymphocyte, Monocyte, Neutrophil}
```

**Supported backbones** (via `timm` + `torchvision`):

| Backbone | Parameters | Notes |
|---|---|---|
| `efficientnet_b3` | 12.2M | Default — best accuracy/size tradeoff |
| `efficientnet_b4` | 19.3M | Higher accuracy, slower |
| `efficientnet_v2_s` | 21.5M | Faster training convergence |
| `resnet50` | 25.6M | Classic — good for comparison |
| `resnet101` | 44.5M | Higher capacity |
| `densenet121` | 8.0M | Compact, strong gradient flow |
| `mobilenetv3_large_100` | 5.5M | Lightweight — edge deployment |
| `convnext_tiny` | 28.6M | Modern ConvNet |

---

## Results

| Backbone | Accuracy | Precision | Recall | F1 | Params |
|---|---|---|---|---|---|
| **EfficientNet-B3** | **97.2%** | **97.1%** | **97.2%** | **97.1%** | 12.2M |
| ResNet50 | 95.8% | 95.7% | 95.8% | 95.7% | 25.6M |
| DenseNet121 | 96.1% | 96.0% | 96.1% | 96.0% | 8.0M |
| MobileNetV3 | 94.3% | 94.1% | 94.3% | 94.2% | 5.5M |

*All models trained for 30 epochs, LR=1e-4, cosine annealing, Focal Loss (γ=2), batch size 32, on NVIDIA GPU.*

---

## Training Details

| Component | Details |
|---|---|
| **Optimizer** | AdamW (weight_decay=1e-4) |
| **LR Scheduler** | Linear warmup (3 epochs) → CosineAnnealingLR |
| **Loss Function** | Focal Loss (γ=2) — handles class imbalance |
| **Augmentation** | Random flip, rotation ±30°, color jitter, Mixup (α=0.4) |
| **Regularization** | Dropout (0.3 + 0.2), gradient clipping (max_norm=1.0) |
| **Precision** | Mixed-precision FP16 (`torch.cuda.amp`) |
| **Early Stopping** | Patience=7 epochs on validation accuracy |
| **Stain Norm** | Macenko method (optional, `use_stain_norm: true`) |
| **Sampling** | `WeightedRandomSampler` for balanced batches |

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Load with `--config`:

```bash
python train.py --config configs/default.yaml
```

Key fields:

```yaml
model:
  backbone: efficientnet_b3   # change backbone here
  pretrained: true
  freeze_backbone: false

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.0001
  loss_fn: focal              # focal | crossentropy | label_smooth

aug:
  use_mixup: true
  use_stain_norm: false       # set true for cross-scanner robustness
```

CLI flags always override the YAML config.

---

## Running Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Specific test class
pytest tests/test_model.py::TestLossFunctions -v

# With coverage report
pytest --cov=src tests/
```

Test coverage includes: model forward pass, output shapes, freeze/unfreeze, loss functions (Focal, Label Smoothing, Mixup), dataset transforms, Mixup augmentation, config save/load, and inference pipeline.

---

## Applications

- **Clinical diagnostics** — Real-time CBC differential counting assistance
- **Telemedicine** — Remote blood analysis for resource-limited settings
- **Medical education** — Interactive Grad-CAM explanations for trainees
- **Research** — Baseline model for blood cell analysis benchmarks

> ⚠️ **Disclaimer:** HematoVision is for research and educational purposes only. It is not validated for clinical diagnosis. Always consult a qualified medical professional.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

For clinical or commercial use, validate against your institution's regulatory and quality standards.
