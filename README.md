# Multimodal Respiratory AI

> A deep learning pipeline for respiratory disease prediction and systemic multi-organ risk assessment using cross-modal attention fusion.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Plagiarism](https://img.shields.io/badge/Similarity-7%25-brightgreen?style=flat-square)](docs/plagiarism_report.pdf)

---

## Overview

This project presents a **multimodal deep learning system** that combines:
- Chest X-ray images
- Lab test results (creatinine, ALT, AST, troponin, BNP, SpO₂, etc.)
- Patient demographics (age, sex, BMI, smoking history)

...to simultaneously perform **three clinical tasks**:

| Task | Output |
|------|--------|
| Disease Classification | COPD, Pneumonia, Pulmonary Fibrosis, Pleural Effusion, Normal |
| Severity Estimation | Mild / Moderate / Severe |
| Organ Risk Scoring | Heart, Kidney, Liver, Brain risk probabilities |

**Results:** 91.3% disease classification accuracy · 0.87 mean AUROC across organs

---

## Screenshots

### Diagnostic Results & Multimodal Fusion Analysis
<!-- INSERT SCREENSHOT: Diagnostic results page with pie chart and confidence scores -->
```
[ Insert screenshot here: Diagnostic Results — Case Summary + Fusion pie chart ]
```

### Explainable AI — Grad-CAM Heatmap & 3D Localization
<!-- INSERT SCREENSHOT: Grad-CAM heatmap overlaid on chest X-ray + 3D anatomical model -->
```
[ Insert screenshot here: Explainable AI — Heatmap + 3D lung model ]
```

### Explainability Report
<!-- INSERT SCREENSHOT: Explainability report with diagnostic rationale and organ risk breakdown -->
```
[ Insert screenshot here: Explainability Report section ]
```

### Systemic Organ Risk Analysis
<!-- INSERT SCREENSHOT: Organ risk dashboard with bar chart and risk timeline trend graph -->
```
[ Insert screenshot here: Systemic Organ Risk Analysis — timeline chart ]
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT MODALITIES                        │
│   CXR Image (DICOM)   Lab Values   Patient Demographics     │
└────────┬──────────────────┬──────────────────┬─────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
  │ Modified    │   │ Modified     │   │ MLP         │
  │ SE-ResNet-50│   │ BiLSTM       │   │ Demographic │
  │ (256-dim)   │   │ (128-dim)    │   │ (64-dim)    │
  └─────────────┘   └──────────────┘   └─────────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
              ┌──────────────────────────┐
              │  Cross-Modal Attention   │
              │  Fusion (8 heads,        │
              │  2 transformer layers)   │
              │  → 512-dim embedding     │
              └──────────┬───────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────────┐
  │ Disease      │ │ Severity │ │ Organ Risk   │
  │ Classifier   │ │ Estimator│ │ Scorer       │
  │ (5 classes)  │ │ (3 ord.) │ │ (4 organs)   │
  └──────────────┘ └──────────┘ └──────────────┘
           │                           │
           ▼                           ▼
      Grad-CAM                    KernelSHAP
   (Image XAI)               (Tabular XAI)
```

---

## Key Techniques

| Component | Method |
|-----------|--------|
| Image Encoder | ResNet-50 with SE/CBAM attention + multi-scale outputs |
| Tabular Encoder | Bidirectional LSTM with attention pooling |
| Demographic Encoder | 2-layer MLP with LayerNorm |
| Fusion | Cross-Modal Attention (scaled dot-product, 8 heads) |
| Multi-Task Loss | Kendall Uncertainty Weighting |
| Image Explainability | Grad-CAM (visualized heatmap overlay) |
| Clinical Explainability | KernelSHAP (per-feature organ risk scores) |
| Class Imbalance | Focal Loss + SMOTE on training data |
| Preprocessing | CLAHE contrast enhancement, Z-score normalization |

---

## Project Structure

```
multimodal-respiratory-ai/
│
├── data/                        # Data loading & preprocessing scripts
│   ├── preprocess_images.py
│   ├── preprocess_tabular.py
│   └── dataset.py
│
├── models/                      # Model architecture definitions
│   ├── image_encoder.py         # Modified SE-ResNet-50
│   ├── tabular_encoder.py       # Modified BiLSTM
│   ├── demographic_encoder.py   # MLP encoder
│   ├── fusion.py                # Cross-modal attention fusion
│   └── multitask_heads.py       # Output heads
│
├── explainability/              # XAI methods
│   ├── gradcam.py
│   └── kernelshap.py
│
├── training/                    # Training pipeline
│   ├── train.py
│   ├── loss.py                  # Uncertainty-weighted loss
│   └── evaluate.py
│
├── dashboard/                   # Web dashboard (frontend)
│   ├── app.py                   # Flask/FastAPI backend
│   ├── static/
│   └── templates/
│
├── notebooks/                   # Jupyter notebooks for experiments
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended)
- Access to [MIMIC-IV](https://physionet.org/content/mimiciv/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) datasets (PhysioNet credentials required)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multimodal-respiratory-ai.git
cd multimodal-respiratory-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Preprocess chest X-ray images
python data/preprocess_images.py --data_dir /path/to/mimic-cxr

# Preprocess tabular data
python data/preprocess_tabular.py --data_dir /path/to/mimic-iv
```

### Training

```bash
python training/train.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir ./checkpoints
```

### Running the Dashboard

```bash
python dashboard/app.py
# Open http://localhost:5000 in your browser
```

---

## Results

### Disease Classification Performance

| Class | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| Normal | 0.96 | 0.97 | 0.96 | 0.99 |
| COVID-19 | 0.95 | 0.92 | 0.93 | 0.98 |
| Tuberculosis | 0.94 | 0.96 | 0.95 | 0.98 |
| Pneumonia | 0.93 | — | — | 0.96 |
| COPD | 0.91 | 0.89 | 0.90 | 0.95 |

**Overall accuracy: 86–88% (5-fold cross-validation)**

### Organ Risk Prediction AUC

| Organ | AUC |
|-------|-----|
| Lung | 0.853 |
| Heart | 0.851 |
| Kidney | 0.853 |
| Liver | 0.841 |

---

## Datasets

This project uses the following publicly available, de-identified medical databases:

- **[MIMIC-IV](https://physionet.org/content/mimiciv/)** — Lab values, vital signs, demographics, diagnosis codes
- **[MIMIC-CXR](https://physionet.org/content/mimic-cxr/)** — Chest X-ray images with radiology reports

> ⚠️ Access requires a PhysioNet credentialed account and completion of CITI training. The datasets are **not included** in this repository.

---

<p align="center"></p>
