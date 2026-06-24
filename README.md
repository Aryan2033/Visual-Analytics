# MVTec XAI — Explainable Anomaly Detection

An end-to-end **Explainable AI (XAI)** system for industrial visual quality control. This project classifies product images from the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (bottles) as **normal** or **defective**, and provides deep visual reasoning for its predictions using **Grad-CAM** and **LIME**.

## 🎯 The Goal

Traditional deep learning models act as "black boxes" — they can classify a product as defective but cannot explain *why*. In a manufacturing setting, quality inspectors need to **trust** the AI's decisions.

This system solves that by not only predicting anomalies but also **visually highlighting the exact regions** (like cracks or contamination) that triggered the prediction. We evaluate two distinct model approaches:

1. **ResNet-50 (Transfer Learning)** — Pretrained on ImageNet, fine-tuned on our bottle data.
2. **Custom 2D CNN (From Scratch)** — A lightweight 4-layer CNN to demonstrate why transfer learning matters.

## 🚀 Features

- **Dual-Model Comparison** — See how a 23M-parameter ResNet-50 vs a 106K-parameter custom CNN perform on the same image.
- **Grad-CAM** (Gradient-weighted Class Activation Mapping) — Highlights *where* the model looked using gradient-based heatmaps.
- **LIME** (Local Interpretable Model-agnostic Explanations) — Reveals *which regions* matter by probing the model with perturbations.
- **Interactive Streamlit Dashboard** — Upload any bottle image, get both models' predictions + visual explanations on a single page.
- **Zoom-Ready Layout** — Compact 3-column design with expanders, no scrolling required.

## 📊 Model Performance

| Metric | ResNet-50 (Transfer Learning) | Custom 2D CNN (From Scratch) |
|--------|:---:|:---:|
| **Parameters** | 23,512,130 | 106,306 |
| **Accuracy** | **97.59%** | 75.90% |
| **Precision** | **98.41%** | 75.90% |
| **Recall** | 98.41% | **100%** |
| **F1 Score** | **98.41%** | 86.30% |
| **ROC AUC** | **98.89%** | 58.73% |

> **Key Finding:** The Custom CNN predicts *everything* as anomalous (100% recall, 0% specificity). With only ~200 training images and no pretrained features, it defaults to the safest strategy. This powerfully demonstrates why **transfer learning is essential** in low-data industrial domains.

## 🛠️ Setup & Installation

**Prerequisites:** macOS (Apple Silicon MPS supported) or any Python 3.10+ environment.

```bash
# 1. Clone the repository
git clone https://github.com/AsimNayakawadi/Visual-Analytics.git
cd Visual-Analytics

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify PyTorch sees the GPU (optional, for Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Dataset Structure

Place the MVTec AD bottle dataset in the `data/` folder:

```
data/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   ├── broken_small/
    │   └── contamination/
    └── ground_truth/
        ├── broken_large/
        ├── broken_small/
        └── contamination/
```

## 💻 Running the Dashboard

```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501`. You can upload custom images or pick from the test set.

## 🧠 Pipeline Scripts (CLI)

You can also run each step individually from the command line:

```bash
# Verify dataset
python src/dataset.py

# Verify models build
python src/model.py

# Train ResNet-50 (~5-10 min on Apple M4)
python src/train.py

# Train Custom 2D CNN (~3-5 min)
python src/train_cnn.py

# Predict on a single image
python src/predict.py data/bottle/test/broken_large/000.png

# Generate Grad-CAM heatmaps
python src/gradcam_explain.py data/bottle/test/broken_large/000.png

# Generate LIME explanations (~30s)
python src/lime_explain.py data/bottle/test/broken_large/000.png

# Evaluate both models on the full test set
python src/eval_classification.py
python src/eval_classification_cnn.py
```

## 📁 Project Structure

| File | Purpose |
|:-----|:--------|
| `src/config.py` | Paths, hyperparameters, device selection |
| `src/dataset.py` | PyTorch Dataset for MVTec folder layout |
| `src/model.py` | ResNet-50 + Custom 2D CNN architectures |
| `src/train.py` | Fine-tunes ResNet-50 (saves to `checkpoints/`) |
| `src/train_cnn.py` | Trains the Custom CNN from scratch |
| `src/predict.py` | Inference on a single image |
| `src/gradcam_explain.py` | Grad-CAM heatmap generation |
| `src/lime_explain.py` | LIME superpixel explanations |
| `src/compute_xai.py` | Subprocess runner for dashboard XAI |
| `src/dashboard.py` | Streamlit interactive dashboard |
| `src/eval_classification.py` | Test-set metrics for ResNet-50 |
| `src/eval_classification_cnn.py` | Test-set metrics for Custom CNN |
| `notebooks/01_explore_data.ipynb` | Visual data exploration notebook |

## 🔬 XAI Methods Explained

### Grad-CAM
Uses gradients flowing into the last convolutional layer to produce a spatial heatmap showing which image regions most strongly activated the predicted class. It is **model-intrinsic** (looks inside the network).

### LIME
Treats the model as a **black box**. Creates hundreds of perturbed versions of the image (hiding random superpixels), observes how the prediction changes, and fits a simple linear model to identify which regions matter most.

### Why Both?
If Grad-CAM and LIME both highlight the same defect region, we gain **stronger confidence** that the model is making decisions for the right reasons. This cross-validation of explanations is critical for trust in production.

## 👥 Authors

- Aryan Jadhav
- Asim Nayakawadi

Visual Analytics Course — Summer 2026
