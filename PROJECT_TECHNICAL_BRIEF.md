# MVTec XAI - Technical Project Brief

## 1. Project Overview
This project is an explainable anomaly detection system built on the MVTec AD dataset (Bottle category). It classifies industrial product images into:
- normal
- anomalous

Beyond classification, it explains each prediction using:
- Grad-CAM (gradient-based visual explanation)
- LIME (perturbation-based local explanation)

The core goal is to make model decisions interpretable for quality-inspection use cases.

## 2. Problem Statement
In industrial visual inspection, it is not enough to only predict whether an item is defective. Stakeholders need to know:
- where the model looked
- why it predicted a defect

This project addresses both:
- binary anomaly classification
- visual reasoning outputs for trust and auditability

## 3. Dataset and Structure
The project uses the MVTec AD Bottle class under `data/bottle/`.

High-level structure:
- `train/good/` -> only normal images
- `test/good/`, `test/broken_large/`, `test/broken_small/`, `test/contamination/`
- `ground_truth/` -> segmentation masks for localization evaluation

Important dataset characteristic:
- Defect classes are available in `test/`, not `train/`.

## 4. Model Architecture
The classifier is based on ResNet-50 from torchvision.

Implementation details (`src/model.py`):
- Start from ImageNet-pretrained ResNet-50
- Replace final fully connected layer:
  - original: 2048 -> 1000
  - new: 2048 -> 2 (normal vs anomalous)

Reasoning:
- Transfer learning is used because dataset size is relatively small.
- Pretrained convolutional features improve generalization and convergence speed.

## 5. Configuration and Runtime
Centralized in `src/config.py`.

Key settings:
- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 16`
- `NUM_EPOCHS = 15`
- `LEARNING_RATE = 1e-4`
- `NUM_WORKERS = 2`
- `SEED = 42`

Device selection priority:
1. Apple MPS (for M-series Macs)
2. CUDA (if available)
3. CPU fallback

This makes the code portable across hardware while optimizing speed on Apple Silicon.

## 6. Data Pipeline and Split Strategy
Pipeline (from `src/dataset.py` and `src/train.py`):
1. Build train/test datasets with ImageNet normalization
2. Split test set into:
   - additional supervised training subset
   - held-out validation subset
3. Stratify split by labels to keep class balance

Why this split is needed:
- MVTec train split has only good images.
- To train a supervised binary classifier, the model must see defective samples.

## 7. Training Workflow
Implemented in `src/train.py`.

Training loop:
1. Set random seeds for reproducibility
2. Build dataloaders (with class-balancing support)
3. Train for each epoch
4. Validate each epoch on held-out split
5. Track best validation performance
6. Save best checkpoint to `checkpoints/best_model.pt`

Outputs:
- best model checkpoint for inference and explainability scripts

## 8. Inference Workflow
Implemented in `src/predict.py`.

Behavior:
- Loads `checkpoints/best_model.pt`
- Applies evaluation transforms
- Returns:
  - predicted label (`normal` or `anomalous`)
  - confidence score

Optimization:
- Model and transform are cached at module level to avoid reloading overhead on repeated predictions.

## 9. Explainability Modules

### 9.1 Grad-CAM (`src/gradcam_explain.py`)
- Target layer: last convolutional block (`model.layer4[-1]`)
- Produces:
  - normalized heatmap
  - RGB overlay on original image
  - predicted class and confidence

Interpretation value:
- Highlights spatial regions most responsible for model decision.

### 9.2 LIME (`src/lime_explain.py`)
- Perturbs image regions and observes prediction impact
- Produces local feature-importance explanations

Interpretation value:
- Model-agnostic check of local decision behavior, complementary to Grad-CAM.

## 10. Evaluation and Outputs
Generated artifacts in `outputs/` include:
- `classification_metrics.txt`
- `predictions.csv`
- `gradcam_iou.csv`
- `gradcam_iou_thresholds.csv`
- `localization_iou.csv`
- visual exports in `outputs/visuals/`

These support both:
- classification quality assessment
- explanation/localization quality assessment (via IoU)

## 11. End-to-End Execution Order
From project root:

```bash
python src/dataset.py
python src/model.py
python src/train.py
python src/predict.py data/bottle/test/broken_large/000.png
python src/gradcam_explain.py data/bottle/test/broken_large/000.png
python src/lime_explain.py data/bottle/test/broken_large/000.png
```

## 12. Technical Strengths
- Clear modular architecture (`config`, `dataset`, `model`, `train`, `predict`, `explain`)
- Explainable AI by design (not bolted on later)
- Hardware-aware runtime (Apple MPS support)
- Reproducibility controls through fixed seeds and structured checkpoints
- Evaluation pipeline for both prediction and explanation quality

## 13. Meeting-Ready Talking Points
1. Why transfer learning:
   - ResNet-50 pretrained features avoid overfitting on a small industrial dataset.
2. Why custom split strategy:
   - MVTec train lacks defects, so supervised classification requires carefully reusing part of test defects.
3. Why Grad-CAM plus LIME:
   - Two different explanation paradigms provide stronger trust and cross-validation of model behavior.
4. Why this is practical:
   - Fast inference, interpretable outputs, and measurable quality metrics make it suitable for QA workflows.
