# MVTec XAI — Explainable Anomaly Detection on Image Data

Visual Analytics project, summer 2026.

A Streamlit dashboard that classifies industrial product images (MVTec AD)
as normal vs defective using a fine-tuned ResNet-50, and explains every
prediction with Grad-CAM and LIME side by side.


## Setup on macOS (Apple Silicon)

Tested on M4 MacBook Air.  PyTorch uses Apple's MPS backend, which runs on
the integrated GPU and is much faster than CPU.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies (takes a few minutes)
pip install -r requirements.txt

# 4. Verify PyTorch sees the GPU
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expected output: MPS available: True
```

## Place the dataset

Your friend has the MVTec `bottle` folder.  Drop it under `data/` so the
structure looks like this:

```
mvtec-xai/
└── data/
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

If your friend's folder has a different layout (e.g. an extra `bottle/`
inside `bottle/`), just adjust until the paths match.


## Run the pipeline

Run each command from the project root.  Each one only takes a few seconds
to verify, except `train.py` which takes ~5-10 minutes on M4.

```bash
# 1. Sanity-check the dataset loads
python src/dataset.py

# 2. Sanity-check the model builds
python src/model.py

# 3. Train the classifier (writes checkpoints/best_model.pt)
python src/train.py

# 4. Predict on a single image
python src/predict.py data/bottle/test/broken_large/000.png

# 5. Generate a Grad-CAM heatmap
python src/gradcam_explain.py data/bottle/test/broken_large/000.png
# -> creates gradcam_output.png in the project root

# 6. Generate a LIME explanation (slower, ~30 s)
python src/lime_explain.py data/bottle/test/broken_large/000.png
# -> creates lime_output.png
```

For a more visual exploration, open `notebooks/01_explore_data.ipynb` in
Jupyter:

```bash
jupyter notebook notebooks/01_explore_data.ipynb
```


## What is in each file

| File                          | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `src/config.py`               | Paths, hyperparameters, device selection                  |
| `src/dataset.py`              | PyTorch `Dataset` that reads the MVTec folder layout      |
| `src/model.py`                | Builds the ResNet-50 with a 2-class head                  |
| `src/train.py`                | Fine-tunes the model and saves the best checkpoint        |
| `src/predict.py`              | Loads the trained model and runs inference                |
| `src/gradcam_explain.py`      | Grad-CAM heatmap (where did the model look?)              |
| `src/lime_explain.py`         | LIME explanation (which regions matter?)                  |
| `notebooks/01_explore_data.ipynb` | Visual sanity check of the dataset                    |
| `checkpoints/`                | Trained model weights land here                           |
| `requirements.txt`            | Pinned package versions (required by the project brief)   |


## Project status

- [x] Stage 1 — Project skeleton + data exploration
- [x] Stage 2 — Dataset + dataloader
- [x] Stage 3 — Training loop
- [x] Stage 4 — Grad-CAM and LIME explainers
- [ ] Stage 5 — Streamlit dashboard
- [ ] Stage 6 — Quantitative evaluation (IoU vs ground-truth masks)
- [ ] Stage 7 — Slide deck + dry-run


## Team

(fill in)
