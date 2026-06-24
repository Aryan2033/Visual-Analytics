"""
Compute all XAI outputs (predictions, Grad-CAM, LIME) in an isolated subprocess.

Called by the dashboard. Saves all results to a JSON + images so the
dashboard only has to display them. Uses os._exit(0) at the end to
skip the loky/Python 3.13 cleanup segfault on macOS.

Usage:
    python src/compute_xai.py <image_path> <gradcam_threshold> <lime_samples> <lime_features> <output_dir>
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

# Force CPU before anything else
import config
import torch
config.DEVICE = torch.device("cpu")

import json
import time
import numpy as np
from PIL import Image

from predict import predict_image, CLASS_NAMES
from gradcam_explain import compute_gradcam
from lime_explain import compute_lime


def main():
    image_path = Path(sys.argv[1])
    gradcam_threshold = int(sys.argv[2])
    lime_samples = int(sys.argv[3])
    lime_features = int(sys.argv[4])
    output_dir = Path(sys.argv[5])
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    for m_type in ["resnet50", "cnn"]:
        # --- Prediction ---
        start = time.time()
        pred_class, confidence = predict_image(image_path, model_type=m_type)
        latency = (time.time() - start) * 1000

        results[m_type] = {
            "pred_class": pred_class,
            "confidence": float(confidence),
            "latency": float(latency),
        }

        # --- Grad-CAM ---
        heatmap, overlay, _, _ = compute_gradcam(image_path, model_type=m_type)
        Image.fromarray(overlay).save(output_dir / f"gradcam_{m_type}.png")
        # Save heatmap as numpy for thresholding in the dashboard
        np.save(output_dir / f"heatmap_{m_type}.npy", heatmap)

        # --- LIME ---
        _, lime_overlay, _, _ = compute_lime(
            image_path,
            num_samples=lime_samples,
            num_features=lime_features,
            model_type=m_type,
        )
        Image.fromarray(lime_overlay).save(output_dir / f"lime_{m_type}.png")

    # Save results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)

    # Force immediate exit — skips loky/multiprocessing cleanup
    # which causes segfaults on Python 3.13 + macOS
    os._exit(0)


if __name__ == "__main__":
    main()
