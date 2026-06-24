"""
Helper script to compute LIME in an isolated subprocess.

This script is called by the dashboard to avoid the loky semaphore
segfault that crashes the main Streamlit process on Python 3.13 + macOS.

Usage:
    python src/lime_subprocess.py <image_path> <model_type> <num_samples> <num_features> <output_path>
"""

import sys
import os

# Force CPU mode
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

# Override config to use CPU
import config
import torch
config.DEVICE = torch.device("cpu")

from lime_explain import compute_lime


def main():
    image_path = Path(sys.argv[1])
    model_type = sys.argv[2]
    num_samples = int(sys.argv[3])
    num_features = int(sys.argv[4])
    output_path = Path(sys.argv[5])

    _, lime_overlay, pred_class, confidence = compute_lime(
        image_path,
        num_samples=num_samples,
        num_features=num_features,
        model_type=model_type
    )

    # Save the overlay image
    Image.fromarray(lime_overlay).save(output_path)
    
    # Force immediate exit to skip loky cleanup (which causes the segfault)
    os._exit(0)


if __name__ == "__main__":
    main()
