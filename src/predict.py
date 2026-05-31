"""
Inference helper.  Loads the trained model and predicts on one or more images.

Used by:
  - The Streamlit dashboard (to predict on user-clicked images)
  - The Grad-CAM and LIME scripts (they need a prediction function)
  - You, on the command line, to do a quick sanity check:
        python src/predict.py path/to/image.png
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from config import CHECKPOINTS_DIR, DEVICE
from dataset import build_eval_transform
from model import load_checkpoint


CLASS_NAMES = ["normal", "anomalous"]


# A simple module-level cache so we do not reload the model on every call
# (loading ResNet-50 from disk takes ~1 second on M4).
_MODEL = None
_TRANSFORM = None


def get_model():
    global _MODEL
    if _MODEL is None:
        ckpt = CHECKPOINTS_DIR / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}.  "
                f"Train the model first with: python src/train.py"
            )
        _MODEL = load_checkpoint(ckpt, DEVICE)
    return _MODEL


def get_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = build_eval_transform()
    return _TRANSFORM


def predict_image(image_path: Path) -> Tuple[str, float]:
    """
    Returns (predicted_class_name, confidence_in_that_class).
    """
    image = Image.open(image_path).convert("RGB")
    tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
        idx    = int(probs.argmax().item())
        conf   = float(probs[idx].item())

    return CLASS_NAMES[idx], conf


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <path-to-image>")
        sys.exit(1)

    path = Path(sys.argv[1])
    label, conf = predict_image(path)
    print(f"{path.name}  ->  {label}  ({conf:.3f})")
