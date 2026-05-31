"""
Grad-CAM explanation.

Takes an image, returns:
  1. The 224x224 heatmap (numpy array of values 0..1)
  2. A nice visualization with the heatmap overlaid on the image

We use the `pytorch-grad-cam` library which handles the gradient bookkeeping
for us.  The only thing we have to tell it is "which layer to look at".
For ResNet-50, the right answer is the last conv block: `model.layer4`.

Usage:
    python src/gradcam_explain.py path/to/image.png

It will save the result as gradcam_output.png in the project root.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import DEVICE, IMAGE_SIZE
from dataset import build_eval_transform
from predict import CLASS_NAMES, get_model


def compute_gradcam(image_path: Path, target_class: int = None) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Returns:
        heatmap:        (224, 224) float array, values in [0, 1]
        overlay:        (224, 224, 3) uint8 image with the heatmap drawn over the original
        predicted_class: "normal" or "anomalous"
        confidence:     float in [0, 1]
    """
    # Load and preprocess
    pil_image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_float = np.array(pil_image).astype(np.float32) / 255.0   # used for the overlay
    tensor    = build_eval_transform()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    # Predict
    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())

    # Grad-CAM
    # For ResNet-50, layer4 is the last convolutional block.  This is the
    # standard target.  Grad-CAM will produce a 7x7 attribution map there
    # and upscale it to 224x224 for us.
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(target_class if target_class is not None else pred_idx)]

    # The library handles enabling gradients internally.  We use a `with`
    # block to make sure hooks are cleaned up afterwards.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        heatmap = cam(input_tensor=tensor, targets=targets)[0]   # (224, 224) float

    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    return heatmap, overlay, CLASS_NAMES[pred_idx], confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/gradcam_explain.py <path-to-image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    heatmap, overlay, label, conf = compute_gradcam(img_path)

    out_path = Path("gradcam_output.png")
    Image.fromarray(overlay).save(out_path)
    print(f"Prediction: {label}  ({conf:.3f})")
    print(f"Heatmap shape: {heatmap.shape}  range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"Saved overlay -> {out_path.resolve()}")
