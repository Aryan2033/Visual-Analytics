"""
LIME explanation for images.

How LIME works in one paragraph:
  1. Split the image into ~50 "superpixels" (perceptually similar regions).
  2. Create N=1000 perturbed versions of the image where random superpixels
     are hidden (replaced with gray).
  3. Run all 1000 through the model and collect their predictions.
  4. Fit a simple linear model: which superpixels, when visible, push the
     prediction toward "anomalous"?
  5. Highlight the top-k positive superpixels as the explanation.

This is conceptually very different from Grad-CAM (which peeks inside the
model).  LIME treats the model as a black box and probes it from outside.

Usage:
    python src/lime_explain.py path/to/image.png
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

from config import DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from predict import CLASS_NAMES, get_model


# LIME calls our model many times, so we need a "batch predict" function
# that takes a numpy batch of images and returns class probabilities.
#
# IMPORTANT: We create a CPU copy of the model rather than moving the
# cached model between MPS and CPU, which causes segfaults on macOS
# Apple Silicon due to MPS memory corruption.
import copy

_LIME_CPU_MODELS = {}

def _get_cpu_model(model_type: str):
    """Return a CPU-only copy of the model for LIME (avoids MPS segfault)."""
    global _LIME_CPU_MODELS
    if model_type not in _LIME_CPU_MODELS:
        original = get_model(model_type)
        cpu_copy = copy.deepcopy(original).to("cpu")
        cpu_copy.eval()
        _LIME_CPU_MODELS[model_type] = cpu_copy
    return _LIME_CPU_MODELS[model_type]


def _batch_predict(images_np: np.ndarray, model_type: str = "resnet50") -> np.ndarray:
    """
    images_np: (N, H, W, 3) uint8 or float [0..1] numpy array
    returns:   (N, num_classes) numpy array of probabilities
    """
    model = _get_cpu_model(model_type)

    # Convert to a torch tensor of shape (N, 3, H, W) and normalize the same
    # way our training transforms did, otherwise the model gets confused.
    if images_np.dtype == np.uint8:
        images_np = images_np.astype(np.float32) / 255.0

    tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).to("cpu")
    mean = torch.tensor(IMAGENET_MEAN, device="cpu").view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device="cpu").view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
    return probs.numpy()


def compute_lime(image_path: Path, num_samples: int = 1000, num_features: int = 5, model_type: str = "resnet50") -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Returns:
        explanation_mask: (224, 224) int array, nonzero where LIME found important superpixels
        overlay:          (224, 224, 3) uint8 image with superpixel boundaries drawn
        predicted_class:  "normal" or "anomalous"
        confidence:       float in [0, 1]
    """
    pil_image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np  = np.array(pil_image)

    # Predict once to get the label that LIME should explain
    probs = _batch_predict(image_np[np.newaxis], model_type=model_type)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # Run LIME
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=image_np,
        classifier_fn=lambda imgs: _batch_predict(imgs, model_type=model_type),
        top_labels=2,
        hide_color=0,          # hide superpixels by painting them black
        num_samples=num_samples,
        random_seed=42,
    )

    # `get_image_and_mask` returns the image with non-explanation pixels grayed out,
    # and a mask marking the top `num_features` superpixels.
    _, mask = explanation.get_image_and_mask(
        label=pred_idx,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )

    # Draw superpixel boundaries on the original image for a nicer visualization
    overlay = mark_boundaries(image_np / 255.0, mask, color=(1, 1, 0), mode="thick")
    overlay = (overlay * 255).astype(np.uint8)

    return mask, overlay, CLASS_NAMES[pred_idx], confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/lime_explain.py <path-to-image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    
    print("Running LIME for ResNet-50 (takes ~10-30 seconds)...")
    try:
        mask, overlay, label, conf = compute_lime(img_path, model_type="resnet50")
        out_path = Path("lime_output.png")
        Image.fromarray(overlay).save(out_path)
        print(f"  Prediction: {label} ({conf:.3f})")
        print(f"  Saved overlay -> {out_path.resolve()}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nRunning LIME for Custom 2D CNN (takes ~10-30 seconds)...")
    try:
        mask_cnn, overlay_cnn, label_cnn, conf_cnn = compute_lime(img_path, model_type="cnn")
        out_path_cnn = Path("lime_output_cnn.png")
        Image.fromarray(overlay_cnn).save(out_path_cnn)
        print(f"  Prediction: {label_cnn} ({conf_cnn:.3f})")
        print(f"  Saved overlay -> {out_path_cnn.resolve()}")
    except Exception as e:
        print(f"  Error: {e}")

