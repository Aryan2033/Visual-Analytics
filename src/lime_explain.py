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
def _batch_predict(images_np: np.ndarray) -> np.ndarray:
    """
    images_np: (N, H, W, 3) uint8 or float [0..1] numpy array
    returns:   (N, num_classes) numpy array of probabilities
    """
    model = get_model()

    # Convert to a torch tensor of shape (N, 3, H, W) and normalize the same
    # way our training transforms did, otherwise the model gets confused.
    if images_np.dtype == np.uint8:
        images_np = images_np.astype(np.float32) / 255.0

    tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).to(DEVICE)
    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=DEVICE).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def compute_lime(image_path: Path, num_samples: int = 1000, num_features: int = 5) -> Tuple[np.ndarray, np.ndarray, str, float]:
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
    probs = _batch_predict(image_np[np.newaxis])[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # Run LIME
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=image_np,
        classifier_fn=_batch_predict,
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
    print("Running LIME (this takes ~10-30 seconds, it queries the model 1000 times)...")
    mask, overlay, label, conf = compute_lime(img_path)

    out_path = Path("lime_output.png")
    Image.fromarray(overlay).save(out_path)
    print(f"Prediction: {label}  ({conf:.3f})")
    print(f"Important superpixels: {int(mask.sum() > 0)} regions highlighted")
    print(f"Saved overlay -> {out_path.resolve()}")
