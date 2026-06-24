"""
Visualize: original image, ground-truth mask, and model explanation overlay (Grad-CAM).

Usage:
    python src/visualize_results.py <path-to-image>

Saves `combined_result.png` in the project root.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import IMAGE_SIZE, DATA_DIR
from gradcam_explain import compute_gradcam


def find_ground_truth(image_path: Path) -> Optional[Path]:
    """Given data/<cat>/test/<defect>/<name>.png, return the ground-truth mask path
    data/<cat>/ground_truth/<defect>/<name>
    """
    parts = image_path.parts
    # find the index of 'data' in the path
    try:
        idx = parts.index('data')
    except ValueError:
        return None

    # expected: data/<cat>/test/<defect>/<name>
    try:
        cat = parts[idx + 1]
        split = parts[idx + 2]
        defect = parts[idx + 3]
        name = parts[-1]
    except IndexError:
        return None

    mask_path = Path(*parts[: idx + 1]) / cat / 'ground_truth' / defect / name
    if mask_path.exists():
        return mask_path
    return None


def load_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img)


def load_mask(mask_path: Path) -> np.ndarray:
    # masks are usually single-channel; load and resize to IMAGE_SIZE
    mask = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(mask)
    # normalize mask to 0/1
    arr = (arr > 0).astype(np.uint8)
    return arr


def make_overlay_on_image(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    # heatmap assumed in [0,1], rgb in [0,255]
    from pytorch_grad_cam.utils.image import show_cam_on_image
    rgb_float = rgb.astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    return overlay


def main():
    if len(sys.argv) != 2:
        print('Usage: python src/visualize_results.py <path-to-image>')
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print('Image not found:', img_path)
        sys.exit(1)

    # original image
    orig = load_image(img_path)

    # ground truth mask if available
    gt_path = find_ground_truth(img_path)
    mask = None
    if gt_path is not None:
        mask = load_mask(gt_path)

    # Grad-CAM overlay (heatmap + overlay image)
    heatmap, overlay, label, conf = compute_gradcam(img_path)

    # If ground truth exists, create an overlay of mask on original (red)
    if mask is not None:
        mask_rgb = np.zeros_like(orig)
        mask_rgb[..., 0] = mask * 255  # red channel
        gt_overlay = (0.6 * orig + 0.4 * mask_rgb).astype(np.uint8)
    else:
        gt_overlay = np.zeros_like(orig)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig.astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(gt_overlay.astype(np.uint8))
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Grad-CAM: {label} ({conf:.3f})')
    axes[2].axis('off')

    out_path = Path.cwd() / 'combined_result.png'
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved combined visualization ->', out_path)


if __name__ == '__main__':
    main()
