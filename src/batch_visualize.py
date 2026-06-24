"""
Batch-generate combined visualizations for a set of test images.

For each image we save an image with three columns:
  - Original image
  - Ground-truth mask overlay (red)
  - Grad-CAM overlay (model explanation)

By default the script picks any misclassified images (FP/FN) and a couple
of correctly classified examples for inspection.

Usage:
    python src/batch_visualize.py

Outputs:
    outputs/visuals/<basename>_combined.png
"""
from pathlib import Path
import csv
from typing import List, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import IMAGE_SIZE
from gradcam_explain import compute_gradcam


OUT_DIR = Path("outputs") / "visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_ground_truth(image_path: Path) -> Optional[Path]:
    parts = image_path.parts
    try:
        idx = parts.index('data')
    except ValueError:
        return None
    try:
        cat = parts[idx + 1]
        defect = parts[idx + 3]
        name = parts[-1]
    except IndexError:
        return None
    mask_path = Path(*parts[: idx + 1]) / cat / 'ground_truth' / defect / name
    return mask_path if mask_path.exists() else None


def load_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(mask)
    return (arr > 0).astype(np.uint8)


def make_gt_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[..., 0] = mask * 255
    return (0.6 * rgb + 0.4 * mask_rgb).astype(np.uint8)


def create_combined(img_path: Path, out_path: Path) -> None:
    orig = load_image(img_path)
    gt_path = find_ground_truth(img_path)
    mask = load_mask(gt_path) if gt_path is not None else None

    heatmap, overlay, label, conf = compute_gradcam(img_path)

    if mask is not None:
        gt_overlay = make_gt_overlay(orig, mask)
    else:
        gt_overlay = np.zeros_like(orig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig.astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(gt_overlay.astype(np.uint8))
    axes[1].set_title('GT Mask')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Grad-CAM: {label} ({conf:.3f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def select_images(preds_csv: Path) -> List[Path]:
    # Read predictions.csv and pick misclassified images + a few correct ones
    rows = []
    with open(preds_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    mis = [Path(r['path']) for r in rows if r['label'] != r['pred']]
    correct_pos = [Path(r['path']) for r in rows if r['label'] == r['pred'] and r['label'] == '1']
    correct_neg = [Path(r['path']) for r in rows if r['label'] == r['pred'] and r['label'] == '0']

    selected = []
    # add misclassified first
    selected.extend(mis)
    # then add up to 2 positive and 2 negative correct examples
    selected.extend(correct_pos[:2])
    selected.extend(correct_neg[:2])
    # dedupe while preserving order
    seen = set()
    out = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    preds_csv = Path('outputs') / 'predictions.csv'
    if not preds_csv.exists():
        print('Predictions file not found. Run src/eval_classification.py first.')
        return

    images = select_images(preds_csv)
    if len(images) == 0:
        print('No images selected from predictions.csv')
        return

    for img_path in images:
        out_path = OUT_DIR / f"{img_path.stem}_combined.png"
        print('Rendering', img_path, '->', out_path)
        try:
            create_combined(img_path, out_path)
        except Exception as e:
            print('Failed for', img_path, ':', e)

    print('Saved visuals to', OUT_DIR)


if __name__ == '__main__':
    main()
