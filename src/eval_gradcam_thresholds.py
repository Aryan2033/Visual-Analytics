"""
Sweep Grad-CAM percentile thresholds and report mean IoU per threshold.

Saves `outputs/gradcam_iou_thresholds.csv`.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

from dataset import MVTecDataset
from gradcam_explain import compute_gradcam
from config import IMAGE_SIZE


OUT_DIR = Path.cwd() / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def find_ground_truth(image_path: Path) -> Path:
    parts = image_path.parts
    try:
        idx = parts.index('data')
    except ValueError:
        return None
    try:
        cat = parts[idx + 1]
        split = parts[idx + 2]
        defect = parts[idx + 3]
        name = parts[-1]
    except IndexError:
        return None
    base = Path(*parts[: idx + 1]) / cat / 'ground_truth' / defect
    mask_path1 = base / name
    if mask_path1.exists():
        return mask_path1
    stem = Path(name).stem
    suffix = Path(name).suffix
    mask_path2 = base / f"{stem}_mask{suffix}"
    if mask_path2.exists():
        return mask_path2
    return None


def load_mask(mask_path: Path) -> np.ndarray:
    m = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(m)
    return (arr > 0).astype(np.uint8)


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return float('nan')
    return float(inter) / float(union)


def main():
    ds = MVTecDataset(split='test', transform=None, return_path=True)
    # collect defective image paths with GT
    items = []
    for _, label, p in ds:
        if label == 1:
            pth = Path(p)
            gt = find_ground_truth(pth)
            if gt is not None:
                items.append((pth, gt))

    print(f"Found {len(items)} defect images with GT masks")

    percentiles = list(range(50, 96, 5))  # 50,55,...95
    results = []

    for perc in percentiles:
        ious = []
        for img_path, gt_path in items:
            gt_mask = load_mask(gt_path)
            heatmap, _, _, _ = compute_gradcam(img_path)
            thr = np.percentile(heatmap, perc)
            cam_mask = (heatmap >= thr).astype(np.uint8)
            val = iou(cam_mask, gt_mask)
            if not np.isnan(val):
                ious.append(val)
        mean_iou = float(np.mean(ious)) if ious else float('nan')
        results.append({'percentile': perc, 'mean_iou': f"{mean_iou:.4f}" if not np.isnan(mean_iou) else ''})
        print(f"Percentile {perc}: mean IoU = {mean_iou:.4f}")

    out_csv = OUT_DIR / 'gradcam_iou_thresholds.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['percentile', 'mean_iou'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print('Saved CSV ->', out_csv)


if __name__ == '__main__':
    main()
