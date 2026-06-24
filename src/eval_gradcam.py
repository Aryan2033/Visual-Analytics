"""
Fast Grad-CAM localization evaluation.

Computes IoU between Grad-CAM (thresholded) and ground-truth masks for
defective test images. Saves CSV to `outputs/gradcam_iou.csv` and prints mean IoU.
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
    rows = []
    for img_tensor, label, path_str in ds:
        if label == 0:
            continue
        img_path = Path(path_str)
        gt_path = find_ground_truth(img_path)
        if gt_path is None:
            continue
        gt_mask = load_mask(gt_path)

        heatmap, _, _, _ = compute_gradcam(img_path)
        thr = np.percentile(heatmap, 80)
        cam_mask = (heatmap >= thr).astype(np.uint8)
        cam_iou = iou(cam_mask, gt_mask)

        rows.append({'image': str(img_path.relative_to(Path.cwd())), 'gt': str(gt_path.relative_to(Path.cwd())), 'gradcam_iou': '' if np.isnan(cam_iou) else f"{cam_iou:.4f}"})

    out_csv = OUT_DIR / 'gradcam_iou.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'gt', 'gradcam_iou'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    vals = [float(r['gradcam_iou']) for r in rows if r['gradcam_iou']]
    print(f"Evaluated {len(rows)} defective images with GT masks")
    if vals:
        print(f"Mean Grad-CAM IoU: {np.mean(vals):.4f}")
    else:
        print('No IoU values computed')
    print('Saved CSV ->', out_csv)


if __name__ == '__main__':
    main()
