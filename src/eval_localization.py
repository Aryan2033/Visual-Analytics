"""
Evaluate localization quality of explanations (Grad-CAM and LIME) against
ground-truth masks using IoU.

Saves results to `outputs/localization_iou.csv` and prints mean IoU.
"""
from pathlib import Path
import csv
import numpy as np

from dataset import MVTecDataset
from gradcam_explain import compute_gradcam
from lime_explain import compute_lime
from config import DATA_DIR, IMAGE_SIZE
from PIL import Image


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
    # try _mask suffix (common in MVTec)
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
    # both binary uint8 arrays
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return float('nan')
    return float(inter) / float(union)


def main():
    # Evaluate only defective test images (label == 1)
    ds = MVTecDataset(split='test', transform=None, return_path=True)

    rows = []
    for img_tensor, label, path_str in ds:
        img_path = Path(path_str)
        if label == 0:
            # skip normal images; ground-truth masks only exist for defects
            continue

        gt_path = find_ground_truth(img_path)
        if gt_path is None:
            continue
        gt_mask = load_mask(gt_path)

        # Grad-CAM
        heatmap, _, _, _ = compute_gradcam(img_path)
        # threshold heatmap at 80th percentile
        thr = np.percentile(heatmap, 80)
        cam_mask = (heatmap >= thr).astype(np.uint8)
        cam_iou = iou(cam_mask, gt_mask)

        # LIME
        lime_mask, _, _, _ = compute_lime(img_path)
        # lime_mask may be int with superpixel labels; binarize
        lime_bin = (lime_mask > 0).astype(np.uint8)
        lime_iou = iou(lime_bin, gt_mask)

        rows.append({
            'image': str(img_path.relative_to(Path.cwd())),
            'gt': str(gt_path.relative_to(Path.cwd())),
            'gradcam_iou': '' if np.isnan(cam_iou) else f"{cam_iou:.4f}",
            'lime_iou': '' if np.isnan(lime_iou) else f"{lime_iou:.4f}",
        })

    # Save CSV
    out_csv = OUT_DIR / 'localization_iou.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'gt', 'gradcam_iou', 'lime_iou'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Print summary
    cam_vals = [float(r['gradcam_iou']) for r in rows if r['gradcam_iou']]
    lime_vals = [float(r['lime_iou']) for r in rows if r['lime_iou']]
    print(f"Evaluated {len(rows)} defect images with ground-truth masks")
    if cam_vals:
        print(f"Mean Grad-CAM IoU: {np.mean(cam_vals):.4f}")
    else:
        print("No Grad-CAM IoU values computed")
    if lime_vals:
        print(f"Mean LIME IoU: {np.mean(lime_vals):.4f}")
    else:
        print("No LIME IoU values computed")

    print('Saved CSV ->', out_csv)


if __name__ == '__main__':
    main()
