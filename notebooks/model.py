# ============================================================================
# VERIFY: image ↔ mask pairing, alignment, and prediction accuracy
# ============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))   # in a notebook
# sys.path.insert(0, str(Path('src').resolve()))     # in a .py file at project root

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from config import DATA_DIR, CATEGORIES, IMAGE_SIZE, DEVICE, CHECKPOINTS_DIR
from model import load_checkpoint
from dataset import build_eval_transform

CATEGORY = CATEGORIES[0]
cat_path = DATA_DIR / CATEGORY

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
ckpt = CHECKPOINTS_DIR / 'best_model.pt'
assert ckpt.exists(), 'Train the model first: python src/train.py'
model = load_checkpoint(ckpt, DEVICE)
transform = build_eval_transform()
CLASS_NAMES = ['normal', 'anomalous']


def predict(image_path):
    """Returns (predicted_label, confidence)."""
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax().item())
    return CLASS_NAMES[idx], float(probs[idx].item())


def find_mask(image_path):
    """Find the matching ground-truth mask for an image, or None if normal."""
    defect_type = image_path.parent.name
    if defect_type == 'good':
        return None
    mask_path = cat_path / 'ground_truth' / defect_type / f'{image_path.stem}_mask.png'
    return mask_path if mask_path.exists() else None


# ---------------------------------------------------------------------------
# Pick N random test images, check everything
# ---------------------------------------------------------------------------
N = 6   # change to taste

all_test_images = sorted((cat_path / 'test').rglob('*.png'))
rng = np.random.default_rng(0)
sample = rng.choice(all_test_images, size=N, replace=False)

fig, axes = plt.subplots(N, 3, figsize=(11, 3.2 * N))
if N == 1:
    axes = axes[np.newaxis, :]

print(f'{"file":<35}{"true":<12}{"predicted":<14}{"conf":<7}{"mask?":<7}{"ok"}')
print('-' * 85)

correct = 0
for row, img_path in enumerate(sample):
    img_path = Path(img_path)
    true_label = 'anomalous' if img_path.parent.name != 'good' else 'normal'
    pred_label, conf = predict(img_path)
    mask_path = find_mask(img_path)

    is_correct = (true_label == pred_label)
    correct += is_correct
    flag = '✅' if is_correct else '❌'

    # Mask sanity: if anomalous, mask must exist; if normal, mask must NOT exist
    if true_label == 'anomalous':
        mask_ok = mask_path is not None and mask_path.exists()
    else:
        mask_ok = mask_path is None
    mask_flag = '✓' if mask_ok else '✗ MISSING'

    short_name = f'{img_path.parent.name}/{img_path.name}'
    print(f'{short_name:<35}{true_label:<12}{pred_label:<14}{conf:.3f}  {mask_flag:<7}{flag}')

    # --- Plot ---
    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img)

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title(f'image\n{img_path.parent.name}/{img_path.name}', fontsize=9)
    axes[row, 0].axis('off')

    if mask_path is not None and mask_path.exists():
        mask = np.array(Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title(f'mask: {mask_path.name}', fontsize=9)

        # Overlay to verify pixel alignment
        red = np.zeros_like(img_np); red[..., 0] = 255
        alpha = (mask > 127).astype(np.float32)[..., np.newaxis] * 0.5
        overlay = (img_np * (1 - alpha) + red * alpha).astype(np.uint8)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f'overlay\n(red=defect)', fontsize=9)
    else:
        axes[row, 1].text(0.5, 0.5, 'no mask\n(normal image)',
                          ha='center', va='center', transform=axes[row, 1].transAxes,
                          fontsize=11, color='gray')
        axes[row, 2].text(0.5, 0.5, 'n/a',
                          ha='center', va='center', transform=axes[row, 2].transAxes,
                          fontsize=11, color='gray')

    # Color the title border by correctness
    title_color = '#2E7D32' if is_correct else '#C62828'
    axes[row, 0].set_title(
        f'image\ntrue={true_label} | pred={pred_label} ({conf:.2f})',
        fontsize=9, color=title_color, fontweight='bold'
    )

    for ax in axes[row]:
        ax.axis('off')

print('-' * 85)
print(f'Correct: {correct}/{N}  ({100*correct/N:.1f}%)')

plt.tight_layout()
plt.savefig(Path.cwd().parent / 'figures' / '09_verification.png', dpi=180, bbox_inches='tight')
plt.show()