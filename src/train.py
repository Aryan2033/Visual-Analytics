"""
Training script.

What this does in plain language:
  1. Loads the MVTec train and test sets.
  2. NOTE: MVTec only gives "good" images in train/.  All "broken/cracked"
     images live in test/.  So to teach the model what "anomalous" looks like
     we hold back some of the test set for training.  This is standard for
     MVTec when you treat it as a supervised classification problem.
  3. Fine-tunes ResNet-50 for a few epochs.
  4. Validates on the held-out portion of test/ after every epoch.
  5. Saves the best checkpoint to checkpoints/best_model.pt.

Run from the project root:
    python src/train.py
"""

import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
)
from dataset import build_datasets
from model import build_model, save_checkpoint


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Pin every random source so our results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Train/val split of the MVTec test set
# ---------------------------------------------------------------------------
def split_test_set(test_ds, val_fraction: float = 0.5):
    """
    MVTec puts all defective images in test/.  To train a classifier we need
    defective images during training too.  So we split the test set:
      - half becomes part of the training data (so the model sees defects)
      - half stays as a held-out validation set we never train on

    We stratify by label so both halves have a similar normal/anomalous mix.
    """
    rng = np.random.default_rng(SEED)
    indices_by_label = {0: [], 1: []}
    for idx, (_, lbl) in enumerate(test_ds.samples):
        indices_by_label[lbl].append(idx)

    train_idx, val_idx = [], []
    for lbl, idxs in indices_by_label.items():
        idxs = rng.permutation(idxs)
        n_val = int(len(idxs) * val_fraction)
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_n       += images.size(0)

    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# One epoch of validation (no gradients, no weight updates)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_n       += images.size(0)

    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    print(f"Training on device: {DEVICE}")

    # 1. Datasets ----------------------------------------------------------
    train_pure_ds, test_full_ds = build_datasets()
    train_extra_idx, val_idx = split_test_set(test_full_ds, val_fraction=0.5)

    # The model trains on:
    #   - all the "good" images from MVTec's train/ folder
    #   - half of the test set (a mix of good + defective)
    # IMPORTANT: train_pure_ds uses the augmenting transform, while
    # test_full_ds uses the plain eval transform.  We accept this small
    # inconsistency for simplicity: the defective-during-training images
    # do not get the train-time augmentation.  That is fine for a fine-tune.
    train_subset_from_test = Subset(test_full_ds, train_extra_idx)
    train_dataset = torch.utils.data.ConcatDataset([train_pure_ds, train_subset_from_test])
    val_dataset   = Subset(test_full_ds, val_idx)

    # Class imbalance: MVTec train/ is 100% normal, so even after mixing in
    # half the test set our training data is heavily skewed.  A weighted
    # sampler oversamples the rare class so the model does not just learn
    # "predict normal every time".
    train_labels = []
    for ds in train_dataset.datasets:
        if isinstance(ds, Subset):
            train_labels.extend([ds.dataset.samples[i][1] for i in ds.indices])
        else:
            train_labels.extend([lbl for _, lbl in ds.samples])
    label_counts = Counter(train_labels)
    class_weights = {lbl: 1.0 / cnt for lbl, cnt in label_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    print(f"Train size: {len(train_dataset)} (label counts: {dict(label_counts)})")
    print(f"Val   size: {len(val_dataset)}")

    # 2. Model + optimizer + loss -----------------------------------------
    model     = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop ----------------------------------------------------
    best_val_acc = 0.0
    ckpt_path = CHECKPOINTS_DIR / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, ckpt_path, extra={"epoch": epoch, "val_acc": val_acc})
            print(f"  ↳ new best, saved to {ckpt_path}")

    print(f"\nDone.  Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
