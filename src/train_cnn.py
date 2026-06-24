"""
Training script for the Custom 2D CNN.

What this does:
  1. Loads the MVTec train and test sets.
  2. Splits the test set into a 50/50 split (mixes half into train, reserves half for val).
  3. Trains the custom 2D CNN from scratch for 30 epochs with a learning rate of 1e-3.
  4. Validates on the held-out portion of test/ after every epoch.
  5. Saves the best checkpoint to checkpoints/best_model_cnn.pt.
"""

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    NUM_WORKERS,
    SEED,
)
from dataset import build_datasets
from model import build_cnn_model, save_checkpoint
from train import set_seed, split_test_set, train_one_epoch, evaluate

# Hyperparameters for training from scratch
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3


def main() -> None:
    set_seed(SEED)
    print(f"Training Custom 2D CNN on device: {DEVICE}")

    # 1. Datasets
    train_pure_ds, test_full_ds = build_datasets()
    train_extra_idx, val_idx = split_test_set(test_full_ds, val_fraction=0.5)

    train_subset_from_test = Subset(test_full_ds, train_extra_idx)
    train_dataset = torch.utils.data.ConcatDataset([train_pure_ds, train_subset_from_test])
    val_dataset   = Subset(test_full_ds, val_idx)

    # Rebalance classes
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

    # 2. Model + optimizer + loss
    model     = build_cnn_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop
    best_val_acc = 0.0
    ckpt_path = CHECKPOINTS_DIR / "best_model_cnn.pt"

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

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
