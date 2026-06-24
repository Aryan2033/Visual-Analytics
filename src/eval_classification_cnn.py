"""
Evaluate the trained Custom 2D CNN classifier on the MVTec test split and report
classification metrics (accuracy, precision, recall, F1, AUC).

Usage:
    python src/eval_classification_cnn.py

Writes:
    outputs/predictions_cnn.csv             -- per-image predictions
    outputs/classification_metrics_cnn.txt  -- human-readable summary
"""
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, DEVICE, BATCH_SIZE, NUM_WORKERS
from dataset import MVTecDataset, build_eval_transform
from model import load_checkpoint

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def main():
    ckpt = CHECKPOINTS_DIR / "best_model_cnn.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}. Train CNN first.")

    model = load_checkpoint(ckpt, DEVICE, model_type="cnn")

    test_ds = MVTecDataset(split="test", transform=build_eval_transform(), return_path=True)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, labels, paths = batch
            imgs = imgs.to(DEVICE)
            labels = labels.int().tolist()

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()  # positive-class prob
            preds = (torch.tensor(probs) >= 0.5).long().tolist()

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_paths.extend(paths)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')

    # Save per-image predictions
    preds_csv = OUT_DIR / "predictions_cnn.csv"
    with open(preds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "pred", "prob_pos"])
        for p, l, pr, prob in zip(all_paths, all_labels, all_preds, all_probs):
            writer.writerow([p, l, pr, f"{prob:.4f}"])

    # Save human-readable metrics
    metrics_txt = OUT_DIR / "classification_metrics_cnn.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Samples: {len(all_labels)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))

    # Print concise summary
    print(f"CNN Evaluation Results:")
    print(f"  Samples: {len(all_labels)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"Saved CNN predictions -> {preds_csv}")
    print(f"Saved CNN metrics -> {metrics_txt}")


if __name__ == "__main__":
    main()
