"""
Plot training and validation loss/accuracy from the last run's epoch logs.

This script hardcodes the epoch metrics captured during the recent training
run (15 epochs) and saves `training_loss.png` and `training_accuracy.png`
to the project root.
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent

# Metrics captured from the training run
epochs = list(range(1, 16))
train_loss = [0.3737, 0.0789, 0.0248, 0.0215, 0.0116, 0.0071, 0.0042, 0.0111, 0.0046, 0.0029, 0.0036, 0.0025, 0.0013, 0.0019, 0.0016]
train_acc  = [0.904, 0.988, 0.992, 0.996, 0.996, 1.000, 1.000, 0.996, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
val_loss   = [0.5087, 0.2671, 0.2056, 0.2425, 0.1704, 0.2445, 0.2177, 0.2476, 0.4067, 0.3220, 0.2519, 0.3601, 0.3240, 0.2315, 0.3403]
val_acc    = [0.756, 0.902, 0.951, 0.902, 0.951, 0.902, 0.927, 0.902, 0.854, 0.854, 0.927, 0.854, 0.854, 0.927, 0.854]


def plot_loss(out_dir: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, marker='o', label='train_loss')
    plt.plot(epochs, val_loss, marker='o', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    path = out_dir / 'training_loss.png'
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_accuracy(out_dir: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, marker='o', label='train_acc')
    plt.plot(epochs, val_acc, marker='o', label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.legend()
    path = out_dir / 'training_accuracy.png'
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


if __name__ == '__main__':
    loss_path = plot_loss(OUTPUT_DIR)
    acc_path = plot_accuracy(OUTPUT_DIR)
    print('Saved:', loss_path)
    print('Saved:', acc_path)
