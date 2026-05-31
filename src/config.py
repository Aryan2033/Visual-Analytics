"""
Central configuration. Everything that you might want to tweak lives here.
Importing from one place means we never have to hunt through files to change
a hyperparameter or fix a path.
"""

from pathlib import Path
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder containing src/, data/, checkpoints/, etc.
# We compute it from this file's location so it works no matter where you
# run the scripts from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Which MVTec categories to use
# ---------------------------------------------------------------------------
# Start with "bottle" since that is what we already have.  Adding more later
# is just a matter of appending to this list.
CATEGORIES = ["bottle"]


# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 224          # ResNet-50 expects 224x224 input
BATCH_SIZE   = 16           # safe on an M4 Air with 16 GB unified memory
NUM_EPOCHS   = 15           # 15 is plenty for fine-tuning, training stops early if it plateaus
LEARNING_RATE = 1e-4        # small LR because we are fine-tuning, not training from scratch
NUM_WORKERS  = 2            # parallel data loading; keep low on a laptop
SEED         = 42           # reproducibility


# ---------------------------------------------------------------------------
# Device selection (CPU / CUDA / Apple Silicon MPS)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Picks the fastest device available.

    On an Apple Silicon Mac (M1/M2/M3/M4) this returns "mps", which uses
    the integrated GPU.  On a CUDA machine it returns "cuda".  Otherwise
    it falls back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------
# ResNet-50 was pretrained on ImageNet, which used these mean/std values.
# Our inputs must be normalized the same way or the model gets confused.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
