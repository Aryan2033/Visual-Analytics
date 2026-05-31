"""
PyTorch Dataset for MVTec AD.

MVTec is laid out like this:

    bottle/
        train/
            good/            <- only normal images, used for training
                000.png
                ...
        test/
            good/            <- normal test images
            broken_large/    <- defective: broken
            broken_small/
            contamination/
        ground_truth/
            broken_large/    <- B&W masks marking where the defect is
            broken_small/
            contamination/

For our classification model we only care about (image, label) pairs:
    label = 0  ->  normal     (folder name "good")
    label = 1  ->  anomalous  (any other subfolder under test/)

The ground-truth masks are NOT used during training.  We only load them
later, when we evaluate how well Grad-CAM and LIME localize the defect.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CATEGORIES, DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# Training: light augmentation so the model generalizes a bit.
# Validation/test: no augmentation, just resize + normalize.
def build_train_transform() -> Callable:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform() -> Callable:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MVTecDataset(Dataset):
    """
    Loads MVTec images for binary classification (normal vs anomalous).

    Args:
        split:       "train" or "test"
        categories:  list of MVTec category names, e.g. ["bottle", "hazelnut"]
        transform:   torchvision transform to apply to each image
        return_path: if True, __getitem__ also returns the file path
                     (useful for the dashboard so we can show the user
                      which file we just predicted on)
    """

    def __init__(
        self,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        assert split in {"train", "test"}, f"split must be 'train' or 'test', got {split!r}"
        self.split       = split
        self.categories  = categories or CATEGORIES
        self.transform   = transform or build_eval_transform()
        self.return_path = return_path

        # samples is a list of (image_path, label) tuples we will index into.
        self.samples: List[Tuple[Path, int]] = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found.  Looked for categories {self.categories} "
                f"in split '{split}' under {DATA_DIR}.  "
                f"Did you extract MVTec into the data/ folder?"
            )

    def _build_index(self) -> None:
        """Walks the folder tree and builds a list of (path, label) pairs."""
        for cat in self.categories:
            split_root = DATA_DIR / cat / self.split
            if not split_root.exists():
                raise FileNotFoundError(
                    f"Expected folder {split_root} does not exist.  "
                    f"Check that MVTec is extracted under data/{cat}/."
                )

            # Each subfolder under split_root is a defect type ("good", "broken_large", ...)
            for defect_dir in sorted(split_root.iterdir()):
                if not defect_dir.is_dir():
                    continue
                # "good" -> normal (label 0).  Anything else -> anomalous (label 1).
                label = 0 if defect_dir.name == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # MVTec has a couple of grayscale categories; force RGB so the
        # tensor shape is always (3, H, W).
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        if self.return_path:
            return image, label, str(path)
        return image, label


# ---------------------------------------------------------------------------
# Convenience function used by train.py
# ---------------------------------------------------------------------------
def build_datasets() -> Tuple[Dataset, Dataset]:
    """Returns (train_dataset, test_dataset) with the appropriate transforms."""
    train_ds = MVTecDataset(split="train", transform=build_train_transform())
    test_ds  = MVTecDataset(split="test",  transform=build_eval_transform())
    return train_ds, test_ds


if __name__ == "__main__":
    # Quick smoke test: run `python src/dataset.py` to verify the dataset loads.
    train_ds, test_ds = build_datasets()
    print(f"Train set: {len(train_ds)} images")
    print(f"Test set:  {len(test_ds)} images")

    # Show class distribution (sanity check: train should be 100% normal)
    train_labels = [lbl for _, lbl in train_ds.samples]
    test_labels  = [lbl for _, lbl in test_ds.samples]
    print(f"Train labels: normal={train_labels.count(0)}, anomalous={train_labels.count(1)}")
    print(f"Test labels:  normal={test_labels.count(0)}, anomalous={test_labels.count(1)}")

    # Load one image and print its tensor shape
    img, label = train_ds[0]
    print(f"First image tensor shape: {tuple(img.shape)}, label: {label}")
