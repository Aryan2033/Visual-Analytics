"""
The model: a ResNet-50 pretrained on ImageNet, with its final classifier
layer swapped for a 2-output head (normal vs anomalous).

We are doing TRANSFER LEARNING:
  - keep the convolutional layers (they already know how to see edges,
    textures, shapes; we do not want to relearn that from scratch).
  - replace only the final fully-connected layer, then fine-tune end-to-end
    on MVTec.

This is the standard approach when you have a small dataset.  Training a
ResNet-50 from scratch on ~400 images would massively overfit.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds a ResNet-50 ready for fine-tuning.

    Args:
        num_classes: how many output classes (2 for normal/anomalous).
        pretrained:  if True, load ImageNet-pretrained weights.

    Returns:
        A nn.Module you can move to a device and start training.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace the final fully-connected layer.
    # The original FC maps 2048 features -> 1000 ImageNet classes.
    # We swap it for 2048 features -> num_classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def save_checkpoint(model: nn.Module, path: Path, extra: Optional[dict] = None) -> None:
    """Saves model weights (plus optional metadata like epoch, accuracy)."""
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device, num_classes: int = 2) -> nn.Module:
    """
    Loads a saved model and puts it in eval mode, ready for inference.
    """
    model = build_model(num_classes=num_classes, pretrained=False)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick smoke test: build the model and pass a dummy image through it.
    model = build_model()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Model output shape: {tuple(out.shape)} (expected (1, 2))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
