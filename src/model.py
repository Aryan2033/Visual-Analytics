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


class Simple2DCNN(nn.Module):
    """
    A custom 2D CNN built from scratch for anomaly detection on MVTec.
    Includes 4 blocks of Conv -> BatchNorm -> ReLU -> MaxPool,
    followed by Adaptive Average Pooling and fully connected layers.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


def build_cnn_model(num_classes: int = 2) -> nn.Module:
    """
    Builds the custom 2D CNN model.
    """
    return Simple2DCNN(num_classes=num_classes)


def save_checkpoint(model: nn.Module, path: Path, extra: Optional[dict] = None) -> None:
    """Saves model weights (plus optional metadata like epoch, accuracy)."""
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device, model_type: str = "resnet50", num_classes: int = 2) -> nn.Module:
    """
    Loads a saved model and puts it in eval mode, ready for inference.
    """
    if model_type == "cnn":
        model = build_cnn_model(num_classes=num_classes)
    else:
        model = build_model(num_classes=num_classes, pretrained=False)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick smoke test: build the models and pass a dummy image through them.
    print("Testing ResNet-50...")
    model_resnet = build_model()
    dummy = torch.randn(1, 3, 224, 224)
    out_resnet = model_resnet(dummy)
    print(f"ResNet output shape: {tuple(out_resnet.shape)} (expected (1, 2))")
    n_params_resnet = sum(p.numel() for p in model_resnet.parameters())
    print(f"Total ResNet parameters: {n_params_resnet:,}")

    print("\nTesting Custom 2D CNN...")
    model_cnn = build_cnn_model()
    out_cnn = model_cnn(dummy)
    print(f"CNN output shape: {tuple(out_cnn.shape)} (expected (1, 2))")
    n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"Total CNN parameters: {n_params_cnn:,}")

