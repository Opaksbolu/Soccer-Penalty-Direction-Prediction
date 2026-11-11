"""Single-frame MobileNetV3-Small penalty direction classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from src.data.common import CLASSES


class MobileNetSingleFrame(nn.Module):
    """Predict penalty direction from the last pre-contact frame."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes: int = len(CLASSES),
    ):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = mobilenet_v3_small(weights=weights)

        self.encoder = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )

        feature_size = backbone.classifier[0].in_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, num_classes),
        )

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep the frozen encoder in evaluation mode.
        self.encoder.eval()

        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 5:
            # Existing dataset format: [batch, time, channels, H, W].
            # Only the final frame is used.
            frames = frames[:, -1]

        elif frames.ndim != 4:
            raise ValueError(
                "Expected [B,T,C,H,W] or [B,C,H,W] input."
            )

        with torch.no_grad():
            features = self.encoder(frames)

        return self.head(features)
