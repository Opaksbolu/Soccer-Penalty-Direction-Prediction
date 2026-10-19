"""MobileNetV3-Small frame encoder followed by a compact GRU."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.data.common import CLASSES


class MobileNetGRU(nn.Module):
    def __init__(self, pretrained: bool = True, hidden_size: int = 128, dropout: float = 0.3,
                 num_classes: int = len(CLASSES)):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())
        feature_size = backbone.classifier[0].in_features
        self.gru = nn.GRU(feature_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Freeze batch-norm statistics as well as weights.
        self.encoder.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        with torch.no_grad():
            features = self.encoder(frames.reshape(batch * time, channels, height, width))
        features = features.reshape(batch, time, -1)
        _, hidden = self.gru(features)
        return self.head(hidden[-1])
