from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class UserAttributionBranch(nn.Module):
    """One independent registered-user detector."""

    def __init__(
        self,
        channels: Iterable[int] = (32, 64, 128),
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = [int(value) for value in channels]
        if not channels or min(channels) <= 0:
            raise ValueError("invalid network dimensions")
        blocks = []
        source_channels = 1
        for target_channels in channels:
            groups = min(8, target_channels)
            while target_channels % groups:
                groups -= 1
            blocks.extend(
                [
                    nn.Conv2d(source_channels, target_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(groups, target_channels),
                    nn.GELU(),
                ]
            )
            source_channels = target_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(source_channels, int(embedding_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embedding_dim), 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(images)))


class MultiUserAttributionNetwork(nn.Module):
    """A jointly optimized collection of independent user detectors."""

    def __init__(
        self,
        user_count: int,
        channels: Iterable[int] = (32, 64, 128),
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if user_count < 2:
            raise ValueError("user_count must be at least two")
        self.detectors = nn.ModuleList(
            UserAttributionBranch(channels, embedding_dim, dropout)
            for _ in range(user_count)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.cat([detector(images) for detector in self.detectors], dim=1)


def build_model(config, user_count: int) -> MultiUserAttributionNetwork:
    return MultiUserAttributionNetwork(user_count=user_count, **config)
