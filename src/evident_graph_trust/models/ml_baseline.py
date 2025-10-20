"""Traditional machine learning baselines."""
from __future__ import annotations

import torch
from torch import nn

from .base import GraphClassifier


class LogisticRegressionClassifier(GraphClassifier):
    """A simple logistic regression baseline that ignores graph structure."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.linear(features)
