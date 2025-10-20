"""Base classes for trust-aware GNN models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class GraphClassifier(nn.Module):
    """Common interface for node classification models used in the project."""

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class EvidenceProvider(Protocol):
    """Protocol describing models that output evidential parameters."""

    def predict_evidence(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    evidence: torch.Tensor | None = None
