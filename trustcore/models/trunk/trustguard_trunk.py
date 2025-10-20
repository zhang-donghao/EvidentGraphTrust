"""Simplified TrustGuard trunk placeholder.

This module provides a drop-in node-embedding trunk that mirrors the
interface of TrustGuard's spatial-temporal encoder. It should be replaced or
extended with the official TrustGuard implementation when integrating with the
full project. The implementation relies on PyTorch Geometric SAGEConv layers to
produce node representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency resolved at runtime
    from torch_geometric.nn import SAGEConv
except ImportError:  # pragma: no cover - depends on optional dependency
    SAGEConv = None


@dataclass
class TrustGuardTrunkConfig:
    in_dim: int
    hidden_dim: int = 128
    out_dim: int | None = None
    num_layers: int = 3
    dropout: float = 0.0


class TrustGuardTrunk(nn.Module):
    """Simplified spatial-temporal trunk producing node embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError(
                "torch_geometric is required for TrustGuardTrunk. Install PyG to use this trunk."
            )
        if num_layers < 1:
            raise ValueError("TrustGuardTrunk requires at least one layer")

        dims: List[int] = [in_dim]
        for _ in range(num_layers - 1):
            dims.append(hidden_dim)
        dims.append(out_dim or hidden_dim)

        self.dropout = dropout
        self.convs = nn.ModuleList()
        for idx in range(num_layers):
            in_channels = dims[idx]
            out_channels = dims[idx + 1]
            self.convs.append(SAGEConv(in_channels, out_channels))

    def forward(self, data) -> torch.Tensor:  # type: ignore[override]
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Input data must contain node features 'x'")
        if not hasattr(data, "edge_index"):
            raise ValueError("Input data must contain 'edge_index'")

        x = data.x
        edge_index = data.edge_index
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be of shape [2, E]")

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
