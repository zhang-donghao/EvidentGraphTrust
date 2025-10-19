"""Baseline graph classifiers for comparison experiments."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Type

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


@dataclass
class BaselineConfig:
    """Configuration shared by baseline classifiers."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout: float = 0.3
    heads: int = 4
    dirichlet_strength: float = 1.0


class _GraphStack(nn.Module):
    """Stack of message-passing layers for baseline classifiers."""

    def __init__(
        self,
        config: BaselineConfig,
        conv_cls: Type[nn.Module],
        gat: bool = False,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        in_dim = config.input_dim
        supports_edge_weight = []
        for _ in range(config.num_layers):
            if gat:
                layer = conv_cls(
                    in_channels=in_dim,
                    out_channels=config.hidden_dim // config.heads,
                    heads=config.heads,
                    dropout=config.dropout,
                )
                in_dim = config.hidden_dim
            else:
                layer = conv_cls(in_dim, config.hidden_dim)
                in_dim = config.hidden_dim
            layers.append(layer)
            signature = inspect.signature(layer.forward)
            supports_edge_weight.append("edge_weight" in signature.parameters)
        self.layers = layers
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self._supports_edge_weight = supports_edge_weight

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x
        for layer, supports_edge_weight in zip(self.layers, self._supports_edge_weight):
            if isinstance(layer, GATConv):
                h = layer(h, edge_index)
            elif supports_edge_weight and edge_weight is not None:
                h = layer(h, edge_index, edge_weight)
            else:
                h = layer(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        return h


class GCNClassifier(nn.Module):
    """Standard GCN baseline producing class logits."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.backbone = _GraphStack(config, GCNConv)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        self.dirichlet_strength = config.dirichlet_strength

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        h = self.backbone(x, edge_index, edge_weight)
        logits = self.head(h)
        alpha = F.softplus(logits) + self.dirichlet_strength
        return {"logits": logits, "alpha": alpha}


class GATClassifier(nn.Module):
    """Graph attention baseline."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.backbone = _GraphStack(config, GATConv, gat=True)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        self.dirichlet_strength = config.dirichlet_strength

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, _: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.backbone(x, edge_index)
        logits = self.head(h)
        alpha = F.softplus(logits) + self.dirichlet_strength
        return {"logits": logits, "alpha": alpha}


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE baseline using mean aggregation."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.backbone = _GraphStack(config, SAGEConv)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        self.dirichlet_strength = config.dirichlet_strength

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        h = self.backbone(x, edge_index, edge_weight)
        logits = self.head(h)
        alpha = F.softplus(logits) + self.dirichlet_strength
        return {"logits": logits, "alpha": alpha}


class MLPClassifier(nn.Module):
    """Node-wise MLP ignoring graph structure."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for _ in range(max(1, config.num_layers)):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        self.dirichlet_strength = config.dirichlet_strength

    def forward(self, x: torch.Tensor, _: torch.Tensor, __: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        logits = self.head(h)
        alpha = F.softplus(logits) + self.dirichlet_strength
        return {"logits": logits, "alpha": alpha}


def build_baseline(model: str, config: BaselineConfig) -> nn.Module:
    """Factory helper returning the requested baseline classifier."""

    model = model.lower()
    if model == "gcn":
        return GCNClassifier(config)
    if model == "gat":
        return GATClassifier(config)
    if model == "graphsage":
        return GraphSAGEClassifier(config)
    if model == "mlp":
        return MLPClassifier(config)
    raise ValueError(f"Unsupported baseline model: {model}")
