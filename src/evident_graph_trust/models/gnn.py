"""Graph neural network models for trust assessment."""
from __future__ import annotations

import torch
from torch import nn

from .base import EvidenceProvider, ForwardOutput, GraphClassifier


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        out = adjacency @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_self = nn.Linear(in_features, out_features, bias=False)
        self.weight_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_self.weight)
        nn.init.xavier_uniform_(self.weight_neigh.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        agg = adjacency @ x
        out = self.weight_self(x) + self.weight_neigh(agg) + self.bias
        return out


class GCNClassifier(GraphClassifier):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.layer1(features, adjacency)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, adjacency)
        return x


class GraphSAGEClassifier(GraphClassifier):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_features, hidden_features)
        self.layer2 = GraphSAGELayer(hidden_features, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.layer1(features, adjacency)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, adjacency)
        return x


class EvidentialGNN(GraphClassifier, EvidenceProvider):
    """Two-layer GCN that outputs Dirichlet evidence."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.ModuleList(
            [
                GCNLayer(in_features, hidden_features),
                GCNLayer(hidden_features, hidden_features),
            ]
        )
        self.classifier = nn.Linear(hidden_features, num_classes)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = features
        for layer in self.backbone:
            x = layer(x, adjacency)
            x = self.activation(x)
            x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def predict_evidence(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features, adjacency)
        evidence = self.softplus(logits)
        return evidence

    def forward_with_evidence(self, features: torch.Tensor, adjacency: torch.Tensor) -> ForwardOutput:
        logits = self.forward(features, adjacency)
        evidence = self.softplus(logits)
        return ForwardOutput(logits=logits, evidence=evidence)
