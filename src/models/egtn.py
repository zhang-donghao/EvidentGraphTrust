"""Evidential Graph Trust Network implementation.

This module defines a PyTorch Geometric compatible graph neural network that
produces Dirichlet evidence for trust-aware classification tasks. The design
follows the evidential GNN formulation described in the EGT research plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


@dataclass
class EGTNConfig:
    """Configuration container for Evidential Graph Trust Network."""

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 2
    num_layers: int = 2
    dropout: float = 0.3
    use_gat: bool = False
    heads: int = 4
    evidence_activation: str = "softplus"
    kl_strength: float = 1e-3


class EvidentialGraphTrustNetwork(nn.Module):
    """Graph neural network that outputs Dirichlet evidence.

    The model stacks multiple GCN/GAT layers and projects the resulting node
    embeddings to non-negative evidence values. The evidence is transformed
    into Dirichlet concentration parameters that can be consumed by evidential
    loss functions.
    """

    def __init__(self, config: EGTNConfig) -> None:
        super().__init__()
        self.config = config
        conv_cls = GATConv if config.use_gat else GCNConv

        self.layers = nn.ModuleList()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim

        for layer_idx in range(config.num_layers):
            out_dim = hidden_dim
            if config.use_gat:
                conv = conv_cls(
                    in_channels=input_dim,
                    out_channels=hidden_dim // config.heads,
                    heads=config.heads,
                    dropout=config.dropout,
                    add_self_loops=not isinstance(conv_cls, GATConv),
                )
                out_dim = hidden_dim
            else:
                conv = conv_cls(input_dim, hidden_dim)
            self.layers.append(conv)
            input_dim = out_dim

        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
        self.evidence_layer = nn.Linear(hidden_dim, config.output_dim)
        self._evidence_activation = self._build_activation(config.evidence_activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute Dirichlet evidence for each node.

        Args:
            x: Node feature matrix of shape ``[num_nodes, input_dim]``.
            edge_index: Graph connectivity in COO format ``[2, num_edges]``.
            edge_weight: Optional edge weights for weighted convolutions.

        Returns:
            Dictionary with keys ``logits``, ``evidence`` and ``alpha`` for
            downstream loss computation and evaluation.
        """

        h = x
        for conv in self.layers:
            if isinstance(conv, GATConv):
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index, edge_weight)
            h = self.activation(h)
            h = self.dropout(h)

        logits = self.evidence_layer(h)
        evidence = self._evidence_activation(logits)
        alpha = evidence + 1.0
        return {"logits": logits, "evidence": evidence, "alpha": alpha}

    @staticmethod
    def dirichlet_expected_prob(alpha: torch.Tensor) -> torch.Tensor:
        """Return expected class probabilities of the Dirichlet distribution."""

        return alpha / alpha.sum(dim=-1, keepdim=True)

    def regularization_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence between predicted Dirichlet and uniform prior."""

        k = alpha.size(-1)
        target_alpha = torch.ones_like(alpha)
        log_term = torch.lgamma(alpha.sum(-1)) - torch.lgamma(alpha).sum(-1)
        log_term -= torch.lgamma(target_alpha.sum(-1)) - torch.lgamma(target_alpha).sum(-1)
        digamma_term = ((alpha - target_alpha) * (torch.digamma(alpha) - torch.digamma(alpha.sum(-1, keepdim=True)))).sum(-1)
        kl = log_term + digamma_term
        return self.config.kl_strength * kl.mean()

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        if name.lower() == "softplus":
            return nn.Softplus()
        if name.lower() == "relu":
            return nn.ReLU()
        if name.lower() == "exp":
            return torch.exp  # type: ignore[return-value]
        raise ValueError(f"Unsupported evidence activation: {name}")
