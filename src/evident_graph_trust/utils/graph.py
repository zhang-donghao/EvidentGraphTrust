"""Graph utility functions."""
from __future__ import annotations

import torch


def add_self_loops(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.size(0), device=adjacency.device)
    return adjacency + identity


def normalize_adjacency(adjacency: torch.Tensor, add_loops: bool = True) -> torch.Tensor:
    """Symmetrically normalise an adjacency matrix."""

    if add_loops:
        adjacency = add_self_loops(adjacency)
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    d_mat = torch.diag(degree_inv_sqrt)
    return d_mat @ adjacency @ d_mat


def stochastic_normalize(adjacency: torch.Tensor, add_loops: bool = False) -> torch.Tensor:
    """Row-normalise the adjacency matrix."""

    if add_loops:
        adjacency = add_self_loops(adjacency)
    degree = adjacency.sum(dim=1, keepdim=True)
    degree[degree == 0.0] = 1.0
    return adjacency / degree


def enhance_two_hop(adjacency: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    """Inject two-hop connectivity as a prior on missing links."""

    two_hop = torch.where((adjacency @ adjacency) > 0, torch.tensor(1.0, device=adjacency.device), torch.tensor(0.0, device=adjacency.device))
    enhanced = torch.clamp(adjacency + strength * two_hop, max=1.0)
    enhanced.fill_diagonal_(0.0)
    return enhanced
