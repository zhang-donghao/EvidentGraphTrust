"""Graph utility functions."""
from __future__ import annotations

import torch


def add_self_loops(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.size(0), device=adjacency.device)
    return adjacency + identity


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """Convert an ``edge_index`` representation to a dense adjacency matrix."""

    device = edge_index.device
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_weight is None:
        values = torch.ones(edge_index.size(1), device=device)
    else:
        values = edge_weight.to(device)
    adjacency[edge_index[0], edge_index[1]] = values
    if symmetric:
        adjacency[edge_index[1], edge_index[0]] = values
    adjacency.fill_diagonal_(0.0)
    return adjacency


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
