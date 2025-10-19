"""Utilities for constructing synthetic IoT communication graphs.

The goal of this module is to provide a small yet expressive benchmark that
captures the key traits of the "Evidential Graph Neural Networks for
Uncertainty-aware Node Classification" paper when applied to trust assessment in
IoT settings.  The generator mimics benign, suspicious and malicious device
behaviour and produces rich node features alongside a communication graph.

The design of the synthetic process loosely follows the TrustGuard repository
structure, but it intentionally remains lightweight so that experiments can be
executed in restricted environments without additional datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class IoTGraphData:
    """Container that bundles a generated IoT communication graph."""

    features: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    metadata: Dict[str, torch.Tensor]


def _class_block_matrix(num_classes: int) -> torch.Tensor:
    """Return a block matrix controlling inter-class edge densities."""

    base = torch.tensor(
        [
            [0.72, 0.18, 0.10],
            [0.18, 0.55, 0.27],
            [0.10, 0.27, 0.63],
        ],
        dtype=torch.float32,
    )
    if num_classes == 2:
        return base[:2, :2]
    return base


def _sample_class_sizes(num_nodes: int, class_probs: torch.Tensor) -> torch.Tensor:
    multinomial = torch.distributions.Multinomial(total_count=num_nodes, probs=class_probs)
    return multinomial.sample().long()


def generate_iot_graph(
    num_nodes: int = 420,
    feature_dim: int = 18,
    class_probs: Tuple[float, ...] = (0.6, 0.2, 0.2),
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    structure_enhancement: bool = True,
    seed: int = 13,
) -> IoTGraphData:
    """Generate an IoT communication graph with trust annotations.

    Args:
        num_nodes: Number of devices in the graph.
        feature_dim: Dimensionality of the device feature vector.
        class_probs: Class probabilities for trusted/suspicious/malicious devices.
        train_ratio: Fraction of labelled nodes allocated to the training set.
        val_ratio: Fraction allocated to validation (remainder used for testing).
        structure_enhancement: Whether to add two-hop connectivity prior to
            normalisation.  This mirrors the graph structure refinement strategy
            used in TrustGuard to stabilise message passing.
        seed: Reproducibility seed.

    Returns:
        An :class:`IoTGraphData` instance containing all tensors required for the
        experiments.
    """

    torch.manual_seed(seed)

    class_probs_tensor = torch.tensor(class_probs, dtype=torch.float32)
    num_classes = class_probs_tensor.numel()
    class_sizes = _sample_class_sizes(num_nodes, class_probs_tensor)
    labels = torch.empty(num_nodes, dtype=torch.long)

    start = 0
    for cls, size in enumerate(class_sizes.tolist()):
        end = start + size
        labels[start:end] = cls
        start = end

    perm = torch.randperm(num_nodes)
    labels = labels[perm]

    means = torch.stack(
        [
            torch.linspace(-1.2, 1.0, feature_dim),
            torch.linspace(0.4, -0.8, feature_dim),
            torch.linspace(1.1, -1.4, feature_dim),
        ]
    )[:num_classes]

    class_cov = torch.stack([torch.diag(torch.full((feature_dim,), 0.35)) for _ in range(num_classes)])

    features = torch.empty((num_nodes, feature_dim))
    for cls in range(num_classes):
        mask = labels == cls
        count = mask.sum().item()
        if count == 0:
            continue
        base = torch.distributions.MultivariateNormal(means[cls], class_cov[cls])
        sample = base.sample((count,))
        behaviour = torch.randn(count, feature_dim) * 0.15
        temporal_shift = torch.sin(torch.linspace(0, 3.14, count)).unsqueeze(1) * torch.linspace(0.1, 1.0, feature_dim)
        sample = sample + behaviour + temporal_shift
        features[mask] = sample

    features = torch.nn.functional.normalize(features, dim=1)

    block = _class_block_matrix(num_classes)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    rand = torch.rand((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = block[labels[i], labels[j]]
            if rand[i, j] < prob:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    adjacency.fill_diagonal_(0.0)

    if structure_enhancement:
        two_hop = torch.where((adjacency @ adjacency) > 0, torch.tensor(1.0), torch.tensor(0.0))
        adjacency = torch.clamp(adjacency + 0.15 * two_hop, max=1.0)
        adjacency.fill_diagonal_(0.0)

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    indices = torch.randperm(num_nodes)
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train : num_train + num_val]] = True
    test_mask[indices[num_train + num_val :]] = True

    metadata = {
        "class_probs": class_probs_tensor,
        "class_sizes": class_sizes,
        "structure_enhancement": torch.tensor(structure_enhancement),
    }

    return IoTGraphData(
        features=features,
        adjacency=adjacency,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        metadata=metadata,
    )
