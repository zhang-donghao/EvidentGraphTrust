"""Utility functions for trust modeling metrics."""
from __future__ import annotations

from typing import Dict

import torch


def dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """Compute total uncertainty (Dirichlet strength reciprocal).

    Args:
        alpha: Dirichlet concentration parameters ``[batch, num_classes]``.

    Returns:
        Tensor of uncertainty values in ``[0, 1]`` where higher means more
        uncertainty.
    """

    strength = alpha.sum(dim=-1, keepdim=True)
    num_classes = alpha.size(-1)
    return num_classes / strength


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Estimate expected calibration error (ECE).

    Args:
        probs: Model confidence scores ``[batch, num_classes]``.
        labels: Ground-truth labels ``[batch]``.
        n_bins: Number of bins for calibration histogram.
    """

    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = confidences.gt(lower) & confidences.le(upper)
        if mask.sum() == 0:
            continue
        bucket_acc = accuracies[mask].float().mean()
        bucket_conf = confidences[mask].mean()
        weight = mask.float().mean()
        ece += (bucket_conf - bucket_acc).abs() * weight
    return ece.squeeze()


def trust_evaluation(alpha: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute calibration-aware metrics from Dirichlet parameters."""

    probs = alpha / alpha.sum(dim=-1, keepdim=True)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).mean()
    return {
        "ece": expected_calibration_error(probs, labels),
        "brier": ((probs - torch.nn.functional.one_hot(labels, probs.size(-1)).float()) ** 2).sum(dim=-1).mean(),
        "entropy": entropy,
        "uncertainty": dirichlet_uncertainty(alpha).mean(),
    }
