"""Losses and utilities for evidential learning."""
from __future__ import annotations

import torch
from torch import nn


def dirichlet_kl_divergence(alpha: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_prior = prior.sum(dim=-1, keepdim=True)
    lnB = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    lnB_prior = torch.lgamma(sum_prior) - torch.lgamma(prior).sum(dim=-1, keepdim=True)
    term1 = (alpha - prior) * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    return (lnB - lnB_prior + term1.sum(dim=-1, keepdim=True)).squeeze(-1)


def evidential_cross_entropy(
    alpha: torch.Tensor,
    target: torch.Tensor,
    kl_weight: float,
    epoch: int,
    num_epochs: int,
    prior_strength: float = 1.0,
) -> torch.Tensor:
    """Evidential cross-entropy loss with annealed KL term."""

    s = alpha.sum(dim=-1, keepdim=True)
    loglikelihood = torch.sum(target * (torch.digamma(s) - torch.digamma(alpha)), dim=-1)
    annealing = min(1.0, epoch / float(max(1, num_epochs)))
    prior = torch.full_like(alpha, prior_strength)
    kl = dirichlet_kl_divergence(alpha, prior)
    return loglikelihood + kl_weight * annealing * kl


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Compute the Expected Calibration Error (ECE)."""

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lower) * (confidences <= upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.squeeze()


def dirichlet_expected_probability(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True)


def dirichlet_total_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.size(-1)
    s = alpha.sum(dim=-1)
    return num_classes / s
