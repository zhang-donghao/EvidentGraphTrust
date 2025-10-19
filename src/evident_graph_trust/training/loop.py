"""Training utilities for graph-based trust prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from ..models.base import EvidenceProvider, GraphClassifier
from ..models.gnn import GraphSAGEClassifier
from ..utils.graph import enhance_two_hop, normalize_adjacency, stochastic_normalize
from .evidence import (
    dirichlet_expected_probability,
    dirichlet_total_uncertainty,
    evidential_cross_entropy,
    expected_calibration_error,
)


@dataclass
class TrainingConfig:
    epochs: int = 180
    lr: float = 5e-3
    weight_decay: float = 5e-4
    kl_weight: float = 0.8
    use_evidence_loss: bool = True
    use_graph_enhancement: bool = True
    annealing_epochs: int = 50
    device: str = "cpu"


@dataclass
class TrainingResult:
    model: GraphClassifier
    history: List[Dict[str, float]]
    metrics: Dict[str, float]
    trust_scores: torch.Tensor
    uncertainties: torch.Tensor


def _prepare_adjacency(adjacency: torch.Tensor, model: GraphClassifier, use_graph_enhancement: bool) -> torch.Tensor:
    if use_graph_enhancement:
        adjacency = enhance_two_hop(adjacency)
    if isinstance(model, GraphSAGEClassifier):
        return stochastic_normalize(adjacency, add_loops=True)
    return normalize_adjacency(adjacency, add_loops=True)


def _to_device(tensors: Tuple[torch.Tensor, ...], device: str) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in tensors)


def train_model(
    model: GraphClassifier,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    config: TrainingConfig,
) -> TrainingResult:
    """Train a graph classifier and report evaluation metrics."""

    device = torch.device(config.device)
    features, adjacency, labels, train_mask, val_mask, test_mask = _to_device(
        (features, adjacency, labels, train_mask, val_mask, test_mask), device
    )
    model = model.to(device)

    adjacency_norm = _prepare_adjacency(adjacency, model, config.use_graph_enhancement)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: List[Dict[str, float]] = []
    num_classes = labels.max().item() + 1

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency_norm)
        if isinstance(model, EvidenceProvider) and config.use_evidence_loss:
            evidence = torch.nn.functional.softplus(logits)
            alpha = evidence + 1.0
            target = F.one_hot(labels, num_classes=num_classes).float()
            loss = evidential_cross_entropy(
                alpha[train_mask],
                target[train_mask],
                kl_weight=config.kl_weight,
                epoch=epoch,
                num_epochs=config.annealing_epochs,
            ).mean()
        else:
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            logits_eval = model(features, adjacency_norm)
            probs = F.softmax(logits_eval, dim=-1)
            train_acc = (probs[train_mask].argmax(dim=-1) == labels[train_mask]).float().mean().item()
            val_acc = (probs[val_mask].argmax(dim=-1) == labels[val_mask]).float().mean().item()
            history.append({"epoch": epoch, "loss": float(loss.item()), "train_acc": train_acc, "val_acc": val_acc})

    with torch.no_grad():
        model.eval()
        logits = model(features, adjacency_norm)
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        accuracy = (predictions[test_mask] == labels[test_mask]).float().mean().item()
        nll = F.nll_loss(probs.log()[test_mask], labels[test_mask]).item()
        ece = expected_calibration_error(probs[test_mask], labels[test_mask]).item()

        if isinstance(model, EvidenceProvider):
            evidence = torch.nn.functional.softplus(logits)
            alpha = evidence + 1.0
            trust_scores = dirichlet_expected_probability(alpha)
            uncertainties = dirichlet_total_uncertainty(alpha)
        else:
            trust_scores = probs
            uncertainties = (-torch.sum(probs * torch.log(probs + 1e-8), dim=-1)).exp()

    metrics = {
        "test_accuracy": accuracy,
        "test_nll": nll,
        "test_ece": ece,
    }

    return TrainingResult(
        model=model.cpu(),
        history=history,
        metrics=metrics,
        trust_scores=trust_scores.cpu(),
        uncertainties=uncertainties.cpu(),
    )
