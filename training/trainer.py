from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from trustcore.models.heads.evidential_edge_head import EvidentialEdgeHead
from trustcore.models.heads.mlp_edge_classifier import MLPEdgeClassifier
from trustcore.models.efg_components.losses import compose_loss
from training.metrics import (
    ece_metric,
    finalize_metrics,
    nll_metric,
    update_classification_metrics,
)


@dataclass
class EvidentialConfig:
    hidden: int = 128
    fusion: str = "none"
    lambda_disson: float = 0.1
    lambda_falseconf: float = 0.01


@dataclass
class TrainingConfig:
    lr: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stop: int = 8


@dataclass
class Config:
    model: str = "trustguard"
    head: str = "mlp"
    n_classes: int = 2
    embedding_dim: int = 64
    evidential: EvidentialConfig = EvidentialConfig()
    training: TrainingConfig = TrainingConfig()
    log_dir: str = "runs/default"


class Trainer:
    """Minimal trainer orchestrating trunk and head modules."""

    def __init__(self, cfg: Config, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def _build_head(self, in_dim: int, n_classes: int) -> nn.Module:
        if self.cfg.head == "evidential":
            return EvidentialEdgeHead(
                in_dim=in_dim,
                n_classes=n_classes,
                hidden=self.cfg.evidential.hidden,
                fusion=self.cfg.evidential.fusion,
            ).to(self.device)
        return MLPEdgeClassifier(in_dim=in_dim, n_classes=n_classes).to(self.device)

    def _write_metrics(self, epoch: int, split: str, metrics: Dict[str, Any]) -> None:
        csv_path = Path(self.cfg.log_dir) / "metrics.csv"
        file_exists = csv_path.exists()
        columns = ["MCC", "AUC", "BA", "F1", "NLL", "ECE", "uncertainty_mean", "loss"]
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["epoch", "split"] + columns
                writer.writerow(header)
            row = [epoch, split] + [metrics.get(col, "") for col in columns]
            writer.writerow(row)

    def train_one_epoch(
        self,
        trunk: nn.Module,
        head: nn.Module,
        loader: Iterable,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        trunk.train()
        head.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            h = trunk(batch)
            u, v = batch.edge_index
            h_u = h[u]
            h_v = h[v]
            y = batch.edge_attr.long().view(-1)

            optimizer.zero_grad()
            if self.cfg.head == "evidential":
                out = head(h_u, h_v, aux_alphas=None)
                alpha = out["alpha"]
                loss_dict = compose_loss(
                    alpha,
                    y,
                    lambda_disson=self.cfg.evidential.lambda_disson,
                    lambda_falseconf=self.cfg.evidential.lambda_falseconf,
                )
                loss = loss_dict["loss"]
            else:
                logits = head(h_u, h_v)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def evaluate(self, trunk: nn.Module, head: nn.Module, loader: Iterable):
        trunk.eval()
        head.eval()
        meter: Dict[str, Any] = {}
        for batch in loader:
            batch = batch.to(self.device)
            h = trunk(batch)
            u, v = batch.edge_index
            h_u = h[u]
            h_v = h[v]
            y = batch.edge_attr.long().view(-1)

            if self.cfg.head == "evidential":
                out = head(h_u, h_v, aux_alphas=None)
                alpha = out["alpha"]
                probs = out["probs"]
                pred = probs.argmax(dim=-1)
                uncert = out["uncertainty"].mean().item()
                update_classification_metrics(meter, pred, y, probs)
                nll_metric(meter, alpha, y)
                ece_metric(meter, probs, y)
                meter.setdefault("uncertainty_mean", []).append(uncert)
            else:
                logits = head(h_u, h_v)
                probs = logits.softmax(dim=-1)
                pred = probs.argmax(dim=-1)
                update_classification_metrics(meter, pred, y, probs)

        metrics = finalize_metrics(meter)
        if "uncertainty_mean" in meter:
            metrics["uncertainty_mean"] = sum(meter["uncertainty_mean"]) / len(
                meter["uncertainty_mean"]
            )
        return metrics


class IdentityTrunk(nn.Module):
    """Simple trunk returning node features as embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, batch) -> torch.Tensor:  # type: ignore[override]
        return self.project(batch.x)
