from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
class TrunkConfig:
    target: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    builder: str = ""
    root: str = "data"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    model: str = "trustguard"
    head: str = "mlp"
    n_classes: int = 2
    embedding_dim: int = 64
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evidential: EvidentialConfig = field(default_factory=EvidentialConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
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
        columns = [
            "MCC",
            "AUC",
            "BA",
            "F1",
            "NLL",
            "ECE",
            "uncertainty_mean",
            "loss",
            "L_ece",
            "L_disson",
            "L_falseconf",
        ]
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
    ) -> Dict[str, float]:
        trunk.train()
        head.train()
        total_loss = 0.0
        aux_metrics: Dict[str, List[float]] = {}
        num_batches = 0
        for batch in loader:
            batch = self._prepare_batch(batch)
            h = trunk(batch)
            u, v = self._edge_index(batch)
            h_u = h[u]
            h_v = h[v]
            y = self._edge_labels(batch)

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
                for key, value in loss_dict.items():
                    if key == "loss":
                        continue
                    if isinstance(value, torch.Tensor):
                        metric_value = float(value.detach().item())
                    else:
                        metric_value = float(value)
                    aux_metrics.setdefault(key, []).append(metric_value)
            else:
                logits = head(h_u, h_v)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        mean_loss = total_loss / max(1, num_batches)
        metrics = {"loss": mean_loss}
        for key, values in aux_metrics.items():
            if values:
                metrics[key] = sum(values) / len(values)
        return metrics

    @torch.no_grad()
    def evaluate(self, trunk: nn.Module, head: nn.Module, loader: Iterable):
        if loader is None:
            return {}

        trunk.eval()
        head.eval()
        meter: Dict[str, Any] = {}
        for batch in loader:
            batch = self._prepare_batch(batch)
            h = trunk(batch)
            u, v = self._edge_index(batch)
            h_u = h[u]
            h_v = h[v]
            y = self._edge_labels(batch)

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

    def _prepare_batch(self, batch: Any):
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Empty batch encountered during training")
            batch = batch[0]
        return batch.to(self.device)

    @staticmethod
    def _edge_index(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(batch, "edge_label_index"):
            edge_index = batch.edge_label_index
        elif hasattr(batch, "edge_index"):
            edge_index = batch.edge_index
        else:
            raise AttributeError("Batch is missing edge indices for supervision")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("Edge index must have shape [2, E]")
        return edge_index[0], edge_index[1]

    @staticmethod
    def _edge_labels(batch) -> torch.Tensor:
        if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
            labels = batch.edge_attr
        elif hasattr(batch, "edge_label") and batch.edge_label is not None:
            labels = batch.edge_label
        else:
            raise AttributeError("Batch is missing edge labels for supervision")
        return labels.long().view(-1)


class IdentityTrunk(nn.Module):
    """Simple trunk returning node features as embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, batch) -> torch.Tensor:  # type: ignore[override]
        return self.project(batch.x)
