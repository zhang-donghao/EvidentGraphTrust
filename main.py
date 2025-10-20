from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch import nn

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - allow import failure during docs
    Data = None  # type: ignore

from training.trainer import Config, EvidentialConfig, IdentityTrunk, Trainer, TrainingConfig


def _load_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dict_to_config(data: dict) -> Config:
    evidential = EvidentialConfig(**data.get("evidential", {}))
    training = TrainingConfig(**data.get("training", {}))
    cfg = Config(
        model=data.get("model", "trustguard"),
        head=data.get("head", "mlp"),
        n_classes=data.get("n_classes", 2),
        embedding_dim=data.get("embedding_dim", 64),
        evidential=evidential,
        training=training,
        log_dir=data.get("log_dir", "runs/default"),
    )
    return cfg


def _synthetic_data(cfg: Config, seed: int = 7) -> Tuple[list, list]:
    if Data is None:
        raise ImportError("torch_geometric is required for synthetic data generation")
    random.seed(seed)
    torch.manual_seed(seed)
    num_nodes = 16
    num_edges = 32
    x = torch.randn(num_nodes, cfg.embedding_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randint(0, cfg.n_classes, (num_edges,))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # simple train/val split: identical copies for demonstration
    return [data], [data.clone()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trust modeling trainer entrypoint")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Model trunk to use")
    parser.add_argument("--head", type=str, default=None, choices=["mlp", "evidential"], help="Head type")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name")
    parser.add_argument("--snapshots", type=int, default=1, help="Number of snapshots")
    parser.add_argument("--eval_protocol", type=str, default="single_observed", help="Evaluation protocol")
    parser.add_argument("--log_dir", type=str, default="runs/default", help="Logging directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    cfg_dict = _load_config(args.config)
    cfg = _dict_to_config(cfg_dict)
    if args.model:
        cfg.model = args.model
    if args.head:
        cfg.head = args.head
    cfg.log_dir = args.log_dir or cfg.log_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, device=device)
    trunk: nn.Module = IdentityTrunk(cfg.embedding_dim).to(device)
    head = trainer._build_head(cfg.embedding_dim, cfg.n_classes)

    optimizer = torch.optim.Adam(
        list(trunk.parameters()) + list(head.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    train_loader, val_loader = _synthetic_data(cfg, seed=args.seed)
    for epoch in range(cfg.training.epochs):
        train_loss = trainer.train_one_epoch(trunk, head, train_loader, optimizer)
        trainer._write_metrics(epoch, "train", {"loss": train_loss})
        val_metrics = trainer.evaluate(trunk, head, val_loader)
        trainer._write_metrics(epoch, "val", val_metrics)

    print(f"Training complete. Metrics logged to {Path(cfg.log_dir) / 'metrics.csv'}")


if __name__ == "__main__":
    main()
