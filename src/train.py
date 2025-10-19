"""Training entry point for Evident Graph Trust experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

if __package__ is None or __package__ == "":  # Allow ``python src/train.py``.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.datamodules import DataModuleConfig, load_dataset
from src.models.egtn import EGTNConfig, EvidentialGraphTrustNetwork
from src.utils.metrics import trust_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Evidential Graph Trust Network")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to preprocessed dataset root")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset identifier (e.g., veremi, toniot)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-gat", action="store_true")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kl-strength", type=float, default=1e-3)
    return parser.parse_args()


def load_splits(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="train"))
    val_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="val"))
    test_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="test"))
    return (
        DataLoader(train_ds, batch_size=1, shuffle=True),
        DataLoader(val_ds, batch_size=1, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False),
    )


def train_epoch(model: EvidentialGraphTrustNetwork, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
        logits = outputs["logits"]
        loss = criterion(logits, batch.y)
        loss += model.regularization_loss(outputs["alpha"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    dataset_size = len(loader.dataset)
    return total_loss / dataset_size if dataset_size > 0 else 0.0


def evaluate(model: EvidentialGraphTrustNetwork, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    metrics = []
    dataset_size = len(loader.dataset)
    if dataset_size == 0:
        return {"loss": float("nan")}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            logits = outputs["logits"]
            loss = criterion(logits, batch.y)
            loss += model.regularization_loss(outputs["alpha"])
            total_loss += loss.item() * batch.num_nodes
            metrics.append(trust_evaluation(outputs["alpha"], batch.y))
    mean_metrics = {k: torch.stack([m[k] for m in metrics]).mean().item() for k in metrics[0]} if metrics else {}
    mean_metrics["loss"] = total_loss / dataset_size
    return mean_metrics


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_splits(args)

    input_dim = train_loader.dataset.num_node_features  # type: ignore[attr-defined]
    output_dim = train_loader.dataset.num_classes  # type: ignore[attr-defined]

    config = EGTNConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        use_gat=args.use_gat,
        heads=args.heads,
        kl_strength=args.kl_strength,
    )
    model = EvidentialGraphTrustNetwork(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val ECE: {val_metrics.get('ece', float('nan')):.4f} | Val Brier: {val_metrics.get('brier', float('nan')):.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
