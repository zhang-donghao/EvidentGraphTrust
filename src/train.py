"""Training entry point for Evident Graph Trust experiments."""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

if __package__ is None or __package__ == "":  # Allow ``python src/train.py``.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.datamodules import DataModuleConfig, GraphTrustDataset, load_dataset
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


class GraphListDataset(Dataset):
    """Lightweight dataset wrapping an in-memory list of PyG ``Data`` objects."""

    def __init__(
        self,
        data_list: Sequence[Data],
        num_node_features: int,
        num_classes: int,
        metadata: dict,
    ) -> None:
        self._data_list = list(data_list)
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]


def _synthesise_holdout(
    train_ds: GraphTrustDataset,
    val_ds: GraphTrustDataset,
    test_ds: GraphTrustDataset,
) -> Tuple[Dataset, Dataset, Dataset]:
    if len(val_ds) > 0 and len(test_ds) > 0:
        return train_ds, val_ds, test_ds

    combined: List[Data] = []
    combined.extend(train_ds.to_list())
    combined.extend(val_ds.to_list())
    combined.extend(test_ds.to_list())
    total = len(combined)
    if total == 0:
        warnings.warn(
            "No graphs were available to construct validation/test splits. "
            "Consider rerunning preprocessing with a smaller window or stride to generate more samples.",
            RuntimeWarning,
        )
        return train_ds, val_ds, test_ds

    if total < 3:
        # For extremely small corpora we duplicate the available graphs so the
        # trainer can still report metrics. Each clone is detached to avoid
        # sharing storage between the splits.
        base_graphs = [graph.clone() for graph in combined]
        if total == 1:
            train_graphs = [base_graphs[0].clone()]
            val_graphs = [base_graphs[0].clone()]
            test_graphs = [base_graphs[0].clone()]
        else:  # total == 2
            train_graphs = [base_graphs[0].clone()]
            val_graphs = [base_graphs[1].clone()]
            test_graphs = [base_graphs[0].clone()]
        warnings.warn(
            "Validation/test splits were empty; duplicating graphs to create hold-out partitions."
            " For stable metrics rerun preprocessing to increase the number of windows.",
            RuntimeWarning,
        )
        metadata = dict(getattr(train_ds, "metadata", {}))
        num_node_features = train_ds.num_node_features
        num_classes = train_ds.num_classes
        return (
            GraphListDataset(train_graphs, num_node_features, num_classes, metadata),
            GraphListDataset(val_graphs, num_node_features, num_classes, metadata),
            GraphListDataset(test_graphs, num_node_features, num_classes, metadata),
        )

    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(total, generator=generator).tolist()

    val_count = max(1, math.ceil(total * 0.2))
    test_count = max(1, math.ceil(total * 0.2))
    max_holdout = total - 1
    while val_count + test_count > max_holdout:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        remaining = total - train_count
        val_count = max(1, remaining // 2)
        test_count = remaining - val_count
        if test_count == 0 and remaining > 1:
            test_count = 1
            val_count = remaining - test_count
        if val_count == 0 and remaining > 0:
            val_count = 1
            test_count = remaining - val_count

    val_idx = permutation[:val_count]
    test_idx = permutation[val_count : val_count + test_count]
    train_idx = permutation[val_count + test_count :]

    def _gather(indices: Sequence[int]) -> List[Data]:
        return [combined[i].clone() for i in indices]

    metadata = dict(getattr(train_ds, "metadata", {}))
    num_node_features = train_ds.num_node_features
    num_classes = train_ds.num_classes
    warnings.warn(
        "Validation/test splits were empty; synthesising hold-out partitions from available graphs.",
        RuntimeWarning,
    )
    return (
        GraphListDataset(_gather(train_idx), num_node_features, num_classes, metadata),
        GraphListDataset(_gather(val_idx), num_node_features, num_classes, metadata),
        GraphListDataset(_gather(test_idx), num_node_features, num_classes, metadata),
    )


def load_splits(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="train"))
    val_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="val"))
    test_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="test"))
    train_ds, val_ds, test_ds = _synthesise_holdout(train_ds, val_ds, test_ds)
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
        return {
            "loss": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "skipped": True,
            "reason": "No graphs available for this split.",
        }
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
        if val_metrics.get("skipped"):
            val_loss_display = "n/a (empty split)"
            val_ece_display = "n/a"
            val_brier_display = "n/a"
            val_loss = float("inf")
        else:
            val_loss_display = f"{val_loss:.4f}"
            val_ece_display = f"{val_metrics.get('ece', float('nan')):.4f}"
            val_brier_display = f"{val_metrics.get('brier', float('nan')):.4f}"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_display} | "
            f"Val ECE: {val_ece_display} | Val Brier: {val_brier_display}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    if test_metrics.get("skipped"):
        print("Test metrics skipped: no graphs available for evaluation.")
    else:
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
