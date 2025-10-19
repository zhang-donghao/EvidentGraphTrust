"""Training entry point for Evident Graph Trust experiments."""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn, optim
from torch_geometric.loader import DataLoader

if __package__ is None or __package__ == "":  # Allow ``python src/train.py``.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.datamodules import DataModuleConfig, GraphTrustDataset, load_dataset
from src.models import BaselineConfig, EGTNConfig, EvidentialGraphTrustNetwork, build_baseline
from src.utils.metrics import trust_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Evident Graph Trust Network and baselines")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to preprocessed dataset root")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset identifier (e.g., veremi, toni_iot)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="egtn", choices=["egtn", "gcn", "gat", "graphsage", "mlp"], help="Model architecture for comparison experiments")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads for GAT/EGT")
    parser.add_argument("--kl-strength", type=float, default=1e-3)
    parser.add_argument("--disable-evidence-regularizer", action="store_true", help="Disable evidential KL regulariser (ablation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("runs"), help="Directory used to store metrics and visualisations")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for the run folder")
    parser.add_argument("--reliability-bins", type=int, default=15, help="Number of bins for reliability diagrams")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="train"))
    val_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="val"))
    test_ds = load_dataset(DataModuleConfig(args.dataset_root, args.dataset_name, split="test"))

    datasets: Dict[str, object] = {"train": train_ds, "val": val_ds, "test": test_ds}
    synthetic_splits: List[str] = []

    missing = [name for name, dataset in datasets.items() if len(dataset) == 0]
    if missing:
        base_graphs = train_ds.to_list()
        if not base_graphs:
            raise RuntimeError(
                "All dataset splits are empty; unable to synthesise holdouts. Rerun preprocessing with a smaller window, "
                "stride, or --min-nodes setting to generate graphs before training."
            )
        warnings.warn(
            "Dataset split(s) "
            + ", ".join(sorted(missing))
            + " are empty. Synthesising holdout graphs from the training split for diagnostic purposes. "
            "Re-run preprocessing with more windows for reliable evaluation metrics."
        )
        target_size = max(1, len(base_graphs))
        for name in missing:
            synthetic = [base_graphs[idx % len(base_graphs)].clone() for idx in range(target_size)]
            datasets[name] = synthetic
            synthetic_splits.append(name)

    metadata = {
        "num_node_features": train_ds.num_node_features,
        "num_classes": train_ds.num_classes,
        "synthetic_splits": synthetic_splits,
    }

    return (
        DataLoader(datasets["train"], batch_size=1, shuffle=True),
        DataLoader(datasets["val"], batch_size=1, shuffle=False),
        DataLoader(datasets["test"], batch_size=1, shuffle=False),
        metadata,
    )


def create_model(args: argparse.Namespace, metadata: Dict[str, int]) -> Tuple[nn.Module, bool]:
    input_dim = metadata["num_node_features"]
    num_classes = metadata["num_classes"]
    if input_dim == 0 or num_classes == 0:
        raise RuntimeError(
            "Dataset metadata is incomplete. Ensure preprocessing populated feature and class counts in summary.json."
        )

    if args.model == "egtn":
        config = EGTNConfig(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_gat=args.heads > 1,
            heads=args.heads,
            kl_strength=args.kl_strength,
        )
        model = EvidentialGraphTrustNetwork(config)
        evidential = True
    else:
        config = BaselineConfig(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=args.heads,
            dirichlet_strength=float(num_classes),
        )
        model = build_baseline(args.model, config)
        evidential = False
    return model, evidential


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    evidential: bool,
    disable_reg: bool,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    dataset_size = len(loader.dataset)
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
        logits = outputs["logits"]
        loss = criterion(logits, batch.y)
        if evidential and not disable_reg:
            loss += model.regularization_loss(outputs["alpha"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / dataset_size if dataset_size > 0 else 0.0


def _collect_predictions(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    alpha = outputs["alpha"]
    probs = alpha / alpha.sum(dim=-1, keepdim=True)
    preds = probs.argmax(dim=-1)
    return {"alpha": alpha, "probs": probs, "preds": preds, "labels": labels}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    evidential: bool,
    disable_reg: bool,
    synthetic: bool = False,
) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    dataset_size = len(loader.dataset)
    if dataset_size == 0:
        return {
            "loss": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "skipped": True,
            "reason": "No graphs available for this split.",
            "details": {},
            "synthetic_split": synthetic,
            "split_size": dataset_size,
        }

    collected: Dict[str, List[torch.Tensor]] = {"alpha": [], "probs": [], "preds": [], "labels": []}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
            logits = outputs["logits"]
            loss = criterion(logits, batch.y)
            if evidential and not disable_reg:
                loss += model.regularization_loss(outputs["alpha"])
            total_loss += loss.item()
            pred_bundle = _collect_predictions(outputs, batch.y)
            for key, value in pred_bundle.items():
                collected[key].append(value.cpu())

    alpha = torch.cat(collected["alpha"], dim=0)
    probs = torch.cat(collected["probs"], dim=0)
    labels = torch.cat(collected["labels"], dim=0)
    preds = torch.cat(collected["preds"], dim=0)

    metrics = trust_evaluation(alpha, labels)
    metrics = {key: value.item() for key, value in metrics.items()}
    metrics["loss"] = total_loss / dataset_size
    metrics["accuracy"] = accuracy_score(labels.numpy(), preds.numpy())
    metrics["f1"] = f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)

    try:
        if probs.size(-1) == 2:
            pos_probs = probs[:, 1].numpy()
            metrics["roc_auc"] = roc_auc_score(labels.numpy(), pos_probs)
            metrics["pr_auc"] = average_precision_score(labels.numpy(), pos_probs)
        else:
            metrics["roc_auc"] = roc_auc_score(labels.numpy(), probs.numpy(), multi_class="ovr", average="macro")
            metrics["pr_auc"] = average_precision_score(
                torch.nn.functional.one_hot(labels, probs.size(-1)).numpy(),
                probs.numpy(),
                average="macro",
            )
    except ValueError:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    metrics["skipped"] = False
    metrics["details"] = {
        "labels": labels.numpy().tolist(),
        "probs": probs.numpy().tolist(),
        "preds": preds.numpy().tolist(),
    }
    metrics["synthetic_split"] = synthetic
    metrics["split_size"] = dataset_size
    return metrics


def compute_reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> Dict[str, List[float]]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres: List[float] = []
    bin_accuracy: List[float] = []
    bin_confidence: List[float] = []
    counts: List[int] = []

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        bin_centres.append((lower + upper) / 2.0)
        bin_accuracy.append(accuracies[mask].mean())
        bin_confidence.append(confidences[mask].mean())
        counts.append(int(mask.sum()))

    return {
        "bin_centres": bin_centres,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "counts": counts,
    }


def plot_reliability(curve: Dict[str, List[float]], path: Path) -> None:
    if not curve["bin_centres"]:
        warnings.warn("Reliability diagram skipped: no probability bins with samples.")
        return
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(curve["bin_centres"], curve["bin_accuracy"], width=1.0 / max(len(curve["bin_centres"]), 1), alpha=0.6, label="Accuracy")
    plt.plot(curve["bin_centres"], curve["bin_confidence"], marker="o", color="tab:orange", label="Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def save_run_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    history: List[Dict[str, float]],
    val_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
    reliability: Dict[str, List[float]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    serialisable_args = vars(args).copy()
    serialisable_args["dataset_root"] = str(serialisable_args["dataset_root"])
    serialisable_args["save_dir"] = str(serialisable_args["save_dir"])
    (run_dir / "config.json").write_text(json.dumps(serialisable_args, indent=2, ensure_ascii=False))
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    (run_dir / "reliability.json").write_text(json.dumps(reliability, indent=2, ensure_ascii=False))
    plot_reliability(reliability, run_dir / "reliability.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    train_loader, val_loader, test_loader, metadata = load_splits(args)
    model, evidential = create_model(args, metadata)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.dataset_name}_{args.model}_seed{args.seed}_{timestamp}"
    run_dir = args.save_dir / args.dataset_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, evidential, args.disable_evidence_regularizer)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            evidential,
            args.disable_evidence_regularizer,
            synthetic="val" in metadata.get("synthetic_splits", []),
        )
        val_loss = float(val_metrics.get("loss", float("nan")))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_loss,
                "val_ece": float(val_metrics.get("ece", float("nan"))),
                "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
                "val_synthetic": bool(val_metrics.get("synthetic_split", False)),
            }
        )
        message_parts = [f"Epoch {epoch:03d}", f"Train Loss: {train_loss:.4f}"]
        if val_metrics.get("skipped"):
            message_parts.extend(["Val Loss: n/a", "Val Acc: n/a", "Val ECE: n/a"])
        else:
            suffix = " (synthetic)" if val_metrics.get("synthetic_split") else ""
            message_parts.append(f"Val Loss{suffix}: {val_metrics['loss']:.4f}")
            message_parts.append(f"Val Acc: {val_metrics['accuracy']:.4f}")
            message_parts.append(f"Val ECE: {val_metrics['ece']:.4f}")
        val_message = " | ".join(message_parts)
        print(val_message)
        if not val_metrics.get("skipped") and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        evidential,
        args.disable_evidence_regularizer,
        synthetic="test" in metadata.get("synthetic_splits", []),
    )
    if test_metrics.get("skipped"):
        print("Test metrics skipped: no graphs available for evaluation.")
    elif test_metrics.get("synthetic_split"):
        print(
            "Test metrics computed on a synthetic holdout. Regenerate preprocessing outputs with more windows for stable "
            "evaluation."
        )
        print(
            f"Test metrics (synthetic): accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
            f"ece={test_metrics['ece']:.4f}"
        )
    else:
        print(f"Test metrics: accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, ece={test_metrics['ece']:.4f}")

    reliability: Dict[str, List[float]]
    if test_metrics.get("skipped"):
        reliability = {"bin_centres": [], "bin_accuracy": [], "bin_confidence": [], "counts": []}
    else:
        details = test_metrics.get("details", {})
        probs = np.array(details.get("probs", []), dtype=float)
        labels = np.array(details.get("labels", []), dtype=int)
        reliability = compute_reliability_curve(probs, labels, args.reliability_bins)

    save_run_artifacts(run_dir, args, history, val_metrics, test_metrics, reliability)
    print(f"Run artefacts saved to {run_dir}")


if __name__ == "__main__":
    main()
