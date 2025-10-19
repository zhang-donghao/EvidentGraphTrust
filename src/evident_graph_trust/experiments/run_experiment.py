"""Command line entry point for reproducing the EvidentGraphTrust study."""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch

from ..data.iot_graph import generate_iot_graph
from ..evaluation.reporting import tabulate_metrics
from ..models.gnn import EvidentialGNN, GCNClassifier, GraphSAGEClassifier
from ..models.ml_baseline import LogisticRegressionClassifier
from ..training.loop import TrainingConfig, train_model
from ..utils.seed import set_seed


def _train_and_collect(
    name: str,
    model,
    features,
    adjacency,
    labels,
    masks,
    config: TrainingConfig,
) -> Tuple[str, Dict[str, float], torch.Tensor, torch.Tensor]:
    train_mask, val_mask, test_mask = masks
    result = train_model(
        model=model,
        features=features,
        adjacency=adjacency,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        config=config,
    )
    return name, result.metrics, result.trust_scores, result.uncertainties


def summarise_suspicious_nodes(
    trust_scores: torch.Tensor,
    uncertainties: torch.Tensor,
    labels: torch.Tensor,
    test_mask: torch.Tensor,
    top_k: int = 5,
) -> str:
    malicious_prob = trust_scores[:, -1]
    selection = torch.nonzero(test_mask).squeeze(-1)
    subset = torch.stack([selection.float(), malicious_prob[test_mask], uncertainties[test_mask]], dim=1)
    sorted_idx = torch.argsort(subset[:, 1], descending=True)
    lines = ["Top nodes flagged as untrustworthy (test set):"]
    for rank, idx in enumerate(sorted_idx[:top_k]):
        node_id = int(subset[idx, 0].item())
        prob = subset[idx, 1].item()
        uncert = subset[idx, 2].item()
        label = int(labels[node_id].item())
        lines.append(f"  #{rank + 1}: node={node_id}, malicious_prob={prob:.3f}, uncertainty={uncert:.3f}, label={label}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data = generate_iot_graph(
        num_nodes=args.num_nodes,
        feature_dim=args.feature_dim,
        seed=args.seed,
        structure_enhancement=True,
    )

    features = data.features
    adjacency = data.adjacency
    labels = data.labels
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    models = [
        (
            "EvidentialGNN",
            EvidentialGNN(
                in_features=features.size(1),
                hidden_features=args.hidden_dim,
                num_classes=labels.max().item() + 1,
                dropout=0.35,
            ),
        ),
        (
            "GCN",
            GCNClassifier(
                in_features=features.size(1),
                hidden_features=args.hidden_dim,
                num_classes=labels.max().item() + 1,
            ),
        ),
        (
            "GraphSAGE",
            GraphSAGEClassifier(
                in_features=features.size(1),
                hidden_features=args.hidden_dim,
                num_classes=labels.max().item() + 1,
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegressionClassifier(
                in_features=features.size(1),
                num_classes=labels.max().item() + 1,
            ),
        ),
    ]

    results: Dict[str, Dict[str, float]] = {}
    evidential_trust = None
    evidential_uncertainty = None

    for name, model in models:
        config = TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            kl_weight=args.kl_weight,
            use_evidence_loss=(name == "EvidentialGNN" and not args.disable_evidence_loss),
            use_graph_enhancement=(not args.disable_graph_enhancement),
            annealing_epochs=args.annealing_epochs,
            device=args.device,
        )
        model_name, metrics, trust_scores, uncertainties = _train_and_collect(
            name,
            model,
            features,
            adjacency,
            labels,
            (train_mask, val_mask, test_mask),
            config,
        )
        results[model_name] = metrics
        if model_name == "EvidentialGNN":
            evidential_trust = trust_scores
            evidential_uncertainty = uncertainties

    print("=== Main comparison ===")
    print(tabulate_metrics(results))

    if evidential_trust is not None and evidential_uncertainty is not None:
        print()
        print(summarise_suspicious_nodes(evidential_trust, evidential_uncertainty, labels, test_mask, top_k=args.top_k))

    print()
    print("=== Ablation study (EvidentialGNN variants) ===")
    ablation_settings = {
        "Full": dict(use_evidence_loss=True, use_graph_enhancement=True),
        "No Evidence Loss": dict(use_evidence_loss=False, use_graph_enhancement=True),
        "No Graph Enhancement": dict(use_evidence_loss=True, use_graph_enhancement=False),
    }

    ablation_results: Dict[str, Dict[str, float]] = {}
    for label, flags in ablation_settings.items():
        model = EvidentialGNN(
            in_features=features.size(1),
            hidden_features=args.hidden_dim,
            num_classes=labels.max().item() + 1,
            dropout=0.35,
        )
        config = TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            kl_weight=args.kl_weight,
            use_evidence_loss=flags["use_evidence_loss"],
            use_graph_enhancement=flags["use_graph_enhancement"],
            annealing_epochs=args.annealing_epochs,
            device=args.device,
        )
        _, metrics, _, _ = _train_and_collect(
            label,
            model,
            features,
            adjacency,
            labels,
            (train_mask, val_mask, test_mask),
            config,
        )
        ablation_results[label] = metrics

    print(tabulate_metrics(ablation_results))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidential trust analysis in IoT graphs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-nodes", type=int, default=420)
    parser.add_argument("--feature-dim", type=int, default=18)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--kl-weight", type=float, default=0.8)
    parser.add_argument("--annealing-epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-evidence-loss", action="store_true")
    parser.add_argument("--disable-graph-enhancement", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
