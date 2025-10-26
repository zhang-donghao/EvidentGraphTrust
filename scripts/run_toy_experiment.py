"""Run a lightweight trust classification toy experiment without heavy deps."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class ToyConfig:
    num_train: int = 180
    num_val: int = 60
    num_test: int = 60
    num_features: int = 4
    num_classes: int = 3
    lr: float = 0.05
    epochs: int = 400
    weight_decay: float = 1e-3
    concentration: float = 12.0
    seed: int = 42


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def generate_gaussian_sample(
    rng: random.Random, mean: Sequence[float], std: Sequence[float]
) -> List[float]:
    return [rng.gauss(m, s) for m, s in zip(mean, std)]


def make_dataset(
    cfg: ToyConfig,
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int], List[List[float]], List[int]]:
    rng = random.Random(cfg.seed)
    means = [
        [-1.5, 0.0, 0.5, -0.5],
        [0.0, 1.5, -0.5, 0.25],
        [1.25, -1.0, 0.0, 1.0],
    ]
    stds = [
        [math.sqrt(v) for v in [0.6, 0.4, 0.3, 0.2]],
        [math.sqrt(v) for v in [0.5, 0.7, 0.4, 0.3]],
        [math.sqrt(v) for v in [0.4, 0.6, 0.5, 0.4]],
    ]

    num_total = cfg.num_train + cfg.num_val + cfg.num_test
    per_class = num_total // cfg.num_classes
    features: List[List[float]] = []
    labels: List[int] = []
    for cls in range(cfg.num_classes):
        for _ in range(per_class):
            features.append(generate_gaussian_sample(rng, means[cls], stds[cls]))
            labels.append(cls)

    indices = list(range(len(features)))
    rng.shuffle(indices)
    train_idx = indices[: cfg.num_train]
    val_idx = indices[cfg.num_train : cfg.num_train + cfg.num_val]
    test_idx = indices[cfg.num_train + cfg.num_val : cfg.num_train + cfg.num_val + cfg.num_test]

    def subset(idx: List[int]) -> Tuple[List[List[float]], List[int]]:
        return [features[i] for i in idx], [labels[i] for i in idx]

    X_train, y_train = subset(train_idx)
    X_val, y_val = subset(val_idx)
    X_test, y_test = subset(test_idx)
    return X_train, y_train, X_val, y_val, X_test, y_test


def initialize_parameters(cfg: ToyConfig) -> Tuple[List[List[float]], List[float]]:
    rng = random.Random(cfg.seed + 1)
    weights = [
        [rng.uniform(-0.1, 0.1) for _ in range(cfg.num_classes)]
        for _ in range(cfg.num_features)
    ]
    bias = [0.0 for _ in range(cfg.num_classes)]
    return weights, bias


def train_softmax(
    X: List[List[float]],
    y: List[int],
    cfg: ToyConfig,
) -> Tuple[List[List[float]], List[float]]:
    weights, bias = initialize_parameters(cfg)
    n = len(X)

    for _ in range(cfg.epochs):
        grad_w = [[0.0 for _ in range(cfg.num_classes)] for _ in range(cfg.num_features)]
        grad_b = [0.0 for _ in range(cfg.num_classes)]
        for features, label in zip(X, y):
            logits = [
                bias[c]
                + sum(features[i] * weights[i][c] for i in range(cfg.num_features))
                for c in range(cfg.num_classes)
            ]
            probs = softmax(logits)
            for c in range(cfg.num_classes):
                error = probs[c] - (1.0 if c == label else 0.0)
                for i in range(cfg.num_features):
                    grad_w[i][c] += error * features[i]
                grad_b[c] += error
        for i in range(cfg.num_features):
            for c in range(cfg.num_classes):
                grad = grad_w[i][c] / n + cfg.weight_decay * weights[i][c]
                weights[i][c] -= cfg.lr * grad
        for c in range(cfg.num_classes):
            grad = grad_b[c] / n
            bias[c] -= cfg.lr * grad
    return weights, bias


def evaluate(
    X: List[List[float]],
    y: List[int],
    weights: List[List[float]],
    bias: List[float],
    cfg: ToyConfig,
) -> Dict[str, float]:
    probs_list: List[List[float]] = []
    preds: List[int] = []
    for features in X:
        logits = [
            bias[c] + sum(features[i] * weights[i][c] for i in range(cfg.num_features))
            for c in range(cfg.num_classes)
        ]
        probs = softmax(logits)
        probs_list.append(probs)
        preds.append(max(range(cfg.num_classes), key=lambda c: probs[c]))

    accuracy = sum(int(p == t) for p, t in zip(preds, y)) / len(y)

    nll = 0.0
    brier = 0.0
    concentration = cfg.concentration
    uncertainty_total = 0.0
    entropy = 0.0

    for probs, label in zip(probs_list, y):
        prob_label = max(min(probs[label], 1.0), 1e-12)
        nll -= math.log(prob_label)
        one_hot = [1.0 if c == label else 0.0 for c in range(cfg.num_classes)]
        brier += sum((p - oh) ** 2 for p, oh in zip(probs, one_hot))
        alpha = [p * concentration + 1.0 for p in probs]
        strength = sum(alpha)
        uncertainty_total += cfg.num_classes / strength
        entropy -= sum(p * math.log(max(p, 1e-12)) for p in probs)
    nll /= len(y)
    brier /= len(y)
    uncertainty = uncertainty_total / len(y)
    entropy /= len(y)

    ece = expected_calibration_error(probs_list, preds, y, n_bins=10)
    macro_f1 = macro_f1_score(preds, y, cfg.num_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "uncertainty": uncertainty,
        "entropy": entropy,
    }


def expected_calibration_error(
    probs_list: List[List[float]], preds: List[int], labels: List[int], n_bins: int = 10
) -> float:
    confidences = [max(probs) for probs in probs_list]
    accuracies = [1.0 if p == t else 0.0 for p, t in zip(preds, labels)]
    ece = 0.0
    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins
        bucket_indices = [i for i, conf in enumerate(confidences) if lower < conf <= upper]
        if not bucket_indices:
            continue
        bucket_acc = sum(accuracies[i] for i in bucket_indices) / len(bucket_indices)
        bucket_conf = sum(confidences[i] for i in bucket_indices) / len(bucket_indices)
        weight = len(bucket_indices) / len(confidences)
        ece += abs(bucket_conf - bucket_acc) * weight
    return ece


def macro_f1_score(preds: List[int], labels: List[int], num_classes: int) -> float:
    f1_scores = []
    for cls in range(num_classes):
        tp = sum(1 for p, t in zip(preds, labels) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, labels) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, labels) if p != cls and t == cls)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    for key in ["accuracy", "macro_f1", "nll", "brier", "ece", "uncertainty", "entropy"]:
        value = metrics[key]
        print(f"  {key:>10}: {value:.4f}")


def main() -> None:
    cfg = ToyConfig()
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_dataset(cfg)
    weights, bias = train_softmax(X_tr, y_tr, cfg)

    val_metrics = evaluate(X_val, y_val, weights, bias, cfg)
    test_metrics = evaluate(X_te, y_te, weights, bias, cfg)

    print_metrics("Toy validation metrics:", val_metrics)
    print()
    print_metrics("Toy test metrics:", test_metrics)


if __name__ == "__main__":
    main()
