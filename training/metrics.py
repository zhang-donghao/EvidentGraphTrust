from __future__ import annotations

import math
from typing import Dict, List

import torch

from trustcore.models.efg_components.dirichlet import dirichlet_nll

try:  # optional dependency for richer metrics
    from sklearn import metrics as skmetrics
except Exception:  # pragma: no cover - fallback path
    skmetrics = None


def _concat(meter: Dict[str, List[torch.Tensor]], key: str) -> torch.Tensor:
    return torch.cat(meter.get(key, []), dim=0) if meter.get(key) else torch.empty(0)


def update_classification_metrics(
    meter: dict,
    pred: torch.Tensor,
    target: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    meter.setdefault("preds", []).append(pred.detach().cpu())
    meter.setdefault("targets", []).append(target.detach().cpu())
    meter.setdefault("probs", []).append(probs.detach().cpu())


def _balanced_accuracy(target: torch.Tensor, pred: torch.Tensor) -> float:
    num_classes = int(target.max().item() + 1) if target.numel() else 0
    if num_classes == 0:
        return float("nan")
    ba = 0.0
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        ba += (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return ba / num_classes


def _f1_macro(target: torch.Tensor, pred: torch.Tensor) -> float:
    num_classes = int(target.max().item() + 1) if target.numel() else 0
    if num_classes == 0:
        return float("nan")
    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / num_classes


def _mcc(target: torch.Tensor, pred: torch.Tensor) -> float:
    if skmetrics is not None and target.numel():
        return float(skmetrics.matthews_corrcoef(target.numpy(), pred.numpy()))
    # fallback binary MCC
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / denom


def _auc(target: torch.Tensor, probs: torch.Tensor) -> float:
    if probs.numel() == 0:
        return float("nan")
    if probs.size(1) == 2:
        pos_scores = probs[:, 1]
        sorted_scores, indices = torch.sort(pos_scores)
        sorted_targets = target[indices]
        cum_pos = torch.cumsum(sorted_targets == 1, dim=0)
        cum_neg = torch.cumsum(sorted_targets == 0, dim=0)
        tp = cum_pos.float()
        fp = cum_neg.float()
        tp = torch.cat([torch.tensor([0.0]), tp])
        fp = torch.cat([torch.tensor([0.0]), fp])
        tp_rate = tp / (tp[-1] if tp[-1] > 0 else 1.0)
        fp_rate = fp / (fp[-1] if fp[-1] > 0 else 1.0)
        auc = torch.trapz(tp_rate, fp_rate).item()
        return auc
    if skmetrics is not None:
        return float(skmetrics.roc_auc_score(target.numpy(), probs.numpy(), multi_class="ovr"))
    return float("nan")


def nll_metric(meter: dict, alpha: torch.Tensor, target: torch.Tensor):
    val = dirichlet_nll(alpha, target).mean().item()
    meter.setdefault("NLL", []).append(val)


def ece_metric(meter: dict, probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    conf, pred = probs.max(dim=-1)
    correct = (pred == target).float()
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc = correct[mask].mean()
            conf_avg = conf[mask].mean()
            ece += mask.float().mean() * (acc - conf_avg).abs()
    meter.setdefault("ECE", []).append(ece.item())


def finalize_metrics(meter: dict) -> dict:
    preds = _concat(meter, "preds")
    targets = _concat(meter, "targets")
    probs = _concat(meter, "probs")

    results: Dict[str, float] = {}
    if preds.numel():
        results["MCC"] = _mcc(targets, preds)
        results["BA"] = _balanced_accuracy(targets, preds)
        results["F1"] = _f1_macro(targets, preds)
    if probs.numel():
        results["AUC"] = _auc(targets, probs)

    for key, values in meter.items():
        if key in {"preds", "targets", "probs"}:
            continue
        if isinstance(values, list) and values:
            results[key] = float(sum(values) / len(values))
    return results
