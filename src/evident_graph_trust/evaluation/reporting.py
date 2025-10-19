"""Utility helpers for experiment reporting."""
from __future__ import annotations

from typing import Dict, Iterable


def tabulate_metrics(results: Dict[str, Dict[str, float]]) -> str:
    """Return a simple Markdown table summarising metrics."""

    headers = ["Model", "Accuracy", "NLL", "ECE"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---|---|---|"]
    for model_name, metrics in results.items():
        line = f"| {model_name} | {metrics['test_accuracy']:.3f} | {metrics['test_nll']:.3f} | {metrics['test_ece']:.3f} |"
        lines.append(line)
    return "\n".join(lines)
