"""Batch runner for comparison and ablation experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


COMPARISON_EXPERIMENTS = {
    "egtn": ["--model", "egtn"],
    "gcn": ["--model", "gcn"],
    "gat": ["--model", "gat"],
    "graphsage": ["--model", "graphsage"],
    "mlp": ["--model", "mlp"],
}

ABLATION_EXPERIMENTS = {
    "egtn_no_reg": ["--model", "egtn", "--disable-evidence-regularizer"],
    "egtn_one_layer": ["--model", "egtn", "--num-layers", "1"],
    "egtn_no_attention": ["--model", "egtn", "--heads", "1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison and ablation experiments")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["comparison", "ablation"],
        default=["comparison", "ablation"],
        help="Select which experiment groups to execute",
    )
    parser.add_argument("--reliability-bins", type=int, default=15)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def build_command(
    base: List[str],
    experiment_name: str,
    extra_args: List[str],
    seed: int,
) -> List[str]:
    run_name = f"{experiment_name}_seed{seed}"
    cmd = base + ["--seed", str(seed), "--run-name", run_name]
    cmd.extend(extra_args)
    return cmd


def load_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_file = run_dir / "test_metrics.json"
    if not metrics_file.exists():
        return {}
    try:
        with metrics_file.open("r", encoding="utf-8") as fp:
            metrics = json.load(fp)
    except json.JSONDecodeError:
        return {}
    filtered = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    filtered["skipped"] = metrics.get("skipped", False)
    return filtered


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        args.dataset_root / args.dataset_name / "processed" / "summary.json"
    )
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        empty = [
            name
            for name in ("train", "val", "test")
            if isinstance(summary.get(name), dict) and summary[name].get("num_graphs", 0) == 0
        ]
        if empty:
            raise RuntimeError(
                "Dataset split(s) "
                + ", ".join(sorted(empty))
                + " contain zero graphs according to processed/summary.json. "
                "Regenerate the dataset with scripts/preprocess_toniot.py using a smaller window "
                "or stride before running experiments."
            )

    base_cmd = [
        sys.executable,
        "src/train.py",
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-name",
        args.dataset_name,
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--save-dir",
        str(args.output_dir),
        "--reliability-bins",
        str(args.reliability_bins),
    ]
    if args.no_cuda:
        base_cmd.append("--no-cuda")

    executed: List[Dict[str, object]] = []

    def _run_group(group: Dict[str, List[str]], label: str) -> None:
        for name, extra in group.items():
            for seed in args.seeds:
                cmd = build_command(base_cmd.copy(), name, extra, seed)
                print("Running", " ".join(cmd))
                subprocess.run(cmd, check=True)
                run_dir = args.output_dir / args.dataset_name / f"{name}_seed{seed}"
                metrics = load_metrics(run_dir)
                record = {
                    "group": label,
                    "experiment": name,
                    "seed": seed,
                }
                record.update(metrics)
                executed.append(record)

    if "comparison" in args.modes:
        _run_group(COMPARISON_EXPERIMENTS, "comparison")
    if "ablation" in args.modes:
        _run_group(ABLATION_EXPERIMENTS, "ablation")

    summary_path = args.output_dir / args.dataset_name / "summary.json"
    summary_path.write_text(json.dumps(executed, indent=2, ensure_ascii=False))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
