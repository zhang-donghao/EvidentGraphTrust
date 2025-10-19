#!/usr/bin/env python
"""Preprocess the TON_IoT network telemetry into TrustGuard-style graph windows.

This script mirrors the data handling flow used by the TrustGuard project.  The
original repository aggregates IoT flow records into temporal windows,
constructs directed host graphs, and stores PyTorch Geometric ``Data`` objects
for downstream graph neural network training.  The implementation below follows
that recipe but keeps the model layer flexible so we can train Evidential GNNs
on top of the processed artefacts.

The CLI expects the raw ``Train_Test_Network.csv`` file (or its renamed
counterpart ``train_test_network.csv``).  By default we look for the file inside
``data/raw/ton_iot`` to match TrustGuard's layout, but any custom location can be
specified via ``--raw-root`` or ``--network-file``.  The output is saved under
``<output-root>/toni_iot/processed`` with one ``*.pt`` file per split and a
compact ``summary.json`` mirroring TrustGuard's dataset manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    raw_root: Path
    output_root: Path
    network_file: Optional[Path]
    window_size: float
    stride: float
    train_ratio: float
    val_ratio: float
    min_rows_per_window: int
    min_nodes: int
    min_graphs_per_split: int
    random_seed: int

    @property
    def test_ratio(self) -> float:
        return max(0.0, 1.0 - self.train_ratio - self.val_ratio)


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/ton_iot"),
        help="Directory containing the raw TON_IoT CSV assets (TrustGuard layout).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory where processed graphs will be stored.",
    )
    parser.add_argument(
        "--network-file",
        type=Path,
        default=None,
        help="Explicit path to Train_Test_Network.csv. Overrides --raw-root lookup.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=120.0,
        help="Temporal window size in seconds (row-count fallback when timestamps are missing).",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=60.0,
        help="Stride between windows in seconds (row-count fallback when timestamps are missing).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Proportion of windows assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion of windows assigned to the validation split (test receives the remainder).",
    )
    parser.add_argument(
        "--min-rows-per-window",
        type=int,
        default=32,
        help="Discard windows containing fewer records than this threshold.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Discard graphs with fewer than this number of nodes after aggregation.",
    )
    parser.add_argument(
        "--min-graphs-per-split",
        type=int,
        default=30,
        help="Abort preprocessing if any split would end up with fewer graphs than this count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling (applies when timestamps are absent).",
    )
    args = parser.parse_args()

    return PreprocessConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        network_file=args.network_file,
        window_size=args.window_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_rows_per_window=args.min_rows_per_window,
        min_nodes=args.min_nodes,
        min_graphs_per_split=args.min_graphs_per_split,
        random_seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Data discovery utilities
# ---------------------------------------------------------------------------


def _candidate_network_paths(config: PreprocessConfig) -> List[Path]:
    if config.network_file is not None:
        return [config.network_file]

    candidates: List[Path] = []
    if config.raw_root.exists():
        candidates.extend(sorted(config.raw_root.glob("**/*network*.csv")))
    # historical fallbacks inside the repo for quick starts
    repo_default = Path("src/data/train_test_network.csv")
    if repo_default.exists():
        candidates.append(repo_default)
    return candidates


def _load_network_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Network CSV {path} is empty – cannot build graphs.")
    # Normalise column names to lower-case for easier matching
    df.columns = [col.lower() for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Column resolution helpers mirroring TrustGuard's preprocessing
# ---------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, options: Sequence[str]) -> Optional[str]:
    for candidate in options:
        if candidate.lower() in df.columns:
            return candidate.lower()
    return None


def _resolve_required_column(df: pd.DataFrame, options: Sequence[str], description: str) -> str:
    column = _resolve_column(df, options)
    if column is None:
        raise RuntimeError(
            f"Could not find a column for {description}. Looked for: {', '.join(options)}."
        )
    return column


# ---------------------------------------------------------------------------
# Window generation and graph construction
# ---------------------------------------------------------------------------


def _extract_timestamps(df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    ts_col = _resolve_column(df, ["ts", "timestamp", "time", "date"])
    if ts_col is None:
        # Fallback: fabricate a pseudo timeline using the row index.  This
        # mirrors TrustGuard's behaviour when working with subsets lacking
        # explicit timestamps.
        return np.arange(len(df), dtype=float), False

    values = df[ts_col]
    if np.issubdtype(values.dtype, np.number):
        ts_seconds = values.astype(float).to_numpy()
        return ts_seconds, True

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isnull().all():
        # Unable to parse the textual timestamps, fall back to row indices.
        return np.arange(len(df), dtype=float), False
    ts_seconds = parsed.astype("int64") / 1_000_000_000
    return ts_seconds.to_numpy(), True


def _iter_time_windows(
    timestamps: np.ndarray,
    has_real_time: bool,
    window_size: float,
    stride: float,
    min_rows: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield index arrays for each temporal window.

    Parameters mirror TrustGuard's sliding-window generation.  When true
    timestamps are missing we interpret ``window_size`` and ``stride`` as row
    counts to retain deterministic behaviour.
    """

    if len(timestamps) == 0:
        return

    if has_real_time:
        order = np.argsort(timestamps)
        ordered_ts = timestamps[order]
        start_time = float(ordered_ts[0])
        end_time = float(ordered_ts[-1])
        window = float(window_size)
        step = max(float(stride), 1.0)
        current = start_time
        while current <= end_time:
            mask = (ordered_ts >= current) & (ordered_ts < current + window)
            indices = order[mask]
            if indices.size >= min_rows:
                yield int(current), indices
            current += step
    else:
        # Interpret sizes as counts
        order = np.arange(len(timestamps))
        window_count = max(int(round(window_size)), min_rows)
        step = max(int(round(stride)), 1)
        for start in range(0, len(order), step):
            stop = start + window_count
            indices = order[start:stop]
            if indices.size >= min_rows:
                yield start, indices


def _aggregate_window(
    df: pd.DataFrame,
    index_array: np.ndarray,
    time_anchor: float,
    has_real_time: bool,
    src_col: str,
    dst_col: str,
    src_port_col: Optional[str],
    dst_port_col: Optional[str],
    duration_col: Optional[str],
    orig_bytes_col: Optional[str],
    resp_bytes_col: Optional[str],
    orig_pkts_col: Optional[str],
    resp_pkts_col: Optional[str],
    label_col: Optional[str],
    min_nodes: int,
) -> Optional[Data]:
    window_df = df.iloc[index_array]
    if window_df.empty:
        return None

    # Determine the window bounds for metadata
    if has_real_time and "_egt_ts" in window_df:
        ts_values = pd.to_numeric(window_df["_egt_ts"], errors="coerce").to_numpy(dtype=float)
        if ts_values.size > 0 and not np.isnan(ts_values).all():
            window_start = float(np.nanmin(ts_values))
            window_end = float(np.nanmax(ts_values))
        else:
            window_start = float(time_anchor)
            window_end = float(time_anchor + window_df.shape[0])
    else:
        window_start = float(index_array.min())
        window_end = float(index_array.max())

    # Helper to retrieve numeric columns with fallback zeros
    def column_or_default(name: Optional[str]) -> np.ndarray:
        if name is None:
            return np.zeros(len(window_df), dtype=float)
        series = window_df[name]
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return numeric.to_numpy(dtype=float)

    src_nodes = window_df[src_col].astype(str).to_numpy()
    dst_nodes = window_df[dst_col].astype(str).to_numpy()
    orig_bytes = column_or_default(orig_bytes_col)
    resp_bytes = column_or_default(resp_bytes_col)
    orig_pkts = column_or_default(orig_pkts_col)
    resp_pkts = column_or_default(resp_pkts_col)
    durations = column_or_default(duration_col)

    edge_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    node_stats: Dict[str, Dict[str, float]] = {}

    labels = None
    if label_col is not None and label_col in window_df:
        labels = window_df[label_col].astype(str).str.lower().to_numpy()

    for idx, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
        key = (src, dst)
        stats = edge_stats.setdefault(
            key,
            {
                "records": 0.0,
                "bytes_total": 0.0,
                "pkts_total": 0.0,
                "duration_sum": 0.0,
            },
        )
        stats["records"] += 1.0
        stats["bytes_total"] += float(orig_bytes[idx] + resp_bytes[idx])
        stats["pkts_total"] += float(orig_pkts[idx] + resp_pkts[idx])
        stats["duration_sum"] += float(durations[idx])

        src_stats = node_stats.setdefault(
            src,
            {
                "out_bytes": 0.0,
                "out_pkts": 0.0,
                "in_bytes": 0.0,
                "in_pkts": 0.0,
                "out_degree": 0.0,
                "in_degree": 0.0,
                "partners": set(),
                "attack_records": 0.0,
                "total_records": 0.0,
            },
        )
        dst_stats = node_stats.setdefault(
            dst,
            {
                "out_bytes": 0.0,
                "out_pkts": 0.0,
                "in_bytes": 0.0,
                "in_pkts": 0.0,
                "out_degree": 0.0,
                "in_degree": 0.0,
                "partners": set(),
                "attack_records": 0.0,
                "total_records": 0.0,
            },
        )
        src_stats["out_bytes"] += float(orig_bytes[idx])
        src_stats["out_pkts"] += float(orig_pkts[idx])
        src_stats["out_degree"] += 1.0
        src_stats["partners"].add(dst)
        src_stats["total_records"] += 1.0

        dst_stats["in_bytes"] += float(resp_bytes[idx])
        dst_stats["in_pkts"] += float(resp_pkts[idx])
        dst_stats["in_degree"] += 1.0
        dst_stats["partners"].add(src)
        dst_stats["total_records"] += 1.0

        if labels is not None:
            is_attack = 1.0 if labels[idx] != "benign" else 0.0
            src_stats["attack_records"] += is_attack
            dst_stats["attack_records"] += is_attack

    if len(node_stats) < min_nodes:
        return None

    node_index = {node: idx for idx, node in enumerate(sorted(node_stats))}
    node_features: List[List[float]] = []
    for node in sorted(node_stats):
        stats = node_stats[node]
        total_records = max(stats["total_records"], 1.0)
        unique_partners = float(len(stats["partners"]))
        attack_ratio = stats["attack_records"] / total_records
        node_features.append(
            [
                stats["out_bytes"],
                stats["in_bytes"],
                stats["out_pkts"],
                stats["in_pkts"],
                stats["out_degree"],
                stats["in_degree"],
                unique_partners,
                attack_ratio,
            ]
        )

    edge_index = np.zeros((2, len(edge_stats)), dtype=np.int64)
    edge_features: List[List[float]] = []
    for col, ((src, dst), stats) in enumerate(sorted(edge_stats.items())):
        edge_index[0, col] = node_index[src]
        edge_index[1, col] = node_index[dst]
        avg_duration = stats["duration_sum"] / max(stats["records"], 1.0)
        edge_features.append(
            [
                stats["records"],
                stats["bytes_total"],
                stats["pkts_total"],
                avg_duration,
            ]
        )

    label_value = 0
    if labels is not None:
        label_value = int(any(lbl != "benign" for lbl in labels))

    data = Data()
    data.x = torch.tensor(node_features, dtype=torch.float32)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long)
    data.edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    data.y = torch.tensor([label_value], dtype=torch.long)
    data.num_nodes = data.x.size(0)
    data.window_range = torch.tensor([window_start, window_end], dtype=torch.float32)

    if src_port_col is not None and dst_port_col is not None:
        # Capture histogram style statistics to mimic TrustGuard's port buckets
        ports = window_df[[src_port_col, dst_port_col]].apply(pd.to_numeric, errors="coerce").fillna(0)
        data.port_stats = torch.tensor([
            float((ports[src_port_col] < 1024).sum()),
            float(((ports[src_port_col] >= 1024) & (ports[src_port_col] < 49152)).sum()),
            float((ports[src_port_col] >= 49152).sum()),
            float((ports[dst_port_col] < 1024).sum()),
            float(((ports[dst_port_col] >= 1024) & (ports[dst_port_col] < 49152)).sum()),
            float((ports[dst_port_col] >= 49152).sum()),
        ], dtype=torch.float32)

    return data


# ---------------------------------------------------------------------------
# Dataset splitting and normalisation utilities
# ---------------------------------------------------------------------------


def _fit_scaler(tensors: List[torch.Tensor]) -> Optional[StandardScaler]:
    if not tensors:
        return None
    stacked = torch.cat(tensors, dim=0).numpy()
    if stacked.size == 0:
        return None
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def _apply_scaler(scaler: Optional[StandardScaler], tensor: torch.Tensor) -> torch.Tensor:
    if scaler is None:
        return tensor
    transformed = scaler.transform(tensor.numpy())
    return torch.tensor(transformed, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_preprocessing(config: PreprocessConfig) -> None:
    candidates = _candidate_network_paths(config)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate Train_Test_Network.csv. Specify --network-file or place the file "
            "under data/raw/ton_iot as in the TrustGuard project."
        )

    network_path = candidates[0]
    if not network_path.is_absolute():
        network_path = network_path.resolve()
    df = _load_network_dataframe(network_path)
    timestamps, has_real_time = _extract_timestamps(df)
    df = df.copy()
    df["_egt_ts"] = timestamps
    if has_real_time:
        df = df.sort_values("_egt_ts").reset_index(drop=True)
        timestamps = df["_egt_ts"].to_numpy()

    src_col = _resolve_required_column(df, ["id.orig_h", "src_ip", "source", "source_ip"], "source IP address")
    dst_col = _resolve_required_column(df, ["id.resp_h", "dst_ip", "destination", "dest_ip"], "destination IP address")
    src_port_col = _resolve_column(df, ["id.orig_p", "src_port", "source_port", "sport"])
    dst_port_col = _resolve_column(df, ["id.resp_p", "dst_port", "destination_port", "dport"])
    duration_col = _resolve_column(df, ["duration", "flow_duration", "dur"])
    orig_bytes_col = _resolve_column(df, ["orig_bytes", "src_bytes", "bytes_sent", "bytes"])
    resp_bytes_col = _resolve_column(df, ["resp_bytes", "dst_bytes", "bytes_received"])
    orig_pkts_col = _resolve_column(df, ["orig_pkts", "src_pkts", "packets_sent"])
    resp_pkts_col = _resolve_column(df, ["resp_pkts", "dst_pkts", "packets_received"])
    label_col = _resolve_column(df, ["label", "detailed-label", "class", "attack" ])

    windows: List[Data] = []
    node_features_to_scale: List[torch.Tensor] = []
    edge_features_to_scale: List[torch.Tensor] = []

    for time_anchor, indices in _iter_time_windows(
        timestamps,
        has_real_time,
        config.window_size,
        config.stride,
        config.min_rows_per_window,
    ):
        graph = _aggregate_window(
            df,
            indices,
            float(time_anchor),
            has_real_time,
            src_col,
            dst_col,
            src_port_col,
            dst_port_col,
            duration_col,
            orig_bytes_col,
            resp_bytes_col,
            orig_pkts_col,
            resp_pkts_col,
            label_col,
            config.min_nodes,
        )
        if graph is not None:
            windows.append(graph)
            node_features_to_scale.append(graph.x)
            if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                edge_features_to_scale.append(graph.edge_attr)

    if not windows:
        raise RuntimeError(
            "No valid graphs were generated from the provided CSV. Try decreasing --min-rows-per-window or --min-nodes."
        )

    # Normalise node and edge features using global scalers, as done in TrustGuard
    node_scaler = _fit_scaler(node_features_to_scale)
    edge_scaler = _fit_scaler(edge_features_to_scale)
    for graph in windows:
        graph.x = _apply_scaler(node_scaler, graph.x)
        if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            graph.edge_attr = _apply_scaler(edge_scaler, graph.edge_attr)

    # Sort by the recorded window start to enforce temporal splits
    windows.sort(key=lambda data: float(data.window_range[0]))

    num_graphs = len(windows)
    train_count = max(int(num_graphs * config.train_ratio), 1)
    val_count = max(int(num_graphs * config.val_ratio), 0)
    test_count = num_graphs - train_count - val_count
    if test_count <= 0:
        test_count = max(1, num_graphs - train_count)
        val_count = num_graphs - train_count - test_count

    if min(train_count, val_count, test_count) < config.min_graphs_per_split:
        raise RuntimeError(
            "TON_IoT preprocessing produced insufficient graphs. Increase observation time, reduce --min-rows-per-window, "
            "or lower --min-graphs-per-split."
        )

    train_graphs = windows[:train_count]
    val_graphs = windows[train_count : train_count + val_count]
    test_graphs = windows[train_count + val_count :]

    def label_stats(graphs: Sequence[Data]) -> Dict[str, int]:
        positives = sum(int(graph.y.item()) for graph in graphs)
        return {"pos": positives, "neg": len(graphs) - positives}

    dataset_dir = config.output_root / "toni_iot"
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_features": int(train_graphs[0].x.size(-1)) if train_graphs else 0,
        "num_edge_features": int(train_graphs[0].edge_attr.size(-1)) if train_graphs and train_graphs[0].edge_attr is not None else 0,
        "num_classes": 2,
        "scalers": {
            "node_mean": node_scaler.mean_.tolist() if node_scaler is not None else None,
            "node_scale": node_scaler.scale_.tolist() if node_scaler is not None else None,
            "edge_mean": edge_scaler.mean_.tolist() if edge_scaler is not None else None,
            "edge_scale": edge_scaler.scale_.tolist() if edge_scaler is not None else None,
        },
        "window_settings": {
            "window_size": config.window_size,
            "stride": config.stride,
            "min_rows_per_window": config.min_rows_per_window,
            "min_nodes": config.min_nodes,
        },
        "source_csv": str(network_path),
    }

    def save_split(name: str, graphs: Sequence[Data]) -> None:
        torch.save({"data_list": list(graphs), "metadata": metadata}, processed_dir / f"{name}_graph.pt")

    save_split("train", train_graphs)
    save_split("val", val_graphs)
    save_split("test", test_graphs)

    summary = {
        "num_graphs": num_graphs,
        "split_sizes": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
        "label_distribution": {
            "train": label_stats(train_graphs),
            "val": label_stats(val_graphs),
            "test": label_stats(test_graphs),
        },
        "config": {
            "window_size": config.window_size,
            "stride": config.stride,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "min_rows_per_window": config.min_rows_per_window,
            "min_nodes": config.min_nodes,
            "min_graphs_per_split": config.min_graphs_per_split,
        },
    }
    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("Preprocessing complete:")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    config = parse_args()
    run_preprocessing(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
