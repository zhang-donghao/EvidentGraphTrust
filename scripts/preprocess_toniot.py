"""Preprocess TON_IoT dataset into PyG graphs for Evident Graph Trust."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm


TELEMETRY_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "timestamp": ("ts", "timestamp", "time", "date"),
    "device_id": ("device", "device_id", "source_id", "node_id"),
    "value": ("value", "reading", "metric", "measure"),
}

NETWORK_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "src": ("src_device", "src_ip", "source_id", "src"),
    "dst": ("dst_device", "dst_ip", "destination_id", "dst"),
    "protocol": ("protocol", "proto"),
}

EXCLUDED_TELEMETRY_COLUMNS = {
    "timestamp",
    "device_id",
    "label",
    "attack",
    "class",
    "target",
    "is_anomaly",
    "category",
}


@dataclass
class TONTConfig:
    raw_root: Path
    output_root: Path
    window_size: float = 300.0
    stride: float = 120.0
    min_nodes: int = 4
    seed: int = 42


def parse_args() -> TONTConfig:
    parser = argparse.ArgumentParser(description="Preprocess TON_IoT dataset")
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing TON_IoT CSV files")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to store processed dataset")
    parser.add_argument("--window-size", type=float, default=300.0, help="Window length (seconds) for telemetry aggregation")
    parser.add_argument("--stride", type=float, default=120.0, help="Sliding window stride in seconds")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum number of devices per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits")
    args = parser.parse_args()
    return TONTConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        min_nodes=args.min_nodes,
        seed=args.seed,
    )


def _select_column(df: pd.DataFrame, aliases: Sequence[str], required: bool, default: Optional[str] = None) -> str:
    for name in aliases:
        if name in df.columns:
            return name
    if default is not None:
        return default
    if required:
        raise KeyError(f"Missing required column. Expected one of {aliases}, got {list(df.columns)}")
    return ""


def _load_csvs(root: Path, pattern: str) -> List[pd.DataFrame]:
    files = sorted(root.glob(pattern))
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except pd.errors.EmptyDataError:
            continue
    return frames


def _load_telemetry(config: TONTConfig) -> pd.DataFrame:
    telemetry_frames = _load_csvs(config.raw_root, "**/*Telemetry*.csv")
    if not telemetry_frames:
        telemetry_frames = _load_csvs(config.raw_root, "**/*telemetry*.csv")
    if not telemetry_frames:
        raise FileNotFoundError(f"No telemetry CSV files found under {config.raw_root}")
    df = pd.concat(telemetry_frames, ignore_index=True)

    col_map = {
        "timestamp": _select_column(df, TELEMETRY_COLUMN_ALIASES["timestamp"], required=True),
        "device_id": _select_column(df, TELEMETRY_COLUMN_ALIASES["device_id"], required=True),
    }
    df = df.rename(columns={col_map["timestamp"]: "timestamp", col_map["device_id"]: "device_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
    for metric_col in df.columns:
        if metric_col not in EXCLUDED_TELEMETRY_COLUMNS:
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["device_id", "timestamp"])
    df["device_id"] = df["device_id"].astype(str)
    return df


def _load_network(config: TONTConfig) -> pd.DataFrame:
    network_frames = _load_csvs(config.raw_root, "**/*Network*.csv")
    if not network_frames:
        network_frames = _load_csvs(config.raw_root, "**/*network*.csv")
    if not network_frames:
        return pd.DataFrame(columns=["src", "dst", "protocol"])
    df = pd.concat(network_frames, ignore_index=True)
    col_map = {}
    for key, aliases in NETWORK_COLUMN_ALIASES.items():
        col_map[key] = _select_column(df, aliases, required=(key in {"src", "dst"}))
    df = df.rename(columns={col_map[k]: k for k in col_map if col_map[k]})
    df = df[[col for col in ("src", "dst", "protocol") if col in df.columns]]
    for col in ("src", "dst"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _window_iterator(df: pd.DataFrame, window: float, stride: float) -> Iterable[pd.DataFrame]:
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    current = start
    while current + window <= end:
        mask = (df["timestamp"] >= current) & (df["timestamp"] < current + window)
        chunk = df.loc[mask]
        if not chunk.empty:
            yield chunk
        current += stride


def _build_edges(network_df: pd.DataFrame, devices: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if network_df.empty or len(devices) < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    id_to_idx = {device: idx for idx, device in enumerate(devices)}
    counts: Dict[Tuple[int, int], int] = {}
    for _, row in network_df.iterrows():
        src = row.get("src")
        dst = row.get("dst")
        if pd.isna(src) or pd.isna(dst):
            continue
        if src not in id_to_idx or dst not in id_to_idx:
            continue
        src_idx, dst_idx = id_to_idx[src], id_to_idx[dst]
        counts[(src_idx, dst_idx)] = counts.get((src_idx, dst_idx), 0) + 1
        counts[(dst_idx, src_idx)] = counts.get((dst_idx, src_idx), 0) + 1
    if not counts:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    edges = np.array(list(counts.keys()), dtype=np.int64).T
    weights = np.array(list(counts.values()), dtype=np.float32)
    # Normalise weights
    if weights.size:
        weights = weights / weights.max()
    return edges, weights


def _aggregate_features(chunk: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    metrics = [col for col in chunk.columns if col not in EXCLUDED_TELEMETRY_COLUMNS]
    grouped = chunk.groupby("device_id")
    device_ids: List[str] = []
    feature_list: List[np.ndarray] = []
    for device_id, group in grouped:
        device_id_str = str(device_id)
        device_ids.append(device_id_str)
        stats = []
        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            if values.size == 0:
                stats.extend([0.0, 0.0, 0.0])
            else:
                stats.extend([values.mean(), values.std(ddof=0), values.max()])
        stats.append(float(len(group)))
        feature_list.append(np.array(stats, dtype=np.float32))
    if not feature_list:
        return np.empty((0, 0), dtype=np.float32), [], []
    features = np.stack(feature_list)
    feature_names: List[str] = []
    for metric in metrics:
        feature_names.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_max"])
    feature_names.append("message_count")
    return features, device_ids, feature_names


def _aggregate_labels(chunk: pd.DataFrame) -> Dict[str, int]:
    label_col = None
    for candidate in ("label", "attack", "target", "is_anomaly", "class"):
        if candidate in chunk.columns:
            label_col = candidate
            break
    if label_col is None:
        return {device: 0 for device in chunk["device_id"].unique()}
    grouped = chunk.groupby("device_id")[label_col]
    normal_tokens = {"normal", "benign", "0", "false", "no"}
    labels: Dict[str, int] = {}
    for device, series in grouped:
        values = series.dropna()
        label = 0
        if not values.empty:
            if values.dtype == object:
                lower = values.astype(str).str.lower()
                label = int(any(token not in normal_tokens for token in lower))
            else:
                numeric = pd.to_numeric(values, errors="coerce").fillna(0)
                label = int((numeric > 0).any())
        labels[str(device)] = label
    return labels


def _extract_graphs(telemetry_df: pd.DataFrame, network_df: pd.DataFrame, config: TONTConfig) -> Tuple[List[Data], Dict[str, object]]:
    graphs: List[Data] = []
    feature_cache: List[np.ndarray] = []
    metadata_feature_names: Optional[List[str]] = None
    network_df = network_df.copy()
    for chunk in tqdm(list(_window_iterator(telemetry_df, config.window_size, config.stride)), desc="Processing windows"):
        features, device_ids, feature_names = _aggregate_features(chunk)
        if features.size == 0 or len(device_ids) < config.min_nodes:
            continue
        labels_map = _aggregate_labels(chunk)
        labels = torch.tensor([labels_map.get(device, 0) for device in device_ids], dtype=torch.long)
        window_network = network_df[network_df["src"].isin(device_ids) & network_df["dst"].isin(device_ids)] if not network_df.empty else network_df
        edge_index, edge_weight = _build_edges(window_network, device_ids)
        data = Data(
            x=torch.from_numpy(features.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index),
            edge_weight=torch.from_numpy(edge_weight),
            y=labels,
        )
        data.device_ids = device_ids
        data.window_start = float(chunk["timestamp"].min())
        data.window_end = float(chunk["timestamp"].max())
        graphs.append(data)
        feature_cache.append(features)
        if metadata_feature_names is None:
            metadata_feature_names = feature_names

    if not graphs:
        raise RuntimeError("No graphs generated from TON_IoT telemetry. Check raw files and parameters.")

    scaler = StandardScaler().fit(np.vstack(feature_cache))
    for data in graphs:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy()).astype(np.float32))

    metadata = {
        "feature_names": metadata_feature_names or [],
        "num_features": int(graphs[0].x.size(-1)),
        "num_classes": 2,
        "window_size": config.window_size,
        "stride": config.stride,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    }
    return graphs, metadata


def _safe_split(indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        first, second = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
    except ValueError:
        first, second = train_test_split(indices, test_size=test_size, random_state=seed, stratify=None)
    return np.array(first), np.array(second)


def _split_graphs(graphs: List[Data], seed: int) -> Dict[str, List[Data]]:
    indices = np.arange(len(graphs))
    graph_labels = np.array([int(data.y.sum().item() > 0) for data in graphs])
    train_idx, temp_idx = _safe_split(indices, graph_labels, test_size=0.3, seed=seed)
    temp_labels = graph_labels[temp_idx]
    val_idx, test_idx = _safe_split(temp_idx, temp_labels, test_size=0.5, seed=seed)
    return {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }


def _save_split(split: Dict[str, List[Data]], metadata: Dict[str, object], output_root: Path) -> None:
    processed_dir = output_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, graphs in split.items():
        torch.save({"data_list": graphs, "metadata": metadata, "split": name}, processed_dir / f"{name}_graph.pt")
    summary = {
        name: {
            "num_graphs": len(graphs),
            "avg_nodes": float(np.mean([g.num_nodes for g in graphs])) if graphs else 0.0,
            "attack_ratio": float(np.mean([g.y.float().mean().item() for g in graphs])) if graphs else 0.0,
        }
        for name, graphs in split.items()
    }
    with open(processed_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    config = parse_args()
    telemetry_df = _load_telemetry(config)
    network_df = _load_network(config)
    graphs, metadata = _extract_graphs(telemetry_df, network_df, config)
    split = _split_graphs(graphs, config.seed)
    dataset_root = config.output_root / "toni_iot"
    _save_split(split, metadata, dataset_root)
    print(f"Saved TON_IoT graphs to {dataset_root / 'processed'}")


if __name__ == "__main__":
    main()
