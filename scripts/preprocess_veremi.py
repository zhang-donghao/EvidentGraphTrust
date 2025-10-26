"""Preprocess VeReMi dataset into PyG graphs for Evident Graph Trust."""
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


DEFAULT_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "timestamp": ("time", "timestamp", "Time", "ts"),
    "vehicle_id": ("vehicle", "stationID", "veh_id", "vehicle_id"),
    "pos_x": ("pos_x", "x", "posX", "east", "longitude"),
    "pos_y": ("pos_y", "y", "posY", "north", "latitude"),
    "speed": ("speed", "speed_mps", "v"),
    "acc": ("acc", "acceleration", "a"),
    "heading": ("heading", "yaw", "course"),
    "attack": ("attack", "label", "is_attack", "isAttacker", "malicious"),
}


@dataclass
class VeReMiConfig:
    raw_root: Path
    output_root: Path
    window_size: float = 1.0
    stride: float = 0.5
    distance_threshold: float = 120.0
    rbf_length_scale: float = 40.0
    min_nodes: int = 3
    seed: int = 7


def parse_args() -> VeReMiConfig:
    parser = argparse.ArgumentParser(description="Preprocess VeReMi dataset")
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to directory containing converted VeReMi CSV files")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to store processed dataset")
    parser.add_argument("--window-size", type=float, default=1.0, help="Sliding window size in seconds")
    parser.add_argument("--stride", type=float, default=0.5, help="Sliding window stride in seconds")
    parser.add_argument("--distance-threshold", type=float, default=120.0, help="Max distance (meters) to connect vehicles")
    parser.add_argument("--rbf-length-scale", type=float, default=40.0, help="Length scale for RBF edge weighting")
    parser.add_argument("--min-nodes", type=int, default=3, help="Minimum number of vehicles per graph")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val/test splits")
    args = parser.parse_args()
    return VeReMiConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        distance_threshold=args.distance_threshold,
        rbf_length_scale=args.rbf_length_scale,
        min_nodes=args.min_nodes,
        seed=args.seed,
    )


def _select_column(df: pd.DataFrame, candidates: Sequence[str], required: bool, fallback: Optional[str] = None) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    if fallback is not None:
        return fallback
    if required:
        raise KeyError(f"Required column not found. Expected one of: {candidates}. Available: {list(df.columns)}")
    return ""


def _load_frames(config: VeReMiConfig) -> pd.DataFrame:
    csv_files = sorted(config.raw_root.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {config.raw_root}. Please convert VeReMi logs first.")
    frames = []
    for csv_file in tqdm(csv_files, desc="Loading VeReMi CSV"):
        frame = pd.read_csv(csv_file)
        frame["__scenario"] = csv_file.stem
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    col_map: Dict[str, str] = {}
    for key, aliases in DEFAULT_COLUMN_ALIASES.items():
        col_map[key] = _select_column(df, aliases, required=(key in {"timestamp", "vehicle_id", "pos_x", "pos_y"}))

    df = df.rename(columns={col_map[k]: k for k in col_map if col_map[k]})
    # Normalise timestamp to seconds
    if np.issubdtype(df["timestamp"].dtype, np.number):
        timestamps = df["timestamp"].astype(float)
    else:
        timestamps = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
    df["timestamp"] = timestamps
    df = df.sort_values("timestamp").reset_index(drop=True)

    for optional in ("speed", "acc", "heading", "attack"):
        if optional not in df.columns:
            if optional == "attack":
                df[optional] = 0
            else:
                df[optional] = np.nan
    return df


def _build_edges(positions: np.ndarray, distance_threshold: float, rbf_length_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(positions) < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = (dist > 0.0) & (dist <= distance_threshold)
    sources, targets = np.where(mask)
    if sources.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    weights = np.exp(-np.square(dist[sources, targets]) / (2 * (rbf_length_scale ** 2)))
    # Make graph undirected
    edge_index = np.vstack([np.concatenate([sources, targets]), np.concatenate([targets, sources])])
    edge_weight = np.concatenate([weights, weights]).astype(np.float32)
    return edge_index.astype(np.int64), edge_weight


def _window_iterator(df: pd.DataFrame, window: float, stride: float) -> Iterable[pd.DataFrame]:
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    current = start
    while current + window <= end:
        mask = (df["timestamp"] >= current) & (df["timestamp"] < current + window)
        window_df = df.loc[mask]
        if not window_df.empty:
            yield window_df
        current += stride


def _extract_graphs(df: pd.DataFrame, config: VeReMiConfig) -> Tuple[List[Data], Dict[str, float]]:
    graphs: List[Data] = []
    feature_cache: List[np.ndarray] = []
    prev_trust: Dict[str, float] = {}
    feature_names = ["speed", "acc", "heading", "pos_x", "pos_y", "message_count", "prev_trust"]

    for window_df in tqdm(list(_window_iterator(df, config.window_size, config.stride)), desc="Building graphs"):
        grouped = window_df.groupby("vehicle_id")
        node_ids: List[str] = []
        features: List[List[float]] = []
        labels: List[int] = []
        positions: List[List[float]] = []
        message_counts: Dict[str, int] = grouped.size().to_dict()
        for vehicle_id, group in grouped:
            group = group.sort_values("timestamp")
            node_ids.append(str(vehicle_id))
            last = group.iloc[-1]
            speed = group["speed"].astype(float).mean() if "speed" in group else np.nan
            acc = group["acc"].astype(float).mean() if "acc" in group else np.nan
            heading = group["heading"].astype(float).mean() if "heading" in group else np.nan
            pos_x = float(last["pos_x"])
            pos_y = float(last["pos_y"])
            msg_count = float(message_counts.get(vehicle_id, len(group)))
            trust_prev = prev_trust.get(str(vehicle_id), 0.5)
            features.append([speed, acc, heading, pos_x, pos_y, msg_count, trust_prev])
            positions.append([pos_x, pos_y])
            attack_value = int(group["attack"].astype(float).max()) if "attack" in group else 0
            labels.append(attack_value)

        if len(features) < config.min_nodes:
            continue

        feature_arr = np.array(features, dtype=np.float64)
        feature_cache.append(feature_arr)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        positions_arr = np.array(positions, dtype=np.float64)
        edge_index, edge_weight = _build_edges(positions_arr, config.distance_threshold, config.rbf_length_scale)

        data = Data(
            x=torch.from_numpy(feature_arr.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index),
            edge_weight=torch.from_numpy(edge_weight),
            y=labels_tensor,
        )
        data.node_ids = node_ids
        data.message_counts = torch.tensor([message_counts[nid] for nid in node_ids], dtype=torch.float32)
        data.window_start = float(window_df["timestamp"].min())
        data.window_end = float(window_df["timestamp"].max())
        graphs.append(data)

        for node_id, label in zip(node_ids, labels):
            prev_trust[node_id] = 1.0 - 0.5 * label

    if not graphs:
        raise RuntimeError("No graphs generated. Check preprocessing parameters or raw data integrity.")

    stacked_features = np.vstack(feature_cache)
    scaler = StandardScaler().fit(stacked_features)

    for data in graphs:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy()).astype(np.float32))

    metadata = {
        "feature_names": feature_names,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "num_features": len(feature_names),
        "num_classes": 2,
        "window_size": config.window_size,
        "stride": config.stride,
    }
    return graphs, metadata


def _safe_stratified_split(indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
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
    train_idx, temp_idx = _safe_stratified_split(indices, graph_labels, test_size=0.3, seed=seed)
    temp_labels = graph_labels[temp_idx]
    val_idx, test_idx = _safe_stratified_split(temp_idx, temp_labels, test_size=0.5, seed=seed)
    split = {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }
    return split


def _save_split(split_graphs: Dict[str, List[Data]], metadata: Dict[str, object], output_root: Path) -> None:
    for name, data_list in split_graphs.items():
        processed_dir = output_root / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        path = processed_dir / f"{name}_graph.pt"
        torch.save({"data_list": data_list, "metadata": metadata, "split": name}, path)

    summary = {
        name: {
            "num_graphs": len(graphs),
            "avg_nodes": float(np.mean([g.num_nodes for g in graphs])) if graphs else 0.0,
            "attack_ratio": float(np.mean([g.y.float().mean().item() for g in graphs])) if graphs else 0.0,
        }
        for name, graphs in split_graphs.items()
    }
    with open(output_root / "processed" / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    config = parse_args()
    df = _load_frames(config)
    graphs, metadata = _extract_graphs(df, config)
    split = _split_graphs(graphs, config.seed)
    dataset_root = config.output_root / "veremi"
    _save_split(split, metadata, dataset_root)
    print(f"Saved VeReMi graphs to {dataset_root / 'processed'}")


if __name__ == "__main__":
    main()
