"""Preprocess TON_IoT dataset into PyG graphs for Evident Graph Trust."""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


TELEMETRY_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "timestamp": ("ts", "timestamp", "time", "date"),
    "device_id": ("device", "device_id", "source_id", "node_id"),
    "value": ("value", "reading", "metric", "measure"),
}

NETWORK_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "src": ("src_device", "src_ip", "source_id", "src"),
    "dst": ("dst_device", "dst_ip", "destination_id", "dst"),
    "protocol": ("protocol", "proto"),
    "timestamp": ("ts", "timestamp", "time", "date", "datetime"),
    "label": ("label", "attack", "target", "is_anomaly", "class"),
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

EXCLUDED_NETWORK_FEATURE_COLUMNS = {
    "src",
    "dst",
    "protocol",
    "timestamp",
    "label",
}


@dataclass
class TONTConfig:
    raw_root: Path
    output_root: Path
    window_size: float = 300.0
    stride: float = 120.0
    min_nodes: int = 4
    seed: int = 42
    network_file: Optional[Path] = None


def parse_args() -> TONTConfig:
    default_raw_root = REPO_ROOT / "src" / "data"
    parser = argparse.ArgumentParser(description="Preprocess TON_IoT dataset")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw_root,
        help="Directory containing TON_IoT CSV files (defaults to repo src/data)",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to store processed dataset")
    parser.add_argument("--window-size", type=float, default=300.0, help="Window length (seconds) for telemetry aggregation")
    parser.add_argument("--stride", type=float, default=120.0, help="Sliding window stride in seconds")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum number of devices per graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits")
    parser.add_argument(
        "--network-file",
        type=Path,
        default=None,
        help="Explicit path to Train_Test_Network.csv (defaults to <raw-root>/train_test_network.csv if present)",
    )
    args = parser.parse_args()
    raw_root = args.raw_root.expanduser()
    if not raw_root.exists() and default_raw_root.exists():
        warnings.warn(
            f"Specified raw root {raw_root} does not exist. Falling back to {default_raw_root}.",
            RuntimeWarning,
        )
        raw_root = default_raw_root
    network_file: Optional[Path] = args.network_file
    if network_file is None:
        candidates = [
            raw_root / "train_test_network.csv",
            raw_root / "Train_Test_Network.csv",
        ]
        if raw_root != default_raw_root:
            candidates.extend(
                [
                    default_raw_root / "train_test_network.csv",
                    default_raw_root / "Train_Test_Network.csv",
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                network_file = candidate
                break
        else:
            if raw_root.is_file() and raw_root.suffix.lower() == ".csv":
                network_file = raw_root
                raw_root = raw_root.parent
    return TONTConfig(
        raw_root=raw_root,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        min_nodes=args.min_nodes,
        seed=args.seed,
        network_file=network_file,
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


def _compute_stats(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0)), float(values.max())


def _reduce_label_series(series: pd.Series, normal_tokens: Optional[Sequence[str]] = None) -> int:
    values = series.dropna()
    if values.empty:
        return 0
    if values.dtype == object:
        normal = {"normal", "benign", "0", "false", "no"}
        if normal_tokens is not None:
            normal = set(normal_tokens)
        lower = values.astype(str).str.lower()
        return int(any(token not in normal for token in lower))
    numeric = pd.to_numeric(values, errors="coerce").fillna(0)
    return int((numeric > 0).any())


def _load_csvs(root: Path, pattern: str) -> List[pd.DataFrame]:
    if root.is_file():
        try:
            return [pd.read_csv(root)]
        except pd.errors.EmptyDataError:
            return []
    if not root.exists():
        return []
    files = sorted(root.glob(pattern))
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except pd.errors.EmptyDataError:
            continue
    return frames


def _load_telemetry(config: TONTConfig) -> Optional[pd.DataFrame]:
    telemetry_frames = _load_csvs(config.raw_root, "**/*Telemetry*.csv")
    if not telemetry_frames:
        telemetry_frames = _load_csvs(config.raw_root, "**/*telemetry*.csv")
    if not telemetry_frames:
        warnings.warn(
            "No telemetry CSV files were found. Falling back to network-only preprocessing.",
            RuntimeWarning,
        )
        return None
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
    network_frames: List[pd.DataFrame] = []
    seen_paths = set()

    def _append_from_file(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen_paths or not path.exists():
            return
        try:
            network_frames.append(pd.read_csv(path))
            seen_paths.add(resolved)
        except pd.errors.EmptyDataError:
            return

    if config.network_file is not None:
        if config.network_file.exists():
            _append_from_file(config.network_file)
        else:
            warnings.warn(
                f"Specified network file {config.network_file} does not exist; falling back to directory scan.",
                RuntimeWarning,
            )

    if config.raw_root.exists():
        for pattern in ("**/*Network*.csv", "**/*network*.csv"):
            for file in sorted(config.raw_root.glob(pattern)):
                _append_from_file(file)
            if network_frames:
                break
    if not network_frames:
        return pd.DataFrame(columns=["src", "dst", "protocol"])
    df = pd.concat(network_frames, ignore_index=True)
    col_map: Dict[str, str] = {}
    for key, aliases in NETWORK_COLUMN_ALIASES.items():
        required = key in {"src", "dst"}
        try:
            col_map[key] = _select_column(df, aliases, required=required)
        except KeyError:
            if required:
                raise
            col_map[key] = ""
    rename_map = {col_map[k]: k for k in col_map if col_map[k]}
    df = df.rename(columns=rename_map)
    for col in ("src", "dst"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "timestamp" in df.columns:
        if not np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("Int64") / 1e9
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = np.nan
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].astype(str)

    for column in df.columns:
        if column in EXCLUDED_NETWORK_FEATURE_COLUMNS:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _row_fallback_windows(df: pd.DataFrame, min_chunk_size: int) -> Iterable[pd.DataFrame]:
    if df.empty:
        yield df
        return
    if min_chunk_size <= 0:
        min_chunk_size = 1
    chunk_count = max(1, min(10, len(df) // max(1, min_chunk_size)))
    if chunk_count <= 1 and len(df) >= min_chunk_size * 2:
        chunk_count = 2
    if chunk_count <= 2 and len(df) >= min_chunk_size * 3:
        chunk_count = 3
    if chunk_count <= 1:
        yield df
        return
    for chunk in np.array_split(df.sort_index(), chunk_count):
        if not chunk.empty:
            yield chunk


def _window_iterator(
    df: pd.DataFrame, window: float, stride: float, min_chunk_size: int
) -> Iterable[pd.DataFrame]:
    if "timestamp" not in df.columns:
        yield from _row_fallback_windows(df, min_chunk_size)
        return
    timestamp_series = pd.to_numeric(df["timestamp"], errors="coerce")
    valid = timestamp_series.dropna()
    if valid.empty:
        yield from _row_fallback_windows(df, min_chunk_size)
        return
    start = float(valid.min())
    end = float(valid.max())
    if not np.isfinite(start) or not np.isfinite(end) or end - start < window:
        yield from _row_fallback_windows(df, min_chunk_size)
        return
    current = start
    while current <= end:
        mask = (timestamp_series >= current) & (timestamp_series < current + window)
        chunk = df.loc[mask]
        if not chunk.empty:
            yield chunk
        current += stride
        if stride <= 0:
            break


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
            mean, std, max_val = _compute_stats(values)
            stats.extend([mean, std, max_val])
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
        labels[str(device)] = _reduce_label_series(series, normal_tokens)
    return labels


def _infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []
    for column in df.columns:
        if column in EXCLUDED_NETWORK_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_cols.append(column)
    return numeric_cols


def _aggregate_network_features(
    chunk: pd.DataFrame, numeric_columns: Sequence[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    device_series: List[pd.Series] = []
    if "src" in chunk.columns:
        device_series.append(chunk["src"].astype(str))
    if "dst" in chunk.columns:
        device_series.append(chunk["dst"].astype(str))
    if not device_series:
        return np.empty((0, 0), dtype=np.float32), [], []
    devices_array = pd.unique(pd.concat(device_series, ignore_index=True).dropna()).astype(str)
    device_ids: List[str] = devices_array.tolist()
    feature_list: List[np.ndarray] = []
    feature_names: List[str] = ["outgoing_count", "incoming_count"]
    for column in numeric_columns:
        feature_names.extend(
            [
                f"out_{column}_mean",
                f"out_{column}_std",
                f"out_{column}_max",
                f"in_{column}_mean",
                f"in_{column}_std",
                f"in_{column}_max",
            ]
        )
    empty_frame = chunk.iloc[0:0]
    for device_str in device_ids:
        outgoing = chunk[chunk["src"] == device_str] if "src" in chunk.columns else empty_frame
        incoming = chunk[chunk["dst"] == device_str] if "dst" in chunk.columns else empty_frame
        stats: List[float] = [float(len(outgoing)), float(len(incoming))]
        for column in numeric_columns:
            out_values = outgoing[column].dropna().to_numpy() if column in outgoing.columns else np.array([])
            in_values = incoming[column].dropna().to_numpy() if column in incoming.columns else np.array([])
            stats.extend(_compute_stats(out_values))
            stats.extend(_compute_stats(in_values))
        feature_list.append(np.array(stats, dtype=np.float32))
    if not feature_list:
        return np.empty((0, 0), dtype=np.float32), [], []
    return np.stack(feature_list), device_ids, feature_names


def _aggregate_network_labels(chunk: pd.DataFrame) -> Dict[str, int]:
    label_col = None
    for candidate in ("label", "attack", "target", "is_anomaly", "class"):
        if candidate in chunk.columns:
            label_col = candidate
            break
    device_series: List[pd.Series] = []
    if "src" in chunk.columns:
        device_series.append(chunk["src"].astype(str))
    if "dst" in chunk.columns:
        device_series.append(chunk["dst"].astype(str))
    devices: List[str] = []
    if device_series:
        devices = pd.unique(pd.concat(device_series, ignore_index=True).dropna()).astype(str).tolist()
    labels: Dict[str, int] = {device: 0 for device in devices}
    if label_col is None:
        return labels
    for role in ("src", "dst"):
        if role not in chunk.columns:
            continue
        grouped = chunk.groupby(role)[label_col]
        for device, series in grouped:
            device_str = str(device)
            labels[device_str] = max(labels.get(device_str, 0), _reduce_label_series(series))
    return labels


def _extract_graphs_from_network(
    network_df: pd.DataFrame, config: TONTConfig
) -> Tuple[List[Data], Dict[str, object]]:
    if network_df.empty:
        searched_hint = (
            f"explicit file {config.network_file}" if config.network_file else "default search locations"
        )
        raise RuntimeError(
            "Network CSV files were not found. Provide Train_Test_Network.csv or matching files to continue. "
            f"Checked {searched_hint} within {config.raw_root}."
        )
    numeric_columns = _infer_numeric_columns(network_df)
    graphs: List[Data] = []
    feature_cache: List[np.ndarray] = []
    metadata_feature_names: Optional[List[str]] = None
    for chunk in tqdm(
        list(_window_iterator(network_df, config.window_size, config.stride, config.min_nodes)),
        desc="Processing windows",
    ):
        features, device_ids, feature_names = _aggregate_network_features(chunk, numeric_columns)
        if features.size == 0 or len(device_ids) < config.min_nodes:
            continue
        labels_map = _aggregate_network_labels(chunk)
        labels = torch.tensor([labels_map.get(device, 0) for device in device_ids], dtype=torch.long)
        window_network = chunk if not chunk.empty else network_df
        edge_index, edge_weight = _build_edges(window_network, list(device_ids))
        data = Data(
            x=torch.from_numpy(features.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index),
            edge_weight=torch.from_numpy(edge_weight),
            y=labels,
        )
        data.device_ids = list(device_ids)
        if "timestamp" in chunk.columns and not chunk["timestamp"].dropna().empty:
            data.window_start = float(np.nanmin(chunk["timestamp"].to_numpy()))
            data.window_end = float(np.nanmax(chunk["timestamp"].to_numpy()))
        graphs.append(data)
        feature_cache.append(features)
        if metadata_feature_names is None:
            metadata_feature_names = feature_names

    if not graphs:
        raise RuntimeError(
            "Network-only preprocessing did not yield any graphs. Check that the CSV contains src/dst columns and adjust"
            " window parameters."
        )

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
        "source_modalities": ["network"],
    }
    return graphs, metadata


def _extract_graphs_from_telemetry(
    telemetry_df: pd.DataFrame, network_df: pd.DataFrame, config: TONTConfig
) -> Tuple[List[Data], Dict[str, object]]:
    graphs: List[Data] = []
    feature_cache: List[np.ndarray] = []
    metadata_feature_names: Optional[List[str]] = None
    network_df = network_df.copy()
    for chunk in tqdm(
        list(_window_iterator(telemetry_df, config.window_size, config.stride, config.min_nodes)),
        desc="Processing windows",
    ):
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
        "source_modalities": ["telemetry"] + (["network"] if not network_df.empty else []),
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
        if indices.size <= 1:
            return indices.astype(int), np.empty(0, dtype=int)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        second_count = int(round(len(shuffled) * test_size))
        second_count = max(1, second_count)
        if second_count >= len(shuffled):
            second_count = len(shuffled) - 1
        second = shuffled[:second_count]
        first = shuffled[second_count:]
    return np.array(first, dtype=int), np.array(second, dtype=int)


def _split_graphs(graphs: List[Data], seed: int) -> Tuple[Dict[str, List[Data]], Dict[str, object]]:
    indices = np.arange(len(graphs))
    graph_labels = np.array([int(data.y.sum().item() > 0) for data in graphs])
    train_idx, temp_idx = _safe_split(indices, graph_labels, test_size=0.3, seed=seed)
    temp_labels = graph_labels[temp_idx]
    val_idx, test_idx = _safe_split(temp_idx, temp_labels, test_size=0.5, seed=seed)
    split: Dict[str, List[Data]] = {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }

    diagnostics: Dict[str, object] = {}

    if not graphs:
        return split, diagnostics

    empty_splits = [name for name, items in split.items() if len(items) == 0]
    if empty_splits:
        redistribution_log: List[Tuple[str, str]] = []
        for target in empty_splits:
            donors = sorted(
                [name for name, items in split.items() if len(items) > 1 and name != target],
                key=lambda name: len(split[name]),
                reverse=True,
            )
            moved = False
            for donor in donors:
                if split[donor]:
                    split[target].append(split[donor].pop())
                    redistribution_log.append((donor, target))
                    moved = True
                    break
            if not moved:
                continue
        empty_splits = [name for name, items in split.items() if len(items) == 0]
        if empty_splits:
            diagnostics["empty_splits"] = empty_splits
            diagnostics["total_graphs"] = len(graphs)
        elif redistribution_log:
            diagnostics["redistribution"] = redistribution_log

    return split, diagnostics


def _candidate_configs(config: TONTConfig) -> Iterable[TONTConfig]:
    scales = [1.0, 0.5, 0.25, 0.125]
    seen: set[Tuple[float, float]] = set()
    for scale in scales:
        window = max(30.0, float(config.window_size * scale))
        stride = max(10.0, float(config.stride * scale))
        if stride > window:
            stride = window
        key = (round(window, 4), round(stride, 4))
        if key in seen:
            continue
        seen.add(key)
        yield replace(config, window_size=window, stride=stride)


def _generate_graphs_with_split(
    telemetry_df: Optional[pd.DataFrame],
    network_df: pd.DataFrame,
    config: TONTConfig,
) -> Tuple[List[Data], Dict[str, object], Dict[str, List[Data]], Dict[str, object], TONTConfig, List[Dict[str, object]]]:
    attempts: List[Dict[str, object]] = []
    last_error: Optional[str] = None
    for candidate in _candidate_configs(config):
        try:
            if telemetry_df is None:
                graphs, metadata = _extract_graphs_from_network(network_df, candidate)
            else:
                graphs, metadata = _extract_graphs_from_telemetry(telemetry_df, network_df, candidate)
        except RuntimeError as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "window_size": candidate.window_size,
                    "stride": candidate.stride,
                    "error": last_error,
                }
            )
            continue

        summary = {
            "window_size": candidate.window_size,
            "stride": candidate.stride,
            "num_graphs": len(graphs),
        }
        if len(graphs) < 3:
            summary["reason"] = "insufficient_graphs"
            attempts.append(summary)
            continue
        split, diagnostics = _split_graphs(graphs, candidate.seed)
        empty_splits = diagnostics.get("empty_splits")
        if empty_splits:
            summary["empty_splits"] = empty_splits
            attempts.append(summary)
            continue
        return graphs, metadata, split, diagnostics, candidate, attempts
    detail = {
        "attempts": attempts,
    }
    if last_error is not None:
        detail["last_error"] = last_error
    raise RuntimeError(
        "TON_IoT preprocessing could not produce non-empty train/val/test splits. "
        "Review window/stride parameters or lower --min-nodes. Details: "
        f"{detail}"
    )


def _save_split(
    split: Dict[str, List[Data]],
    metadata: Dict[str, object],
    output_root: Path,
    diagnostics: Optional[Dict[str, object]] = None,
) -> None:
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
    if diagnostics:
        summary["diagnostics"] = diagnostics
    with open(processed_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    config = parse_args()
    telemetry_df = _load_telemetry(config)
    network_df = _load_network(config)
    (
        graphs,
        metadata,
        split,
        diagnostics,
        applied_config,
        attempts,
    ) = _generate_graphs_with_split(telemetry_df, network_df, config)
    diagnostics.setdefault("applied_window", {})
    diagnostics["applied_window"].update(
        {
            "window_size": applied_config.window_size,
            "stride": applied_config.stride,
        }
    )
    if attempts:
        diagnostics.setdefault("window_search_attempts", attempts)
    dataset_root = config.output_root / "toni_iot"
    _save_split(split, metadata, dataset_root, diagnostics)
    print(f"Saved TON_IoT graphs to {dataset_root / 'processed'}")


if __name__ == "__main__":
    main()
