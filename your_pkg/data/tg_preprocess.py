"""Preprocessing utilities for TrustGuard datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from torch_geometric.data import Data

from .tg_schema import SCHEMA, label_from_rating


def _bin_time(df: pd.DataFrame, n_slices: int, time_col: str) -> List[pd.DataFrame]:
    """Split *df* into *n_slices* contiguous time bins."""

    if n_slices <= 0:
        raise ValueError("Number of snapshots must be positive")

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise ValueError("Time column contains NaN values")

    step = (t_max - t_min) / n_slices if n_slices else 0
    bounds = [t_min + i * step for i in range(n_slices)] + [t_max + 1]

    parts: List[pd.DataFrame] = []
    for idx in range(n_slices):
        lo, hi = bounds[idx], bounds[idx + 1]
        part = df[(df[time_col] >= lo) & (df[time_col] < hi)].copy()
        part["t_idx"] = idx + 1
        parts.append(part)
    return parts


def _global_node_map(df: pd.DataFrame, src_col: str, dst_col: str) -> dict:
    """Create a contiguous node id mapping covering all nodes in *df*."""

    series = pd.concat([df[src_col], df[dst_col]], ignore_index=True)
    unique_nodes = pd.Index(series.unique())
    return {value: idx for idx, value in enumerate(unique_nodes.tolist())}


def _to_edge_level_latest(
    part: pd.DataFrame, src_col: str, dst_col: str, time_col: str
) -> pd.DataFrame:
    """Keep the most recent edge opinion per (u, v) pair within the snapshot."""

    if part.empty:
        return part
    part = part.sort_values(time_col)
    return part.groupby([src_col, dst_col], as_index=False).tail(1)


def _build_data(
    df_snap: pd.DataFrame,
    node_map: dict,
    src_col: str,
    dst_col: str,
    rating_col: str,
    t_idx: int,
) -> Data | None:
    """Convert a snapshot dataframe into a PyG :class:`Data` object."""

    if df_snap.empty:
        return None

    u = df_snap[src_col].map(node_map).astype(int).to_numpy()
    v = df_snap[dst_col].map(node_map).astype(int).to_numpy()

    edge_index = torch.tensor([u, v], dtype=torch.long)
    labels = df_snap[rating_col].apply(label_from_rating).astype(int).to_numpy()
    edge_attr = torch.tensor(labels, dtype=torch.long)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_map))
    data.t_idx = torch.tensor([t_idx], dtype=torch.long)
    return data


def _dump_split(path: Path, items: Iterable[Data | None]) -> None:
    for data_obj in items:
        if data_obj is None:
            continue
        idx = int(data_obj.t_idx.item())
        torch.save(data_obj, path / f"t_{idx:02d}.pt")


def preprocess_trustguard_raw(
    root: str,
    raw_file: str = "ratings.csv",
    schema_map: dict | None = None,
    snapshots: int = 10,
    split: str = "standard",
    sep: str = ",",
    encoding: str = "utf-8",
) -> None:
    """Convert TrustGuard raw CSV data into PyG snapshot files."""

    if split != "standard":
        raise ValueError("Only the 'standard' time-ordered split is supported")

    schema = SCHEMA.copy()
    if schema_map:
        schema.update(schema_map)

    root_path = Path(root)
    raw_path = root_path / "raw" / raw_file
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")

    out_dir = root_path / "processed" / "snapshots"
    for subdir in ("train", "val", "test"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, sep=sep, encoding=encoding)
    for required in schema.values():
        if required not in df.columns:
            raise ValueError(
                f"Missing column '{required}' in raw file. Available columns: {df.columns.tolist()}"
            )

    df = df.dropna(subset=list(schema.values()))
    df = df.sort_values(schema["TIME"]).reset_index(drop=True)

    node_map = _global_node_map(df, schema["SRC"], schema["DST"])
    parts = _bin_time(df, snapshots, schema["TIME"])

    snapshots_list: List[Data | None] = []
    for idx, part in enumerate(parts, start=1):
        latest = _to_edge_level_latest(part, schema["SRC"], schema["DST"], schema["TIME"])
        data_obj = _build_data(latest, node_map, schema["SRC"], schema["DST"], schema["RATING"], idx)
        snapshots_list.append(data_obj)

    total = len(snapshots_list)
    train_cut = max(1, int(0.6 * total))
    val_cut = int(0.2 * total)
    test_cut = total - train_cut - val_cut

    train_idxs = list(range(1, train_cut + 1))
    val_idxs = list(range(train_cut + 1, train_cut + val_cut + 1))
    test_idxs = list(range(train_cut + val_cut + 1, total + 1))

    def _select(indices: List[int]) -> List[Data | None]:
        return [snapshots_list[i - 1] for i in indices if 0 < i <= len(snapshots_list)]

    _dump_split(out_dir / "train", _select(train_idxs))
    _dump_split(out_dir / "val", _select(val_idxs))
    _dump_split(out_dir / "test", _select(test_idxs))

    meta = {
        "num_nodes": len(node_map),
        "snapshots": snapshots,
        "split": {"train": train_idxs, "val": val_idxs, "test": test_idxs},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


__all__ = ["preprocess_trustguard_raw"]
