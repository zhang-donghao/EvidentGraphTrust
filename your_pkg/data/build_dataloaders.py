"""Data loader utilities for TrustGuard snapshot datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def _load_split(path: Path) -> List[Data]:
    if not path.exists():
        return []

    files = sorted(path.glob("t_*.pt"))
    snapshots: List[Data] = []
    for file_path in files:
        data_obj = torch.load(file_path)
        if not isinstance(data_obj, Data):
            raise TypeError(f"File {file_path} did not contain a torch_geometric.data.Data object")
        if not hasattr(data_obj, "edge_index"):
            raise ValueError(f"{file_path} missing edge_index")
        if not hasattr(data_obj, "edge_attr"):
            raise ValueError(f"{file_path} missing edge_attr (edge labels)")
        if data_obj.edge_attr.dtype != torch.long:
            raise ValueError(f"{file_path} edge_attr must have dtype torch.long")
        snapshots.append(data_obj)
    return snapshots


def build_dataloaders(
    dataset_name: str,
    root: str,
    snapshots: int,
    split: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[Iterable, Iterable | None, Iterable | None]:
    """Return train/val/test data loaders for the specified dataset."""

    base_path = Path(root) / "processed" / "snapshots"
    if dataset_name and Path(root).name != dataset_name:
        raise ValueError(
            f"Dataset root '{root}' does not end with dataset name '{dataset_name}'"
        )
    if split != "standard":
        raise ValueError("Only the 'standard' split is supported by this loader")

    train_items = _load_split(base_path / "train")
    val_items = _load_split(base_path / "val")
    test_items = _load_split(base_path / "test")

    if not train_items:
        raise RuntimeError(f"train split empty at {base_path / 'train'}; did you run preprocessing?")

    if snapshots and (len(train_items) + len(val_items) + len(test_items)) > snapshots:
        raise ValueError(
            "Number of loaded snapshots exceeds configured expectation. "
            "Check the preprocessing configuration."
        )

    train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = (
        DataLoader(val_items, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_items
        else None
    )
    test_loader = (
        DataLoader(test_items, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if test_items
        else None
    )
    return train_loader, val_loader, test_loader


__all__ = ["build_dataloaders"]
