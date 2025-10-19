"""Data handling utilities for Evident Graph Trust experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset


@dataclass
class DataModuleConfig:
    dataset_root: Path
    dataset_name: str
    split: str = "train"


class GraphTrustDataset(InMemoryDataset):
    """Placeholder dataset wrapping preprocessed PyG data objects.

    The actual preprocessing steps are described in ``docs/dataset_preprocessing.md``.
    This class expects precomputed ``*.pt`` files containing PyG ``Data`` objects.
    """

    def __init__(self, root: Path, split: str) -> None:
        self.split = split
        self.metadata: Dict[str, Any] = {}
        super().__init__(root, transform=None, pre_transform=None)
        loaded = torch.load(self.processed_paths[0])
        if isinstance(loaded, dict) and "data_list" in loaded:
            data_list = loaded["data_list"]
            self.metadata = loaded.get("metadata", {})
        elif isinstance(loaded, (list, tuple)) and loaded and isinstance(loaded[0], Data):
            data_list = list(loaded)
        else:
            # Assume legacy (data, slices) tuple
            data_list = None
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
        else:
            self.data, self.slices = loaded  # type: ignore[assignment]

        self._infer_dataset_properties()

    def _infer_dataset_properties(self) -> None:
        if hasattr(self.data, "x") and getattr(self.data, "x") is not None:
            self.num_node_features = self.data.x.size(-1)  # type: ignore[attr-defined]
        else:
            self.num_node_features = self.metadata.get("num_features", 0)
        if hasattr(self.data, "y") and getattr(self.data, "y") is not None:
            y_tensor = self.data.y
            num_classes = int(y_tensor.max().item() + 1) if y_tensor.numel() > 0 else 0
        else:
            num_classes = 0
        num_classes = max(num_classes, int(self.metadata.get("num_classes", 0)))
        self.num_classes = num_classes

    @property
    def raw_file_names(self) -> str:
        return f"{self.split}.pt"

    @property
    def processed_file_names(self) -> str:
        return f"{self.split}_graph.pt"

    def download(self) -> None:
        raise RuntimeError("Dataset download is handled externally via scripts/download_data.sh")

    def process(self) -> None:
        raise RuntimeError("Preprocessing should be executed via dedicated scripts before instantiation.")


def load_dataset(config: DataModuleConfig) -> GraphTrustDataset:
    """Load the graph dataset for the given split."""

    dataset_root = Path(config.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist. Please run preprocessing first.")
    return GraphTrustDataset(dataset_root / config.dataset_name, split=config.split)
