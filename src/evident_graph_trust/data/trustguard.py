"""Dataset loader mirroring the TrustGuard preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .iot_graph import IoTGraphData
from ..utils.graph import edge_index_to_adjacency


def _resolve_processed_file(root: Path, dataset_name: Optional[str]) -> Path:
    """Attempt to locate a processed graph exported by TrustGuard."""

    if root.is_file():
        return root

    candidates = []
    if dataset_name:
        candidates.extend(
            [
                root / dataset_name / "processed" / "data.pt",
                root / dataset_name / "processed" / "graph.pt",
                root / dataset_name / "processed.pt",
                root / dataset_name / "data.pt",
            ]
        )
    candidates.extend(
        [
            root / "processed" / "data.pt",
            root / "processed" / "graph.pt",
            root / "processed.pt",
            root / "data.pt",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    pt_files = list(root.glob("**/*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(
        f"Could not locate a processed TrustGuard graph under '{root}'. "
        "Please provide the explicit path to the .pt file exported by TrustGuard."
    )


def _extract_tensors(graph_obj: object) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract tensors from a PyTorch Geometric Data object or dictionary."""

    metadata: Dict[str, torch.Tensor] = {}

    if hasattr(graph_obj, "x") and hasattr(graph_obj, "edge_index"):
        features = graph_obj.x.clone().detach()
        edge_index = graph_obj.edge_index.clone().detach()
        labels = graph_obj.y.clone().detach().long()
        for key in ("train_mask", "val_mask", "test_mask", "edge_weight"):
            value = getattr(graph_obj, key, None)
            if value is not None:
                metadata[key] = value
        return features, edge_index, labels, metadata

    if isinstance(graph_obj, dict):
        tensor_dict = {k: v for k, v in graph_obj.items() if isinstance(v, torch.Tensor)}
        np_dict = {k: torch.from_numpy(v) for k, v in graph_obj.items() if isinstance(v, np.ndarray)}
        tensor_dict.update(np_dict)
        try:
            features = tensor_dict["x"].clone().detach()
            edge_index = tensor_dict["edge_index"].clone().detach()
            labels = tensor_dict["y"].clone().detach().long()
        except KeyError as exc:
            missing = ", ".join(sorted(set(["x", "edge_index", "y"]) - tensor_dict.keys()))
            raise KeyError(f"Processed file is missing required key(s): {missing}") from exc

        for key in ("train_mask", "val_mask", "test_mask", "edge_weight"):
            if key in tensor_dict:
                metadata[key] = tensor_dict[key]
        return features, edge_index, labels, metadata

    raise TypeError(
        "Unsupported processed graph format. Expected a PyTorch Geometric Data object "
        "or a dictionary containing 'x', 'edge_index', and 'y'."
    )


def load_trustguard_graph(
    root: str | Path,
    dataset_name: Optional[str] = None,
    fallback_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> IoTGraphData:
    """Load a graph processed with TrustGuard's preprocessing pipeline."""

    root_path = Path(root)
    processed_path = _resolve_processed_file(root_path, dataset_name)
    graph_obj = torch.load(processed_path, map_location="cpu")

    features, edge_index, labels, metadata = _extract_tensors(graph_obj)
    num_nodes = features.size(0)
    edge_weight = metadata.get("edge_weight")

    adjacency = edge_index_to_adjacency(edge_index, num_nodes=num_nodes, edge_weight=edge_weight, symmetric=True)

    def _mask_from_metadata(name: str, ratio: float) -> torch.Tensor:
        mask_tensor = metadata.get(name)
        if mask_tensor is not None:
            return mask_tensor.bool().clone().detach()
        count = int(round(num_nodes * ratio))
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[:count] = True
        return mask

    train_ratio, val_ratio, test_ratio = fallback_split
    train_mask = _mask_from_metadata("train_mask", train_ratio)
    val_mask = _mask_from_metadata("val_mask", val_ratio)
    test_mask = _mask_from_metadata("test_mask", test_ratio)

    if metadata.get("train_mask") is None:
        permutation = torch.randperm(num_nodes)
        train_end = int(train_ratio * num_nodes)
        val_end = train_end + int(val_ratio * num_nodes)
        train_mask.zero_()
        val_mask.zero_()
        test_mask.zero_()
        train_mask[permutation[:train_end]] = True
        val_mask[permutation[train_end:val_end]] = True
        test_mask[permutation[val_end:]] = True

    return IoTGraphData(
        features=features.float(),
        adjacency=adjacency.float(),
        labels=labels.long(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        metadata=metadata,
    )
