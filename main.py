from __future__ import annotations

import argparse
import importlib
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from torch import nn

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - allow import failure during docs
    Data = None  # type: ignore

from training.trainer import (
    Config,
    DataConfig,
    EvidentialConfig,
    IdentityTrunk,
    Trainer,
    TrainingConfig,
    TrunkConfig,
)


def _load_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dict_to_config(data: dict) -> Config:
    evidential = EvidentialConfig(**data.get("evidential", {}))
    training = TrainingConfig(**data.get("training", {}))
    trunk_section = data.get("trunk", {}) or {}
    trunk = TrunkConfig(
        target=trunk_section.get("target", ""),
        kwargs=dict(trunk_section.get("kwargs", {}) or {}),
    )
    data_section = data.get("data", {}) or {}
    data_cfg = DataConfig(
        builder=data_section.get("builder", ""),
        root=data_section.get("root", "data"),
        kwargs=dict(data_section.get("kwargs", {}) or {}),
    )
    cfg = Config(
        model=data.get("model", "trustguard"),
        head=data.get("head", "mlp"),
        n_classes=data.get("n_classes", 2),
        embedding_dim=data.get("embedding_dim", 64),
        trunk=trunk,
        data=data_cfg,
        evidential=evidential,
        training=training,
        log_dir=data.get("log_dir", "runs/default"),
    )
    return cfg


def _synthetic_data(cfg: Config, seed: int = 7) -> Tuple[list, list, list]:
    if Data is None:
        raise ImportError("torch_geometric is required for synthetic data generation")
    random.seed(seed)
    torch.manual_seed(seed)
    num_nodes = 16
    num_edges = 32
    x = torch.randn(num_nodes, cfg.embedding_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randint(0, cfg.n_classes, (num_edges,))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    train = [data]
    val = [data.clone()]
    test = [data.clone()]
    return train, val, test


def _parse_json_dict(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"Invalid JSON dictionary: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Parsed value must be a JSON object")
    return parsed


def _resolve_target(target: str):
    module_path, _, attr = target.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid target path '{target}'. Expected module.Class format")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - dynamic import path
        raise ImportError(f"Target '{attr}' not found in module '{module_path}'") from exc


TRUNK_REGISTRY = {
    "trustguard": "trustguard.model.TrustGuard",
    "guardian": "trustguard.model.Guardian",
}


def _build_trunk(cfg: Config, device: torch.device) -> nn.Module:
    target = cfg.trunk.target or TRUNK_REGISTRY.get(cfg.model.lower(), "")
    if target:
        try:
            trunk_cls = _resolve_target(target)
        except (ImportError, ValueError) as exc:
            if cfg.trunk.target:
                raise ImportError(
                    f"Unable to import trunk '{target}'. Provide a valid path via config or --trunk-target"
                ) from exc
            warnings.warn(
                f"Falling back to IdentityTrunk because '{target}' could not be imported: {exc}",
                RuntimeWarning,
            )
        else:
            trunk = trunk_cls(**cfg.trunk.kwargs)
            return trunk.to(device)
    return IdentityTrunk(cfg.embedding_dim).to(device)


def _build_dataloaders(cfg: Config, seed: int) -> Tuple[Iterable, Iterable | None, Iterable | None]:
    builder_path = cfg.data.builder
    if not builder_path:
        return _synthetic_data(cfg, seed)

    data_kwargs = dict(cfg.data.kwargs)
    data_kwargs.setdefault("root", cfg.data.root)
    try:
        builder = _resolve_target(builder_path)
    except (ImportError, ValueError) as exc:
        raise ImportError(
            f"Unable to import data builder '{builder_path}'. Provide a valid path via config or --data-builder"
        ) from exc

    try:
        loaders = builder(cfg=cfg, seed=seed, **data_kwargs)
    except TypeError:
        try:
            loaders = builder(seed=seed, **data_kwargs)
        except TypeError:
            loaders = builder(**data_kwargs)
    train_loader: Iterable | None = None
    val_loader: Iterable | None = None
    test_loader: Iterable | None = None

    if isinstance(loaders, dict):
        train_loader = loaders.get("train") or loaders.get("train_loader")
        val_loader = loaders.get("val") or loaders.get("val_loader")
        test_loader = loaders.get("test") or loaders.get("test_loader")
    elif isinstance(loaders, (list, tuple)):
        if len(loaders) >= 1:
            train_loader = loaders[0]
        if len(loaders) >= 2:
            val_loader = loaders[1]
        if len(loaders) >= 3:
            test_loader = loaders[2]
    else:
        train_loader = loaders

    if train_loader is None:
        raise ValueError("Data builder did not return a training loader")
    return train_loader, val_loader, test_loader


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trust modeling trainer entrypoint")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Model trunk to use")
    parser.add_argument("--head", type=str, default=None, choices=["mlp", "evidential"], help="Head type")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name")
    parser.add_argument("--snapshots", type=int, default=1, help="Number of snapshots")
    parser.add_argument("--eval_protocol", type=str, default="single_observed", help="Evaluation protocol")
    parser.add_argument("--log_dir", type=str, default="runs/default", help="Logging directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--trunk-target", type=str, default=None, help="Dotted path to trunk class")
    parser.add_argument(
        "--trunk-kwargs",
        type=str,
        default=None,
        help="JSON dict of keyword arguments for trunk initialization",
    )
    parser.add_argument(
        "--data-builder",
        type=str,
        default=None,
        help="Dotted path to data loader builder returning train/val/(test)",
    )
    parser.add_argument(
        "--data-kwargs",
        type=str,
        default=None,
        help="JSON dict forwarded to the data builder",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root directory")
    args = parser.parse_args()

    cfg_dict = _load_config(args.config)
    cfg = _dict_to_config(cfg_dict)
    if args.model:
        cfg.model = args.model
    if args.head:
        cfg.head = args.head
    cfg.log_dir = args.log_dir or cfg.log_dir
    if args.trunk_target:
        cfg.trunk.target = args.trunk_target
    if args.trunk_kwargs:
        cfg.trunk.kwargs.update(_parse_json_dict(args.trunk_kwargs))
    if args.data_builder:
        cfg.data.builder = args.data_builder
    if args.data_kwargs:
        cfg.data.kwargs.update(_parse_json_dict(args.data_kwargs))
    if args.data_root:
        cfg.data.root = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, device=device)
    trunk: nn.Module = _build_trunk(cfg, device)
    head = trainer._build_head(cfg.embedding_dim, cfg.n_classes)

    optimizer = torch.optim.Adam(
        list(trunk.parameters()) + list(head.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    _set_seed(args.seed)
    train_loader, val_loader, test_loader = _build_dataloaders(cfg, seed=args.seed)
    for epoch in range(cfg.training.epochs):
        train_metrics = trainer.train_one_epoch(trunk, head, train_loader, optimizer)
        trainer._write_metrics(epoch, "train", train_metrics)
        val_metrics = trainer.evaluate(trunk, head, val_loader)
        if val_metrics:
            trainer._write_metrics(epoch, "val", val_metrics)

    if test_loader is not None:
        test_metrics = trainer.evaluate(trunk, head, test_loader)
        if test_metrics:
            trainer._write_metrics(cfg.training.epochs, "test", test_metrics)

    print(f"Training complete. Metrics logged to {Path(cfg.log_dir) / 'metrics.csv'}")


if __name__ == "__main__":
    main()
