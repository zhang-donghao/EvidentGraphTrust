from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml

from training.trainer import (
    Config,
    DataConfig,
    EvidentialConfig,
    Trainer,
    TrainingConfig,
    TrunkConfig,
)
from trustcore.utils.checks import assert_nonempty_loader
from trustcore.utils.resolve import resolve_object


def _load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_config(data: Dict[str, Any]) -> Config:
    evidential = EvidentialConfig(**(data.get("evidential") or {}))
    training = TrainingConfig(**(data.get("training") or {}))

    trunk_section = data.get("trunk") or {}
    trunk = TrunkConfig(
        target=trunk_section.get("target", ""),
        kwargs=dict(trunk_section.get("kwargs") or {}),
    )

    data_section = data.get("data") or {}
    data_cfg = DataConfig(
        builder=data_section.get("builder", ""),
        root=data_section.get("root", "data"),
        kwargs=dict(data_section.get("kwargs") or {}),
    )

    return Config(
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


def _parse_json_dict(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise ValueError(f"Invalid JSON dictionary: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON value must be an object")
    return parsed


def _build_trunk(cfg: Config, device: torch.device) -> torch.nn.Module:
    if not cfg.trunk.target:
        raise ValueError("Trunk target must be specified in the configuration")
    trunk_cls = resolve_object(cfg.trunk.target)
    trunk = trunk_cls(**cfg.trunk.kwargs)
    return trunk.to(device)


def _invoke_builder(builder, cfg: Config, seed: int, **kwargs):
    try:
        return builder(cfg=cfg, seed=seed, **kwargs)
    except TypeError:
        try:
            return builder(seed=seed, **kwargs)
        except TypeError:
            return builder(**kwargs)


def _build_dataloaders(cfg: Config, seed: int) -> Tuple[Iterable, Iterable | None, Iterable | None]:
    if not cfg.data.builder:
        raise ValueError("Data builder must be specified in the configuration")

    builder = resolve_object(cfg.data.builder)
    kwargs = dict(cfg.data.kwargs)
    kwargs.setdefault("root", cfg.data.root)
    loaders = _invoke_builder(builder, cfg, seed, **kwargs)

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
        raise RuntimeError("Data builder did not return a training loader")

    assert_nonempty_loader(train_loader, "train")
    if val_loader is not None:
        assert_nonempty_loader(val_loader, "val")
    if test_loader is not None:
        assert_nonempty_loader(test_loader, "test")

    return train_loader, val_loader, test_loader


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def _describe_loader(name: str, loader: Iterable) -> None:
    iterator = iter(loader)
    batch = next(iterator)

    num_nodes = getattr(batch, "num_nodes", None)
    if num_nodes is None and hasattr(batch, "x"):
        num_nodes = batch.x.size(0)

    edge_index = getattr(batch, "edge_index", None)
    num_edges = edge_index.size(1) if edge_index is not None else None

    labels = None
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        labels = batch.edge_attr
    elif hasattr(batch, "edge_label") and batch.edge_label is not None:
        labels = batch.edge_label

    label_shape = tuple(labels.shape) if labels is not None else None
    print(
        f"{name}: nodes={num_nodes}, edges={num_edges}, edge_label_shape={label_shape}, "
        f"label_dtype={getattr(labels, 'dtype', None)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trust modeling trainer entrypoint")
    parser.add_argument("--config", type=str, default="", help="Path to YAML configuration")
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    parser.add_argument("--head", type=str, choices=["mlp", "evidential"], help="Head type")
    parser.add_argument("--log_dir", type=str, default=None, help="Logging directory override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--trunk-target", type=str, default=None, help="Override trunk dotted path")
    parser.add_argument("--trunk-kwargs", type=str, default=None, help="JSON dict for trunk init kwargs")
    parser.add_argument("--data-builder", type=str, default=None, help="Override data builder dotted path")
    parser.add_argument("--data-kwargs", type=str, default=None, help="JSON dict forwarded to data builder")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root override")
    parser.add_argument("--override_json", type=str, default=None, help="JSON overrides merged into config")
    parser.add_argument("--inspect_data", action="store_true", help="Inspect loaders then exit")
    args = parser.parse_args()

    cfg_dict = _load_config(args.config)
    if args.override_json:
        overrides = _parse_json_dict(args.override_json)
        _deep_update(cfg_dict, overrides)

    cfg = _dict_to_config(cfg_dict)

    if args.model:
        cfg.model = args.model
    if args.head:
        cfg.head = args.head
    if args.log_dir:
        cfg.log_dir = args.log_dir
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

    seed_value = args.seed if args.seed is not None else cfg.training.seed
    cfg.training.seed = seed_value

    _set_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, device=device)

    trunk = _build_trunk(cfg, device)
    head = trainer._build_head(cfg.embedding_dim, cfg.n_classes)

    optimizer = torch.optim.Adam(
        list(trunk.parameters()) + list(head.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    train_loader, val_loader, test_loader = _build_dataloaders(cfg, seed=seed_value)

    if args.inspect_data:
        _describe_loader("train", train_loader)
        if val_loader is not None:
            _describe_loader("val", val_loader)
        if test_loader is not None:
            _describe_loader("test", test_loader)
        raise SystemExit(0)

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
