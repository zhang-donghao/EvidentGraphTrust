"""Runtime checks for dataset integrity."""

from __future__ import annotations

from typing import Iterable


def assert_nonempty_loader(loader: Iterable, name: str) -> None:
    """Assert that *loader* yields at least one batch.

    Parameters
    ----------
    loader: Iterable
        Training/validation/test loader to inspect.
    name: str
        Human readable split name used in the error message.
    """

    iterator = iter(loader)
    try:
        next(iterator)
    except StopIteration as exc:  # pragma: no cover - depends on external data
        raise RuntimeError(f"{name} loader is empty. Check data builder/path/split.") from exc
