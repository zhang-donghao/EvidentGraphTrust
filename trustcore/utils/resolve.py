"""Utilities for dynamically resolving dotted Python objects."""

from __future__ import annotations

import importlib


def resolve_object(dotted_path: str):
    """Return the Python object referenced by *dotted_path*.

    Parameters
    ----------
    dotted_path: str
        Fully qualified module path ending with the attribute/class name.

    Raises
    ------
    ValueError
        If *dotted_path* is empty or does not contain a module separator.
    ImportError
        If the module cannot be imported or the attribute is missing.
    """

    if not dotted_path:
        raise ValueError("Empty dotted path provided for resolution")

    if "." not in dotted_path:
        raise ValueError(
            f"Invalid dotted path '{dotted_path}'. Expected format 'pkg.module:Class'"
        )

    module_path, attr_name = dotted_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - import errors are propagated
        raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - attribute errors are propagated
        raise ImportError(
            f"Module '{module_path}' does not define attribute '{attr_name}'"
        ) from exc
