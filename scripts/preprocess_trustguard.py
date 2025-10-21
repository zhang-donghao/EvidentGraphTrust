#!/usr/bin/env python
"""CLI wrapper for preprocessing TrustGuard raw datasets."""

from __future__ import annotations

import argparse
import json
import shlex

from your_pkg.data.tg_preprocess import preprocess_trustguard_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess TrustGuard datasets")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--raw_file", default="ratings.csv", help="Name of the raw CSV file")
    parser.add_argument("--snapshots", type=int, default=10, help="Number of time snapshots to create")
    parser.add_argument("--sep", default=",", help="CSV delimiter")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding")
    parser.add_argument(
        "--schema_map",
        default=None,
        help=(
            "Column remapping. Accepts JSON (e.g. \"{\"SRC\":\"user\"}\") or a whitespace-separated "
            "list of key=value pairs (e.g. SRC=user DST=other)."
        ),
    )
    args = parser.parse_args()

    schema_map = None
    if args.schema_map:
        schema_map = _parse_schema_arg(args.schema_map)

    preprocess_trustguard_raw(
        root=args.root,
        raw_file=args.raw_file,
        schema_map=schema_map,
        snapshots=args.snapshots,
        sep=args.sep,
        encoding=args.encoding,
    )


def _parse_schema_arg(raw: str) -> dict[str, str]:
    """Parse the --schema_map argument.

    We support two syntaxes to play nicely with different shells:
        1. JSON: "{"SRC": "user", "DST": "other"}"
        2. Key/value tokens: "SRC=user DST=other"
    The second form avoids heavy quoting on Windows PowerShell.
    """

    # First attempt standard JSON parsing (strict, informative errors)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to key=value parsing. We intentionally do not silently ignore
        # malformed tokens—any parsing error results in a clear ValueError.
        try:
            tokens = shlex.split(raw)
        except ValueError as exc:  # unmatched quotes, etc.
            raise ValueError(
                "Failed to parse --schema_map; ensure JSON is valid or provide key=value pairs"
            ) from exc

        if not tokens:
            raise ValueError("--schema_map provided but empty")

        parsed = {}
        for token in tokens:
            if "=" not in token:
                raise ValueError(
                    f"Invalid schema token '{token}'. Expected KEY=VALUE format or valid JSON."
                )
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ValueError(
                    f"Invalid schema token '{token}'. Expected non-empty KEY=VALUE pair."
                )
            parsed[key] = value
    else:
        if not isinstance(parsed, dict):
            raise ValueError("--schema_map JSON must decode to an object")

    # Normalize keys/values to strings
    normalized = {str(k): str(v) for k, v in parsed.items()}
    if not normalized:
        raise ValueError("--schema_map cannot be empty")
    return normalized


if __name__ == "__main__":
    main()
