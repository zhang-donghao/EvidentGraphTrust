#!/usr/bin/env python
"""CLI wrapper for preprocessing TrustGuard raw datasets."""

from __future__ import annotations

import argparse
import json

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
        help="JSON dictionary remapping column names, e.g. {'SRC':'user','DST':'other'}",
    )
    args = parser.parse_args()

    schema_map = None
    if args.schema_map:
        schema_map = json.loads(args.schema_map)
        if not isinstance(schema_map, dict):
            raise ValueError("--schema_map must decode to a JSON object")

    preprocess_trustguard_raw(
        root=args.root,
        raw_file=args.raw_file,
        schema_map=schema_map,
        snapshots=args.snapshots,
        sep=args.sep,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
