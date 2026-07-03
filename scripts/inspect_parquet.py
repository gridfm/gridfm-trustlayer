#!/usr/bin/env python3
"""Small helper to inspect a prediction or output parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a parquet file produced by trustlayer workflows."
    )
    parser.add_argument("path", help="Path to the parquet file to inspect.")
    parser.add_argument(
        "--head", type=int, default=8, help="Number of rows to print from the head."
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Optional subset of columns to display in the row sample.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.path)
    frame = pd.read_parquet(parquet_path)

    print(f"path: {parquet_path}")
    print(f"rows: {len(frame)}")
    print(f"columns: {list(frame.columns)}")
    if "scenario" in frame.columns:
        print(f"n_scenarios: {frame['scenario'].nunique()}")
    if "bus" in frame.columns:
        print(f"n_buses: {frame['bus'].nunique()}")

    interval_columns = [
        column
        for column in frame.columns
        if column.endswith("_pi_lower") or column.endswith("_pi_upper")
    ]
    if interval_columns:
        print(f"interval_columns: {interval_columns}")
        widths = {}
        for column in interval_columns:
            if column.endswith("_pi_lower"):
                prefix = column[:-9]
                upper_name = f"{prefix}_pi_upper"
                if upper_name in frame.columns:
                    width = frame[upper_name] - frame[column]
                    widths[prefix] = {
                        "mean": float(width.mean()),
                        "min": float(width.min()),
                        "max": float(width.max()),
                    }
        if widths:
            print(f"interval_widths: {widths}")

    sample_columns = list(frame.columns) if args.columns is None else args.columns
    print("sample:")
    print(frame[sample_columns].head(args.head).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
