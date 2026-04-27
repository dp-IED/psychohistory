"""CLI wrapper for Arab Spring node warehouse v1 building."""

import argparse
import sys
from datetime import date
from pathlib import Path

from baselines.node_warehouse_build_v0 import build_arab_spring_node_warehouse_v1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Arab Spring node warehouse v1 from JSONL events."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input DuckDB warehouse or JSONL file",
    )
    parser.add_argument(
        "--output-mmap",
        type=Path,
        required=True,
        help="Path to output mmap file",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output manifest JSON file",
    )
    parser.add_argument(
        "--as-of",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="As-of date (YYYY-MM-DD); default is end of data range",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=1461,
        help="Window in days (default: 1461)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show progress bar (default: True)",
    )
    
    args = parser.parse_args()
    
    result = build_arab_spring_node_warehouse_v1(
        warehouse_path=args.input,
        out_mmap=args.output_mmap,
        out_manifest=args.output_manifest,
        as_of=args.as_of,
        window_days=args.window_days,
        show_progress=args.show_progress,
    )
    
    print(f"\nWarehouse build complete:")
    print(f"  Input: {result['warehouse_path']}")
    print(f"  Rows: {result['row_count']:,}")
    print(f"  Mmap: {result['out_mmap']}")
    print(f"  Manifest: {result['out_manifest']}")
    print(f"  as_of: {result['as_of']}")
    print(f"  window_days: {result['window_days']}")


if __name__ == "__main__":
    main()
