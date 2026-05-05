#!/usr/bin/env python3
"""Build Arab Spring node-warehouse mmap + manifest (v0 or v1 recipe; from repo root, overnight if needed)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Allow `python scripts/build_arab_spring_warehouse.py` from repo root without PYTHONPATH=
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.node_warehouse_build_v0 import (
    build_arab_spring_node_warehouse_v0,
    build_arab_spring_node_warehouse_v1,
)
from scripts.warehouse_quality_gate import validate_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--warehouse-path",
        type=Path,
        default=Path("shared_data/arab_spring/events.duckdb"),
    )
    p.add_argument(
        "--recipe",
        choices=("v0", "v1"),
        default="v0",
        help="Warehouse recipe to build (default: v0)",
    )
    p.add_argument(
        "--out-mmap",
        type=Path,
        default=None,
        help=(
            "Output mmap path. Default writes under artifacts/warehouse_validation/ "
            "to avoid touching shared_data unless explicitly overridden."
        ),
    )
    p.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help=(
            "Output manifest path. Default writes under artifacts/warehouse_validation/ "
            "to avoid touching shared_data unless explicitly overridden."
        ),
    )
    p.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="PIT end date (default: 2013-12-31, end of locked Arab Spring range)",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=None,
        help=(
            "Backward window (inclusive) ending at --as-of. "
            "Default: 1 for --recipe v0, 1461 for --recipe v1 (full v1 PIT window)"
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm step progress on stderr (default: progress on)",
    )
    p.add_argument(
        "--quality-gate",
        action="store_true",
        help="Run warehouse quality gate on output manifest and embed summary in stdout JSON.",
    )
    p.add_argument(
        "--quality-gate-strict",
        action="store_true",
        help="With --quality-gate: exit non-zero when any contract check fails.",
    )
    p.add_argument(
        "--allow-duckdb-fallback",
        action="store_true",
        help="Explicitly allow DuckDB fallback when events.jsonl is missing (default: disabled).",
    )
    args = p.parse_args()

    if args.window_days is None:
        args.window_days = 1 if args.recipe == "v0" else 1461

    default_artifact_dir = Path("artifacts/warehouse_validation")
    default_out_mmap = {
        "v0": default_artifact_dir / "node_warehouse_v0.mmap",
        "v1": default_artifact_dir / "node_warehouse_v1.mmap",
    }
    default_out_manifest = {
        "v0": default_artifact_dir / "node_warehouse_v0_manifest.json",
        "v1": default_artifact_dir / "node_warehouse_v1_manifest.json",
    }
    builder = {
        "v0": build_arab_spring_node_warehouse_v0,
        "v1": build_arab_spring_node_warehouse_v1,
    }[args.recipe]

    build_kwargs = {
        "warehouse_path": args.warehouse_path,
        "out_mmap": args.out_mmap or default_out_mmap[args.recipe],
        "out_manifest": args.out_manifest or default_out_manifest[args.recipe],
        "as_of": args.as_of,
        "window_days": args.window_days,
        "show_progress": not args.no_progress,
    }
    if args.recipe == "v1":
        build_kwargs["allow_duckdb_fallback"] = args.allow_duckdb_fallback

    out = builder(**build_kwargs)

    if args.quality_gate:
        manifest_path = Path(out.get("out_manifest", args.out_manifest or default_out_manifest[args.recipe]))
        quality = validate_manifest(manifest_path)
        out["quality_gate"] = quality
        if args.quality_gate_strict and not quality.get("passed", False):
            print(json.dumps(out, indent=2, sort_keys=True))
            raise SystemExit(2)

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
