#!/usr/bin/env python3
"""Build Arab Spring node-warehouse mmap + manifest (v0 or v1 recipe; from repo root, overnight if needed)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Allow `python scripts/build_arab_spring_warehouse.py` from repo root without PYTHONPATH=
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.node_warehouse_build_v0 import (
    build_arab_spring_node_warehouse_v0,
    build_arab_spring_node_warehouse_v1,
)
from scripts.warehouse_quality_gate import validate_manifest


DEFAULT_ARTIFACT_DIR = Path("artifacts/warehouse_validation")
DEFAULT_OUT_MMAP = {
    "v0": DEFAULT_ARTIFACT_DIR / "node_warehouse_v0.mmap",
    "v1": DEFAULT_ARTIFACT_DIR / "node_warehouse_v1.mmap",
}
DEFAULT_OUT_MANIFEST = {
    "v0": DEFAULT_ARTIFACT_DIR / "node_warehouse_v0_manifest.json",
    "v1": DEFAULT_ARTIFACT_DIR / "node_warehouse_v1_manifest.json",
}
BUILDERS = {
    "v0": build_arab_spring_node_warehouse_v0,
    "v1": build_arab_spring_node_warehouse_v1,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    gate = p.add_mutually_exclusive_group()
    gate.add_argument(
        "--quality-gate",
        dest="quality_gate",
        action="store_true",
        default=True,
        help="Run warehouse quality gate on output manifest and embed summary in stdout JSON (default).",
    )
    gate.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        help="Skip warehouse quality gate for local debugging only.",
    )
    p.add_argument(
        "--quality-gate-strict",
        action="store_true",
        help="Exit non-zero when the quality gate fails (quality gate runs by default).",
    )
    p.add_argument(
        "--allow-duckdb-fallback",
        action="store_true",
        help="Explicitly allow DuckDB fallback when events.jsonl is missing (default: disabled).",
    )
    return p.parse_args(argv)


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    if args.window_days is None:
        args.window_days = 1 if args.recipe == "v0" else 1461

    builder = BUILDERS[args.recipe]
    out_manifest = args.out_manifest or DEFAULT_OUT_MANIFEST[args.recipe]
    build_kwargs: dict[str, Any] = {
        "warehouse_path": args.warehouse_path,
        "out_mmap": args.out_mmap or DEFAULT_OUT_MMAP[args.recipe],
        "out_manifest": out_manifest,
        "as_of": args.as_of,
        "window_days": args.window_days,
        "show_progress": not args.no_progress,
    }
    if args.recipe == "v1":
        build_kwargs["allow_duckdb_fallback"] = args.allow_duckdb_fallback

    out = builder(**build_kwargs)

    if args.quality_gate:
        manifest_path = Path(out.get("out_manifest", out_manifest))
        quality = validate_manifest(manifest_path)
        out["quality_gate"] = quality
        if args.quality_gate_strict and not quality.get("passed", False):
            setattr(args, "_quality_gate_failure_payload", out)
            raise SystemExit(2)

    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        out = run_build(args)
    except SystemExit as exc:
        payload = getattr(args, "_quality_gate_failure_payload", None)
        if exc.code == 2 and payload is not None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        raise
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
