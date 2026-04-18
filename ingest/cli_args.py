"""Shared argparse fragments for CLIs that read the DuckDB event warehouse."""

from __future__ import annotations

import argparse

DATA_ROOT_HELP = (
    "Root directory for data/ layout; default warehouse is <data-root>/warehouse/events.duckdb."
)
WAREHOUSE_PATH_HELP = "Path to events.duckdb (overrides the default derived from --data-root)."


def add_warehouse_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=None, help=DATA_ROOT_HELP)
    parser.add_argument("--warehouse-path", default=None, help=WAREHOUSE_PATH_HELP)
