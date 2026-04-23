"""Shared filesystem locations for persistent psychohistory data."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/Users/darenpalmer/conductor/shared-data/psychohistory-v2")


def resolve_data_root(cli_value: str | Path | None = None) -> Path:
    if cli_value is not None:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get("PSYCHOHISTORY_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def warehouse_path(data_root: Path) -> Path:
    return data_root / "warehouse" / "events.duckdb"


def arab_spring_warehouse_path(data_root: Path) -> Path:
    """DuckDB for Arab Spring sources (e.g. ACLED CSV as ``acled_v3``), separate from ``warehouse_path`` (France + GDELT)."""
    return data_root / "arab_spring" / "events.duckdb"


def runs_root(data_root: Path) -> Path:
    return data_root / "runs"
