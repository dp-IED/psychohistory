from __future__ import annotations

import argparse
from pathlib import Path

from ingest.cli_args import DATA_ROOT_HELP, WAREHOUSE_PATH_HELP, add_warehouse_source_args
from ingest.paths import DEFAULT_DATA_ROOT, resolve_data_root, runs_root, warehouse_path


def test_resolve_data_root_prefers_cli_value(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PSYCHOHISTORY_DATA_ROOT", str(tmp_path / "env"))

    assert resolve_data_root(tmp_path / "cli") == (tmp_path / "cli").resolve()


def test_resolve_data_root_uses_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PSYCHOHISTORY_DATA_ROOT", str(tmp_path / "env"))

    assert resolve_data_root() == (tmp_path / "env").resolve()


def test_resolve_data_root_default(monkeypatch) -> None:
    monkeypatch.delenv("PSYCHOHISTORY_DATA_ROOT", raising=False)

    assert resolve_data_root() == DEFAULT_DATA_ROOT


def test_data_root_child_paths(tmp_path: Path) -> None:
    assert warehouse_path(tmp_path) == tmp_path / "warehouse" / "events.duckdb"
    assert runs_root(tmp_path) == tmp_path / "runs"


def test_add_warehouse_source_args_registers_flags() -> None:
    parser = argparse.ArgumentParser()
    add_warehouse_source_args(parser)
    actions = {a.dest: a for a in parser._actions if hasattr(a, "dest")}
    assert actions["data_root"].help == DATA_ROOT_HELP
    assert actions["warehouse_path"].help == WAREHOUSE_PATH_HELP
