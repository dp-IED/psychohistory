from __future__ import annotations

from pathlib import Path

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
