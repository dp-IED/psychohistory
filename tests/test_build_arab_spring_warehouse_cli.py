from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import scripts.build_arab_spring_warehouse as build_cli


def _args(tmp_path: Path, **overrides) -> Namespace:
    base = {
        "warehouse_path": tmp_path / "events.duckdb",
        "recipe": "v0",
        "out_mmap": tmp_path / "nodes.mmap",
        "out_manifest": tmp_path / "manifest.json",
        "as_of": None,
        "window_days": None,
        "no_progress": True,
        "quality_gate": True,
        "quality_gate_strict": False,
        "allow_duckdb_fallback": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_build_embeds_quality_gate_by_default(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    def fake_builder(**kwargs):
        assert kwargs["out_manifest"] == manifest
        return {"out_manifest": str(manifest), "row_count": 1}

    quality = {"passed": True, "checks": {"first_seen_complete": True}}
    monkeypatch.setitem(build_cli.BUILDERS, "v0", fake_builder)
    monkeypatch.setattr(build_cli, "validate_manifest", lambda path: quality)

    out = build_cli.run_build(_args(tmp_path))

    assert out["quality_gate"] == quality


def test_run_build_strict_quality_gate_exits_nonzero_on_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_builder(**kwargs):
        return {"out_manifest": str(kwargs["out_manifest"]), "row_count": 1}

    monkeypatch.setitem(build_cli.BUILDERS, "v0", fake_builder)
    monkeypatch.setattr(build_cli, "validate_manifest", lambda path: {"passed": False, "checks": {}})

    with pytest.raises(SystemExit) as exc:
        build_cli.run_build(_args(tmp_path, quality_gate_strict=True))

    assert exc.value.code == 2


def test_run_build_can_explicitly_skip_quality_gate(monkeypatch, tmp_path: Path) -> None:
    def fake_builder(**kwargs):
        return {"out_manifest": str(kwargs["out_manifest"]), "row_count": 1}

    def fail_if_called(path):
        raise AssertionError("quality gate should be skipped")

    monkeypatch.setitem(build_cli.BUILDERS, "v0", fake_builder)
    monkeypatch.setattr(build_cli, "validate_manifest", fail_if_called)

    out = build_cli.run_build(_args(tmp_path, quality_gate=False))

    assert "quality_gate" not in out
