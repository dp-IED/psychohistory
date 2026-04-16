from __future__ import annotations

from pathlib import Path

from ingest.data_gc import prune, report


def test_report_computes_known_path_sizes(tmp_path: Path) -> None:
    wh = tmp_path / "warehouse" / "events.duckdb"
    wh.parent.mkdir()
    wh.write_bytes(b"abc")
    raw_fragment = tmp_path / "raw" / "gdelt" / "fragments" / "rows.jsonl"
    raw_fragment.parent.mkdir(parents=True)
    raw_fragment.write_bytes(b"12345")
    run_file = tmp_path / "runs" / "source_experiments_smoke" / "predictions.jsonl"
    run_file.parent.mkdir(parents=True)
    run_file.write_bytes(b"12")

    payload = report(tmp_path)

    assert payload["total_bytes"] >= 10
    by_category = {entry["category"] for entry in payload["paths"]}
    assert {"warehouse", "raw", "run"}.issubset(by_category)


def test_prune_dry_run_deletes_nothing(tmp_path: Path) -> None:
    fragments = tmp_path / "raw" / "acled" / "fragments"
    fragments.mkdir(parents=True)
    (fragments / "page.jsonl").write_text("x", encoding="utf-8")

    result = prune(tmp_path, raw=True, dry_run=True)

    assert result["deleted_count"] == 1
    assert fragments.exists()


def test_prune_raw_deletes_only_fragment_directories(tmp_path: Path) -> None:
    fragments = tmp_path / "raw" / "acled" / "fragments"
    fragments.mkdir(parents=True)
    (fragments / "page.jsonl").write_text("x", encoding="utf-8")
    manifest = tmp_path / "raw" / "acled" / "fetch_manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    warehouse = tmp_path / "warehouse" / "events.duckdb"
    warehouse.parent.mkdir()
    warehouse.write_bytes(b"db")

    prune(tmp_path, raw=True)

    assert not fragments.exists()
    assert manifest.exists()
    assert warehouse.exists()


def test_prune_smoke_runs_deletes_only_smoke_dirs(tmp_path: Path) -> None:
    smoke = tmp_path / "runs" / "source_experiments_smoke"
    full = tmp_path / "runs" / "source_experiments_full"
    smoke.mkdir(parents=True)
    full.mkdir(parents=True)

    prune(tmp_path, smoke_runs=True)

    assert not smoke.exists()
    assert full.exists()


def test_prune_smoke_runs_deletes_nested_smoke_dirs(tmp_path: Path) -> None:
    smoke = tmp_path / "runs" / "source_experiments" / "smoke_cli"
    full = tmp_path / "runs" / "source_experiments" / "full"
    smoke.mkdir(parents=True)
    full.mkdir(parents=True)

    prune(tmp_path, smoke_runs=True)

    assert not smoke.exists()
    assert full.exists()


def test_prune_materialized_snapshots_under_runs(tmp_path: Path) -> None:
    snapshots = tmp_path / "runs" / "full" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "as_of_2021-01-04.json.gz").write_text("x", encoding="utf-8")

    prune(tmp_path, materialized_snapshots=True)

    assert not snapshots.exists()
