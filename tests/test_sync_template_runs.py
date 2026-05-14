from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.memory_store import JsonlMemoryStore
from scripts.sync_template_runs import SyncFormatError, sync_template_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_sync_imports_new_runs_and_skips_existing(tmp_path: Path) -> None:
    template_dir = tmp_path / "template"
    memory = JsonlMemoryStore(tmp_path / "memory")

    rows = [
        {
            "question_id": 101,
            "question_text": "Will X happen?",
            "run_timestamp": "2026-05-20T10:00:00Z",
            "close_date": "2026-05-20",
            "resolution_date": "2026-06-01",
            "posted_probability": 0.62,
        },
        {
            "question_id": 102,
            "question_text": "Will Y happen?",
            "run_timestamp": "2026-05-20T11:00:00Z",
            "close_date": "2026-05-21",
            "resolution_date": "2026-06-02",
            "posted_probability": 0.41,
        },
    ]
    _write_jsonl(template_dir / "runs.jsonl", rows)

    first = sync_template_outputs(template_dir, memory)
    assert first.scanned == 2
    assert first.imported == 2
    assert first.skipped_existing == 0

    second = sync_template_outputs(template_dir, memory)
    assert second.scanned == 2
    assert second.imported == 0
    assert second.skipped_existing == 2


def test_sync_resolves_when_outcome_present_and_is_idempotent(tmp_path: Path) -> None:
    template_dir = tmp_path / "template"
    memory = JsonlMemoryStore(tmp_path / "memory")

    rows = [
        {
            "question_id": 201,
            "question_text": "Will Z happen?",
            "run_timestamp": "2026-05-22T08:00:00Z",
            "close_date": "2026-05-22",
            "resolution_date": "2026-06-03",
            "posted_probability": 0.8,
            "resolved_outcome": True,
        }
    ]
    _write_jsonl(template_dir / "resolved.jsonl", rows)

    first = sync_template_outputs(template_dir, memory)
    assert first.imported == 1
    assert first.resolved == 1
    assert first.resolve_skipped == 0

    second = sync_template_outputs(template_dir, memory)
    assert second.imported == 0
    assert second.resolved == 0
    assert second.resolve_skipped == 1


def test_sync_requires_canonical_run_timestamp_field(tmp_path: Path) -> None:
    template_dir = tmp_path / "template"
    memory = JsonlMemoryStore(tmp_path / "memory")

    rows = [
        {
            "question_id": 301,
            "question_text": "Will timestamp fallback be accepted?",
            "timestamp": "2026-05-22T08:00:00Z",
            "close_date": "2026-05-22",
            "resolution_date": "2026-06-03",
            "posted_probability": 0.5,
        }
    ]
    _write_jsonl(template_dir / "bad.jsonl", rows)

    with pytest.raises(SyncFormatError, match="run_timestamp"):
        sync_template_outputs(template_dir, memory)
