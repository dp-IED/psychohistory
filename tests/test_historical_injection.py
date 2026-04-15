from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.historical_injection import build_batches, replay_records_for_cutoff


def _record(source_event_id: str, available_at: str) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=dt.date(2021, 1, 1),
        source_available_at=dt.datetime.fromisoformat(available_at.replace("Z", "+00:00")),
        retrieved_at=dt.datetime(2021, 1, 2, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code="FR11",
        location_name="Paris",
        latitude=None,
        longitude=None,
        event_class="protest",
        event_code="141",
        event_base_code="14",
        event_root_code="14",
        quad_class=3,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name=None,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def _write_tape(path: Path, records: list[EventTapeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


def test_injection_batches_are_monotonic(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "batches.jsonl"
    _write_tape(
        tape_path,
        [
            _record("gdelt:1", "2021-01-04T01:00:00Z"),
            _record("gdelt:2", "2021-01-11T00:00:00Z"),
        ],
    )

    batches = build_batches(tape_path=tape_path, out_path=out_path)

    starts = [batch.source_available_start for batch in batches]
    assert starts == sorted(starts)
    assert all(batch.source_available_end == batch.source_available_start + dt.timedelta(days=7) for batch in batches)
    assert [batch.batch_id for batch in batches] == [
        "gdelt_v2_events__source_week__2021-01-04",
        "gdelt_v2_events__source_week__2021-01-11",
    ]


def test_injection_replay_cutoff_equals_direct_filter(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "batches.jsonl"
    records = [
        _record("gdelt:1", "2021-01-04T01:00:00Z"),
        _record("gdelt:2", "2021-01-06T08:00:00Z"),
        _record("gdelt:3", "2021-01-06T20:00:00Z"),
    ]
    _write_tape(tape_path, records)
    build_batches(tape_path=tape_path, out_path=out_path)
    cutoff = dt.datetime(2021, 1, 6, 12, tzinfo=dt.timezone.utc)

    replayed = replay_records_for_cutoff(batches_path=out_path, cutoff=cutoff)
    direct = [
        record
        for record in load_event_tape(tape_path)
        if record.source_available_at < cutoff
    ]

    assert [record.source_event_id for record in replayed] == [
        record.source_event_id for record in direct
    ]


def test_injection_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing event tape"):
        build_batches(tape_path=tmp_path / "missing.jsonl", out_path=tmp_path / "batches.jsonl")
