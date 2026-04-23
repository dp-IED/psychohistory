from __future__ import annotations

import datetime as dt
from pathlib import Path

from ingest.event_records import load_event_records
from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.event_warehouse import (
    delete_by_source_name,
    export_jsonl,
    import_jsonl,
    init_warehouse,
    query_records,
    source_counts,
    upsert_records,
    warehouse_audit,
)


def _record(
    source_name: str,
    source_event_id: str,
    *,
    event_date: dt.date = dt.date(2021, 1, 5),
    admin1_code: str = "FR11",
    available_at: dt.datetime | None = None,
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name=source_name,
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=available_at or dt.datetime(2021, 1, 6, tzinfo=dt.timezone.utc),
        retrieved_at=dt.datetime(2021, 1, 7, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name="Paris",
        latitude=48.8566,
        longitude=2.3522,
        event_class="protest",
        event_code="141" if source_name == "gdelt_v2_events" else "Protests",
        event_base_code="14" if source_name == "gdelt_v2_events" else "Protests",
        event_root_code="14" if source_name == "gdelt_v2_events" else "Protests",
        quad_class=3 if source_name == "gdelt_v2_events" else None,
        goldstein_scale=-6.5 if source_name == "gdelt_v2_events" else None,
        num_mentions=4 if source_name == "gdelt_v2_events" else None,
        num_sources=2 if source_name == "gdelt_v2_events" else None,
        num_articles=3 if source_name == "gdelt_v2_events" else None,
        avg_tone=-1.2 if source_name == "gdelt_v2_events" else None,
        actor1_name="Protesters",
        actor1_country_code="FRA",
        actor2_name=None,
        actor2_country_code=None,
        source_url="https://example.test/story",
        raw={"id": source_event_id},
    )


def _write_tape(path: Path, records: list[EventTapeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


def test_init_warehouse_creates_database(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "events.duckdb"

    init_warehouse(db_path)

    assert db_path.exists()


def test_upsert_records_deduplicates_primary_key(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    first = _record("gdelt_v2_events", "gdelt:1", admin1_code="FR11")
    second = _record("gdelt_v2_events", "gdelt:1", admin1_code="FR22")

    result = upsert_records(db_path=db_path, records=[first, second])
    records = query_records(db_path=db_path)

    assert result["input_count"] == 2
    assert result["duplicate_input_count"] == 1
    assert len(records) == 1
    assert records[0].admin1_code == "FR22"


def test_query_records_filters_source_and_availability(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    upsert_records(
        db_path=db_path,
        records=[
            _record("gdelt_v2_events", "gdelt:1", available_at=dt.datetime(2021, 1, 5, tzinfo=dt.timezone.utc)),
            _record("acled", "acled:FRA1", available_at=dt.datetime(2021, 1, 10, tzinfo=dt.timezone.utc)),
        ],
    )

    assert [record.source_name for record in query_records(db_path=db_path, source_names={"acled"})] == ["acled"]
    visible = query_records(
        db_path=db_path,
        available_before=dt.datetime(2021, 1, 6, tzinfo=dt.timezone.utc),
    )
    assert [record.source_event_id for record in visible] == ["gdelt:1"]


def test_import_and_export_jsonl_gzip(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    jsonl_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "exported.jsonl.gz"
    _write_tape(
        jsonl_path,
        [_record("gdelt_v2_events", "gdelt:1"), _record("acled", "acled:FRA1")],
    )

    import_result = import_jsonl(db_path=db_path, jsonl_path=jsonl_path)
    export_result = export_jsonl(db_path=db_path, out_path=out_path, source_names={"acled"})
    exported = load_event_tape(out_path)

    assert import_result["upserted_count"] == 2
    assert export_result["gzip_output"] is True
    assert [record.source_name for record in exported] == ["acled"]


def test_source_counts_and_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    upsert_records(
        db_path=db_path,
        records=[
            _record("gdelt_v2_events", "gdelt:1"),
            _record("acled", "acled:FRA1"),
        ],
    )

    audit = warehouse_audit(db_path)

    assert source_counts(db_path) == {"acled": 1, "gdelt_v2_events": 1}
    assert audit["total_event_count"] == 2
    assert audit["admin1_count"] == 1


def test_load_event_records_reads_warehouse(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    upsert_records(db_path=db_path, records=[_record("gdelt_v2_events", "gdelt:1")])

    records = load_event_records(warehouse_db_path=db_path)

    assert [r.source_event_id for r in records] == ["gdelt:1"]


def test_delete_by_source_name(tmp_path: Path) -> None:
    db_path = tmp_path / "events.duckdb"
    upsert_records(
        db_path=db_path,
        records=[
            _record("acled", "acled:FRA1"),
            _record("acled_v3", "acled:EG1"),
        ],
    )
    out = delete_by_source_name(db_path=db_path, source_name="acled_v3")
    assert out["deleted"] == 1
    assert source_counts(db_path) == {"acled": 1}
    assert len(query_records(db_path=db_path)) == 1
