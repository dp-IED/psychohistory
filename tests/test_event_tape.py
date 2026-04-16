from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.event_tape import EventTapeRecord, normalize_raw_row, write_event_tape
from ingest.gdelt_raw import GDELT_V2_EVENT_COLUMNS


def _raw_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in GDELT_V2_EVENT_COLUMNS}
    row.update(
        {
            "GLOBALEVENTID": "100",
            "SQLDATE": "20210105",
            "Actor1Name": "Protesters",
            "EventCode": "141",
            "EventBaseCode": "14",
            "EventRootCode": "14",
            "QuadClass": "3",
            "GoldsteinScale": "-6.5",
            "NumMentions": "4",
            "NumSources": "2",
            "NumArticles": "3",
            "AvgTone": "-1.2",
            "ActionGeo_FullName": "Paris, Ile-de-France, France",
            "ActionGeo_CountryCode": "FR",
            "ActionGeo_ADM1Code": "FR11",
            "ActionGeo_Lat": "48.8566",
            "ActionGeo_Long": "2.3522",
            "DATEADDED": "20210105120000",
            "SOURCEURL": "https://example.test/story",
            "_retrieved_at": "2021-01-05T12:05:00Z",
            "_source_file_timestamp": "2021-01-05T12:00:00Z",
            "_source_file_url": "https://example.test/20210105120000.export.CSV.zip",
        }
    )
    row.update(overrides)
    return row


def _write_raw_fragment(
    raw_dir: Path,
    rows: list[dict[str, str]],
    *,
    relative_path: str = "fragments/2021/01/05/20210105120000.jsonl",
) -> None:
    fragment = raw_dir / relative_path
    fragment.parent.mkdir(parents=True, exist_ok=True)
    with fragment.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_gdelt_row_normalization() -> None:
    record = normalize_raw_row(
        _raw_row(
            ActionGeo_ADM1Code="",
            Actor2Name="",
            Actor2CountryCode="",
            NumSources="",
        )
    )

    assert record is not None
    assert record.source_event_id == "gdelt:100"
    assert record.event_date.isoformat() == "2021-01-05"
    assert record.source_available_at.isoformat().startswith("2021-01-05T12:00:00")
    assert record.event_class == "protest"
    assert record.admin1_code == "FR_UNKNOWN"
    assert record.num_sources is None
    assert record.actor2_name is None


def test_event_tape_deduplicates_by_source_event_id(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "tape" / "events.jsonl"
    _write_raw_fragment(
        raw_dir,
        [
            _raw_row(DATEADDED="20210106120000", _source_file_timestamp="2021-01-06T12:00:00Z"),
            _raw_row(DATEADDED="20210105120000", _source_file_timestamp="2021-01-05T12:00:00Z"),
        ],
    )

    audit = write_event_tape(raw_dir=raw_dir, out_path=out_path)
    records = [EventTapeRecord.model_validate_json(line) for line in out_path.read_text().splitlines()]

    assert len(records) == 1
    assert records[0].source_available_at.isoformat().startswith("2021-01-05T12:00:00")
    assert audit["duplicate_count"] == 1


def test_event_tape_missing_raw_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing raw directory"):
        write_event_tape(raw_dir=tmp_path / "missing", out_path=tmp_path / "events.jsonl")


def test_event_tape_reads_only_current_manifest_run(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "tape" / "events.jsonl"
    stale_fragment = "fragments/2021/01/04/20210104120000.jsonl"
    current_fragment = "fragments/2021/01/05/20210105120000.jsonl"
    _write_raw_fragment(raw_dir, [_raw_row(GLOBALEVENTID="stale")], relative_path=stale_fragment)
    _write_raw_fragment(raw_dir, [_raw_row(GLOBALEVENTID="current")], relative_path=current_fragment)
    (raw_dir / "fetch_metadata.json").write_text(
        json.dumps({"run_id": "current-run"}) + "\n",
        encoding="utf-8",
    )
    manifest_rows = [
        {
            "run_id": "old-run",
            "status": "ok",
            "fragment_path": stale_fragment,
        },
        {
            "run_id": "current-run",
            "status": "ok",
            "fragment_path": current_fragment,
        },
    ]
    (raw_dir / "fetch_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    write_event_tape(raw_dir=raw_dir, out_path=out_path)
    records = [EventTapeRecord.model_validate_json(line) for line in out_path.read_text().splitlines()]

    assert [record.source_event_id for record in records] == ["gdelt:current"]


def test_event_tape_rejects_failed_fetch_without_allow_partial(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    fragment = "fragments/2021/01/05/20210105120000.jsonl"
    _write_raw_fragment(raw_dir, [_raw_row()], relative_path=fragment)
    (raw_dir / "fetch_metadata.json").write_text(
        json.dumps({"run_id": "current-run", "failed_file_count": 1, "allow_partial": False})
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "fetch_manifest.jsonl").write_text(
        json.dumps({"run_id": "current-run", "status": "ok", "fragment_path": fragment})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw fetch has failed files"):
        write_event_tape(raw_dir=raw_dir, out_path=tmp_path / "events.jsonl")


def test_event_tape_allows_failed_fetch_when_partial_is_explicit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "events.jsonl"
    fragment = "fragments/2021/01/05/20210105120000.jsonl"
    _write_raw_fragment(raw_dir, [_raw_row()], relative_path=fragment)
    (raw_dir / "fetch_metadata.json").write_text(
        json.dumps({"run_id": "current-run", "failed_file_count": 1, "allow_partial": False})
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "fetch_manifest.jsonl").write_text(
        json.dumps({"run_id": "current-run", "status": "ok", "fragment_path": fragment})
        + "\n",
        encoding="utf-8",
    )

    audit = write_event_tape(raw_dir=raw_dir, out_path=out_path, allow_partial=True)

    assert audit["output_row_count"] == 1
    assert out_path.exists()
