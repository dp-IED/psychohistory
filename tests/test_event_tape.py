from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from ingest.event_tape import (
    ARAB_SPRING_GDELT_NORMALIZE,
    FRANCE_PROTEST_GDELT_NORMALIZE,
    EventTapeRecord,
    load_event_tape,
    normalize_acled_arab_spring_row,
    normalize_raw_row,
    write_arab_spring_merged_tape,
    write_event_tape,
)
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
        ),
        country_codes=FRANCE_PROTEST_GDELT_NORMALIZE.country_codes,
        event_root_codes=FRANCE_PROTEST_GDELT_NORMALIZE.event_root_codes,
        event_start=FRANCE_PROTEST_GDELT_NORMALIZE.event_start,
        event_end=FRANCE_PROTEST_GDELT_NORMALIZE.event_end,
    )

    assert record is not None
    assert record.source_event_id == "gdelt:100"
    assert record.event_date.isoformat() == "2021-01-05"
    assert record.source_available_at.isoformat().startswith("2021-01-05T12:00:00")
    assert record.event_root_code == "14"
    assert record.admin1_code == "FR_UNKNOWN"
    assert record.num_sources is None
    assert record.actor2_name is None


def test_event_tape_record_accepts_acled_source() -> None:
    record = EventTapeRecord(
        source_name="acled",
        source_event_id="acled:FRA123",
        event_date=dt.date(2021, 1, 5),
        source_available_at=dt.datetime(2021, 1, 6, tzinfo=dt.timezone.utc),
        retrieved_at=dt.datetime(2021, 1, 6, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code="FR11",
        location_name="Ile-de-France",
        latitude=48.8566,
        longitude=2.3522,
        event_code="Protests",
        event_base_code="Protests",
        event_root_code="Protests",
        quad_class=None,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name="Protesters",
        actor1_country_code="FRA",
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={"event_id_cnty": "FRA123"},
    )

    assert record.source_name == "acled"


def test_gdelt_arab_spring_country_and_all_roots() -> None:
    eg = normalize_raw_row(
        _raw_row(
            ActionGeo_CountryCode="EG",
            ActionGeo_ADM1Code="EG11",
            EventRootCode="18",
            SQLDATE="20110115",
        ),
        country_codes=frozenset({"EG", "TU", "LY", "SY"}),
        event_root_codes=None,
        event_start=dt.date(2010, 1, 1),
        event_end=dt.date(2013, 12, 31),
    )
    assert eg is not None
    assert eg.country_code == "EG"
    assert eg.event_root_code == "18"
    blocked = normalize_raw_row(
        _raw_row(ActionGeo_CountryCode="FR", SQLDATE="20110115"),
        country_codes=frozenset({"EG", "TU", "LY", "SY"}),
        event_root_codes=None,
        event_start=dt.date(2010, 1, 1),
        event_end=dt.date(2013, 12, 31),
    )
    assert blocked is None


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

    audit = write_event_tape(
        raw_dir=raw_dir,
        out_path=out_path,
        gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
    )
    records = [EventTapeRecord.model_validate_json(line) for line in out_path.read_text().splitlines()]

    assert len(records) == 1
    assert records[0].source_available_at.isoformat().startswith("2021-01-05T12:00:00")
    assert audit["duplicate_count"] == 1


def test_event_tape_missing_raw_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing raw directory"):
        write_event_tape(
            raw_dir=tmp_path / "missing",
            out_path=tmp_path / "events.jsonl",
            gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
        )


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

    write_event_tape(
        raw_dir=raw_dir,
        out_path=out_path,
        gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
    )
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
        write_event_tape(
            raw_dir=raw_dir,
            out_path=tmp_path / "events.jsonl",
            gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
        )


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

    audit = write_event_tape(
        raw_dir=raw_dir,
        out_path=out_path,
        gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
        allow_partial=True,
    )

    assert audit["output_row_count"] == 1
    assert out_path.exists()


def test_normalize_acled_arab_spring_row() -> None:
    row = {
        "event_id_cnty": "EGY123",
        "event_date": "2011-06-01",
        "country": "Egypt",
        "admin1": "Cairo",
        "actor1": "Protesters",
        "actor2": "Police",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "fatalities": "0",
        "notes": "",
        "_retrieved_at": "2011-06-02T12:00:00Z",
    }
    rec = normalize_acled_arab_spring_row(row)
    assert rec.source_name == "acled_v3"
    assert rec.country_code == "EG"
    assert rec.event_root_code == "Protests"
    assert rec.source_available_at.isoformat().startswith("2011-06-01T23:59:59")


def test_merge_arab_spring_tape_dedupes(tmp_path: Path) -> None:
    gd = tmp_path / "gdelt"
    ac = tmp_path / "acled"
    gd.mkdir()
    ac.mkdir()
    gfrag = gd / "arab_spring_20110101_000000.jsonl"
    shared_id = "999"
    gfrag.write_text(
        json.dumps(
            _raw_row(
                GLOBALEVENTID=shared_id,
                SQLDATE="20110105",
                ActionGeo_CountryCode="EG",
                ActionGeo_ADM1Code="EG11",
                DATEADDED="20110105120000",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    afrag = ac / "acled_arab_spring_page_0001.jsonl"
    afrag.write_text(
        json.dumps(
            {
                "event_id_cnty": "EGY999",
                "event_date": "2011-01-05",
                "country": "Egypt",
                "admin1": "Cairo",
                "actor1": "x",
                "actor2": "",
                "event_type": "Protests",
                "sub_event_type": "Peaceful protest",
                "fatalities": "0",
                "notes": "",
                "_retrieved_at": "2011-01-06T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "events.jsonl"
    write_arab_spring_merged_tape(
        gdelt_raw_dir=gd,
        acled_raw_dir=ac,
        out_path=out,
        gdelt_normalize=ARAB_SPRING_GDELT_NORMALIZE,
        cleanup_fragments=False,
    )
    records = load_event_tape(out)
    assert len(records) == 2


def test_load_event_tape_reads_gzip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "events.jsonl.gz"
    _write_raw_fragment(raw_dir, [_raw_row()])

    write_event_tape(
        raw_dir=raw_dir,
        out_path=out_path,
        gdelt_normalize=FRANCE_PROTEST_GDELT_NORMALIZE,
    )

    records = load_event_tape(out_path)
    assert len(records) == 1
    assert records[0].source_event_id == "gdelt:100"
