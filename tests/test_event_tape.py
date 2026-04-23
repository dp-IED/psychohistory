from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pytest

from ingest.event_tape import (
    EventTapeRecord,
    KNOWN_BAD_FRAGMENTS,
    audit_gdelt_arab_spring_raw_normalization,
    load_event_tape,
    normalize_gdelt_arab_spring_row,
    normalize_raw_row,
    write_arab_spring_merged_tape,
    write_event_tape,
)
from ingest.gdelt_raw import (
    GDELT_V1_EVENT_COLUMNS,
    GDELT_V2_EVENT_COLUMNS,
    format_datetime_z,
    utc_now,
)


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
        event_class="protest",
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


def test_load_event_tape_reads_gzip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "events.jsonl.gz"
    _write_raw_fragment(raw_dir, [_raw_row()])

    write_event_tape(raw_dir=raw_dir, out_path=out_path)

    records = load_event_tape(out_path)
    assert len(records) == 1
    assert records[0].source_event_id == "gdelt:100"


def _arab_gdelt_jsonl_row(gid: str) -> dict[str, str]:
    row: dict[str, str] = {c: "" for c in GDELT_V1_EVENT_COLUMNS}
    row.update(
        {
            "GLOBALEVENTID": gid,
            "SQLDATE": "20120101",
            "MonthYear": "201201",
            "Year": "2012",
            "FractionDate": "2012.0",
            "EventCode": "10",
            "EventBaseCode": "1",
            "EventRootCode": "1",
            "QuadClass": "0",
            "ActionGeo_FullName": "Cairo",
            "ActionGeo_CountryCode": "EG",
            "ActionGeo_ADM1Code": "EGC1",
            "ActionGeo_Lat": "30",
            "ActionGeo_Long": "31",
            "DATEADDED": "20120101120000",
            "SOURCEURL": "https://e.test/x",
        }
    )
    row["_retrieved_at"] = format_datetime_z(utc_now())
    row["_source_file_url"] = "http://data.gdeltproject.org/events/20120101.export.CSV.zip"
    row["_source_file_timestamp"] = "2012-01-01T00:00:00Z"
    return row


def test_normalize_gdelt_arab_spring_row_keeps_adm1_matched_text_country_code() -> None:
    row = _arab_gdelt_jsonl_row("991")
    row["ActionGeo_CountryCode"] = "Cairo, Al Qahirah, Egypt"
    row["ActionGeo_ADM1Code"] = "EGC1"
    rec = normalize_gdelt_arab_spring_row(
        row,
        event_start=dt.date(2012, 1, 1),
        event_end=dt.date(2012, 12, 31),
    )
    assert rec is not None
    assert rec.country_code == "EG"
    assert rec.admin1_code == "EGC1"


def test_normalize_gdelt_arab_spring_row_rejects_non_region_adm1() -> None:
    row = _arab_gdelt_jsonl_row("992")
    row["ActionGeo_CountryCode"] = "California, United States"
    row["ActionGeo_ADM1Code"] = "USCA"
    assert (
        normalize_gdelt_arab_spring_row(
            row,
            event_start=dt.date(2012, 1, 1),
            event_end=dt.date(2012, 12, 31),
        )
        is None
    )


def _acled_page_line(eid: str, row_num: int) -> dict[str, str | int]:
    return {
        "event_id_cnty": eid,
        "event_date": "2012-01-01",
        "year": "2012",
        "time_precision": "1",
        "disorder_type": "",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "actor1": "x",
        "assoc_actor_1": "",
        "inter1": "",
        "actor2": "",
        "assoc_actor_2": "",
        "inter2": "",
        "interaction": "",
        "civilian_targeting": "",
        "iso": "818",
        "region": "",
        "country": "Egypt",
        "admin1": "Cairo",
        "admin2": "",
        "admin3": "",
        "location": "Cairo",
        "latitude": "30",
        "longitude": "31",
        "geo_precision": "1",
        "source": "test",
        "source_scale": "",
        "notes": "",
        "fatalities": "0",
        "tags": "",
        "timestamp": "",
        "_retrieved_at": format_datetime_z(utc_now()),
        "_csv_input_file": "t.csv",
        "_csv_row": row_num,
    }


def test_write_arab_spring_merged_tape_dedup_and_manifest(tmp_path: Path) -> None:
    gdir = tmp_path / "gd"
    gdir.mkdir()
    frag = gdir / "arab_spring_20120101.jsonl"
    lines = [
        _arab_gdelt_jsonl_row("10"),
        _arab_gdelt_jsonl_row("20"),
        _arab_gdelt_jsonl_row("30"),
    ]
    with frag.open("w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
    (gdir / "fetch_manifest.json").write_text(
        json.dumps(
            {
                "date_start": "2012-01-01",
                "date_end": "2012-12-31",
                "rows_written": 3,
            }
        ),
        encoding="utf-8",
    )
    aroot = tmp_path / "ac"
    (aroot / "fragments").mkdir(parents=True)
    ap = aroot / "fragments" / "page_000001.jsonl"
    dup = "EGDUP1"
    with ap.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_acled_page_line(dup, 0)) + "\n")
        f.write(json.dumps(_acled_page_line(dup, 1)) + "\n")
    (aroot / "fetch_metadata.json").write_text(
        json.dumps({"accepted_row_count": 2}),
        encoding="utf-8",
    )
    out = tmp_path / "out" / "events.jsonl"
    res = write_arab_spring_merged_tape(
        gdelt_raw_dir=gdir,
        acled_raw_dir=aroot,
        out_path=out,
        allow_empty=False,
    )
    recs = load_event_tape(out)
    assert len(recs) == 4
    assert res["dedup_dropped"] == 1
    assert res["total_record_count"] == 4
    man = json.loads((out.parent / "tape_manifest.json").read_text())
    assert man["dedup_dropped"] == 1
    assert man["total_record_count"] == 4
    assert man["gdelt_record_count"] == 3
    assert man["acled_record_count"] == 2


def test_audit_gdelt_arab_spring_raw_normalization_per_month_and_reasons(
    tmp_path: Path,
) -> None:
    gdir = tmp_path / "gd"
    gdir.mkdir()
    (gdir / "fetch_manifest.json").write_text(
        json.dumps(
            {
                "date_start": "2012-01-01",
                "date_end": "2012-12-31",
            }
        ),
        encoding="utf-8",
    )
    r_ok = _arab_gdelt_jsonl_row("1")
    r_geo = _arab_gdelt_jsonl_row("2")
    r_geo["ActionGeo_CountryCode"] = "US"
    r_geo["ActionGeo_ADM1Code"] = "USCA"
    r_date = _arab_gdelt_jsonl_row("3")
    r_date["SQLDATE"] = "20010101"
    r_date["MonthYear"] = "200101"
    for name, lines in [
        (
            "arab_spring_201201_monthly.jsonl",
            [r_ok, r_geo, r_date],
        ),
        (
            "arab_spring_20120201.jsonl",
            [_arab_gdelt_jsonl_row("4")],
        ),
    ]:
        with (gdir / name).open("w", encoding="utf-8") as f:
            for d in lines:
                f.write(json.dumps(d) + "\n")

    rep = audit_gdelt_arab_spring_raw_normalization(
        gdir, flag_min_raw=1, flag_drop_rate_ge=0.0001
    )
    assert rep["totals"]["raw_lines"] == 4
    assert rep["totals"]["ok"] == 2
    assert rep["totals"]["filtered_geo"] == 1
    assert rep["totals"]["filtered_date"] == 1
    assert rep["totals"].get("skipped_known_bad", 0) == 0
    by_m = rep["by_month"]
    assert by_m["2012-01"]["raw_lines"] == 3
    assert by_m["2012-01"]["ok"] == 1
    assert by_m["2012-02"]["ok"] == 1
    assert any(f["calendar_month"] == "2012-01" for f in rep["flags"])


def test_audit_gdelt_skips_known_bad_fragment_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    gdir = tmp_path / "gd"
    gdir.mkdir()
    (gdir / "fetch_manifest.json").write_text(
        json.dumps({"date_start": "2012-01-01", "date_end": "2012-12-31"}),
        encoding="utf-8",
    )
    bad_name = "arab_spring_20130901.jsonl"
    assert bad_name in KNOWN_BAD_FRAGMENTS
    with (gdir / bad_name).open("w", encoding="utf-8") as f:
        f.write('{"not":"a","valid":"row"}\n')
        f.write("{}\n")
    rep = audit_gdelt_arab_spring_raw_normalization(
        gdir, flag_min_raw=10_000, flag_drop_rate_ge=0.005
    )
    assert rep["totals"]["raw_lines"] == 2
    assert rep["totals"]["invalid"] == 0
    assert rep["totals"]["ok"] == 0
    assert rep["totals"]["skipped_known_bad"] == 2
    assert any("known-bad" in r.getMessage().lower() for r in caplog.records)
