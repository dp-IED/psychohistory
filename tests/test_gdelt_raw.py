from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest

from ingest.gdelt_raw import (
    GDELT_V1_EVENT_COLUMNS,
    GDELT_V1_EVENT_COLUMNS_NO_SOURCEURL,
    GDELT_V2_EVENT_COLUMNS,
    fetch_arab_spring,
    fetch_france_protests,
    iter_gdelt10_daily_export_urls,
    iter_gdelt10_monthly_v1_event_urls,
    main,
    map_gdelt_event_row,
    parse_gdelt_zip_bytes,
    parse_masterfilelist,
    row_matches_arab_spring,
    row_matches_france_protest,
)


def _raw_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in GDELT_V2_EVENT_COLUMNS}
    row.update(
        {
            "GLOBALEVENTID": "1",
            "SQLDATE": "20210105",
            "MonthYear": "202101",
            "Year": "2021",
            "FractionDate": "2021.0137",
            "Actor1Name": "Protesters",
            "Actor1CountryCode": "FRA",
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
        }
    )
    row.update(overrides)
    return row


def _zip_for_rows(
    rows: list[dict[str, str]],
    *,
    v1: bool = False,
    v1_pre_daily_monthly: bool = False,
) -> bytes:
    if v1 and v1_pre_daily_monthly:
        cols: tuple[str, ...] | list[str] = GDELT_V1_EVENT_COLUMNS_NO_SOURCEURL
    else:
        cols = GDELT_V1_EVENT_COLUMNS if v1 else GDELT_V2_EVENT_COLUMNS
    payload = "\n".join(
        "\t".join(row[column] for column in cols) for row in rows
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("events.CSV", payload + "\n")
    return buffer.getvalue()


def _raw_v1_row(**overrides: str) -> dict[str, str]:
    row: dict[str, str] = {c: "" for c in GDELT_V1_EVENT_COLUMNS}
    row.update(
        {
            "GLOBALEVENTID": "1",
            "SQLDATE": "20120101",
            "MonthYear": "201201",
            "Year": "2012",
            "FractionDate": "2012.0",
            "Actor1Name": "A",
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
            "SOURCEURL": "https://e.test/1",
        }
    )
    row.update(overrides)
    return row


def test_gdelt_raw_masterfilelist_parses_export_entries() -> None:
    text = "\n".join(
        [
            "123 abc http://data.gdeltproject.org/gdeltv2/20210101000000.export.CSV.zip",
            "456 def http://data.gdeltproject.org/gdeltv2/20210101000000.mentions.CSV.zip",
            "789 ghi http://data.gdeltproject.org/gdeltv2/20210101000000.gkg.csv.zip",
        ]
    )

    entries = parse_masterfilelist(text)

    assert len(entries) == 1
    assert entries[0].expected_size == 123
    assert entries[0].expected_md5 == "abc"
    assert entries[0].url.endswith(".export.CSV.zip")
    assert entries[0].source_file_timestamp == dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)


def test_gdelt_raw_csv_row_maps_61_columns() -> None:
    row = _raw_row(Actor1Name="Students")
    metadata = {
        "_source_file_url": "file:///tmp/20210101000000.export.CSV.zip",
        "_source_file_timestamp": "2021-01-01T00:00:00Z",
        "_source_file_size": 99,
        "_source_file_md5": "abc",
        "_retrieved_at": "2021-01-01T00:01:00Z",
    }

    parsed = parse_gdelt_zip_bytes(_zip_for_rows([row]), metadata=metadata)

    assert len(parsed) == 1
    assert {column for column in GDELT_V2_EVENT_COLUMNS}.issubset(parsed[0])
    assert parsed[0]["Actor1Name"] == "Students"
    assert parsed[0]["_source_file_md5"] == "abc"


def test_gdelt_v1_column_count_is_58_and_subset_of_v2() -> None:
    assert len(GDELT_V1_EVENT_COLUMNS) == 58
    assert len(GDELT_V2_EVENT_COLUMNS) == 61
    assert not set(
        (
            "Actor1Geo_Type",
            "Actor2Geo_Type",
            "ActionGeo_Type",
        )
    ) & set(GDELT_V1_EVENT_COLUMNS)
    f = {c: "x" for c in GDELT_V1_EVENT_COLUMNS}
    m = map_gdelt_event_row(
        [f[c] for c in GDELT_V1_EVENT_COLUMNS], column_names=GDELT_V1_EVENT_COLUMNS
    )
    assert "ActionGeo_Type" not in m
    assert m["GLOBALEVENTID"] == "x"


def test_gdelt_parse_1_0_sample_zip_matches_synthetic_row() -> None:
    row = _raw_v1_row(GLOBALEVENTID="999", ActionGeo_CountryCode="EG", SQLDATE="20120101")
    buf = _zip_for_rows([row], v1=True)
    rows = parse_gdelt_zip_bytes(buf, metadata={}, gdelt_version="1.0")
    assert len(rows) == 1
    assert rows[0]["GLOBALEVENTID"] == "999"
    assert rows[0]["ActionGeo_CountryCode"] == "EG"


@pytest.mark.parametrize(
    ("cc", "sql", "lo", "hi", "expect"),
    [
        ("EG", "20120101", "2010-01-01", "2013-12-31", True),
        ("FR", "20120101", "2010-01-01", "2013-12-31", False),
        ("EG", "20150101", "2010-01-01", "2013-12-31", False),
    ],
)
def test_row_matches_arab_spring(
    cc: str, sql: str, lo: str, hi: str, expect: bool
) -> None:
    r = _raw_v1_row(
        ActionGeo_CountryCode=cc,
        SQLDATE=sql,
        ActionGeo_ADM1Code="USCA" if cc == "FR" else "EGC1",
    )
    out = row_matches_arab_spring(
        {k: str(v) for k, v in r.items() if not str(k).startswith("_")},
        event_start=dt.date.fromisoformat(lo),
        event_end=dt.date.fromisoformat(hi),
    )
    assert out is expect


def test_row_matches_arab_spring_uses_adm1_country_prefix_when_cc_is_text() -> None:
    r = _raw_v1_row(
        ActionGeo_CountryCode="Cairo, Al Qahirah, Egypt",
        ActionGeo_ADM1Code="EGC1",
        SQLDATE="20120101",
    )
    assert row_matches_arab_spring(
        {k: str(v) for k, v in r.items() if not str(k).startswith("_")},
        event_start=dt.date(2010, 1, 1),
        event_end=dt.date(2013, 12, 31),
    )


def test_row_matches_arab_spring_rejects_non_arab_spring_adm1() -> None:
    r = _raw_v1_row(
        ActionGeo_CountryCode="New York, United States",
        ActionGeo_ADM1Code="USNY",
        SQLDATE="20120101",
    )
    assert not row_matches_arab_spring(
        {k: str(v) for k, v in r.items() if not str(k).startswith("_")},
        event_start=dt.date(2010, 1, 1),
        event_end=dt.date(2013, 12, 31),
    )


def test_iter_gdelt10_daily_export_urls_yields_three_urls() -> None:
    urls = list(
        iter_gdelt10_daily_export_urls(
            dt.date(2010, 1, 1), dt.date(2010, 1, 3)
        )
    )
    assert len(urls) == 3
    assert "20100101" in urls[0] and "20100103" in urls[2]


def test_iter_gdelt10_monthly_v1_event_urls_yields_three_months() -> None:
    urls = list(
        iter_gdelt10_monthly_v1_event_urls(
            dt.date(2010, 1, 1), dt.date(2010, 3, 31)
        )
    )
    assert urls == [
        "http://data.gdeltproject.org/events/201001.zip",
        "http://data.gdeltproject.org/events/201002.zip",
        "http://data.gdeltproject.org/events/201003.zip",
    ]


def test_gdelt_raw_filter_keeps_france_protests_only() -> None:
    rows = [
        _raw_row(GLOBALEVENTID="1"),
        _raw_row(GLOBALEVENTID="2", EventRootCode="13"),
        _raw_row(GLOBALEVENTID="3", ActionGeo_CountryCode="BE"),
        _raw_row(GLOBALEVENTID="4", SQLDATE="20270101"),
    ]

    kept = [
        row
        for row in rows
        if row_matches_france_protest(
            row,
            event_start=dt.date(2019, 1, 1),
            event_end=dt.date(2026, 1, 4),
        )
    ]

    assert [row["GLOBALEVENTID"] for row in kept] == ["1"]


def test_gdelt_raw_force_refetch_removes_stale_fragment(tmp_path: Path) -> None:
    zip_bytes = _zip_for_rows([_raw_row(GLOBALEVENTID="1", SQLDATE="20210105")])
    zip_path = tmp_path / "20210101000000.export.CSV.zip"
    zip_path.write_bytes(zip_bytes)
    masterfilelist = tmp_path / "masterfilelist.txt"
    masterfilelist.write_text(
        f"{len(zip_bytes)} {hashlib.md5(zip_bytes).hexdigest()} {zip_path.as_uri()}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "raw"
    source_at = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)

    fetch_france_protests(
        masterfilelist_url=masterfilelist.as_uri(),
        event_start=dt.date(2019, 1, 1),
        event_end=dt.date(2026, 1, 4),
        source_start=source_at,
        source_end=source_at,
        out_dir=out_dir,
        workers=1,
        raw_retention="full",
    )
    fragment = out_dir / "fragments" / "2021" / "01" / "01" / "20210101000000.jsonl"
    assert fragment.exists()

    metadata = fetch_france_protests(
        masterfilelist_url=masterfilelist.as_uri(),
        event_start=dt.date(2022, 1, 1),
        event_end=dt.date(2022, 1, 2),
        source_start=source_at,
        source_end=source_at,
        out_dir=out_dir,
        workers=1,
        force=True,
        raw_retention="full",
    )

    assert metadata["kept_row_count"] == 0
    assert not fragment.exists()


def test_fetch_arab_spring_writes_filtered_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eg1 = _raw_v1_row(
        GLOBALEVENTID="1",
        SQLDATE="20130401",
        ActionGeo_CountryCode="EG",
    )
    eg2 = _raw_v1_row(
        GLOBALEVENTID="2",
        SQLDATE="20130401",
        ActionGeo_CountryCode="EG",
    )
    fr = _raw_v1_row(
        GLOBALEVENTID="3",
        SQLDATE="20130401",
        ActionGeo_CountryCode="FR",
        ActionGeo_ADM1Code="USCA",
    )
    z = _zip_for_rows([eg1, eg2, fr], v1=True)
    out_dir = tmp_path / "raw"

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        return 200, z

    monkeypatch.setattr("ingest.gdelt_raw._http_get_bytes", fake_get)
    r = fetch_arab_spring(
        event_start=dt.date(2013, 4, 1),
        event_end=dt.date(2013, 4, 1),
        out_dir=out_dir,
        workers=1,
        allow_partial=True,
    )
    frag = out_dir / "arab_spring_20130401.jsonl"
    assert frag.is_file()
    lines = [json.loads(s) for s in frag.read_text().strip().splitlines()]
    assert len(lines) == 2
    assert {x["ActionGeo_CountryCode"] for x in lines} == {"EG"}
    assert r.get("context") == "arab_spring"
    assert (out_dir / "fetch_manifest.json").is_file()


def test_fetch_arab_spring_monthly_writes_filtered_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eg1 = _raw_v1_row(
        GLOBALEVENTID="1",
        SQLDATE="20110101",
        ActionGeo_CountryCode="EG",
    )
    eg2 = _raw_v1_row(
        GLOBALEVENTID="2",
        SQLDATE="20110101",
        ActionGeo_CountryCode="EG",
    )
    fr = _raw_v1_row(
        GLOBALEVENTID="3",
        SQLDATE="20110101",
        ActionGeo_CountryCode="FR",
        ActionGeo_ADM1Code="USCA",
    )
    z = _zip_for_rows([eg1, eg2, fr], v1=True, v1_pre_daily_monthly=True)
    out_dir = tmp_path / "raw"

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        assert u == "http://data.gdeltproject.org/events/201101.zip"
        return 200, z

    monkeypatch.setattr("ingest.gdelt_raw._http_get_bytes", fake_get)
    r = fetch_arab_spring(
        event_start=dt.date(2011, 1, 1),
        event_end=dt.date(2011, 1, 31),
        out_dir=out_dir,
        workers=4,
        allow_partial=True,
    )
    frag = out_dir / "arab_spring_201101_monthly.jsonl"
    assert frag.is_file()
    lines = [json.loads(s) for s in frag.read_text().strip().splitlines()]
    assert len(lines) == 2
    assert {x["ActionGeo_CountryCode"] for x in lines} == {"EG"}
    assert r.get("annual_files_fetched") == 1
    assert r.get("daily_files_fetched") == 0
    assert r.get("files_fetched") == 1
    assert r.get("days_completed") == 0


def test_fetch_arab_spring_monthly_respects_existing_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arab_spring_201101_monthly.jsonl").write_text(
        json.dumps({"GLOBALEVENTID": "1"}) + "\n",
        encoding="utf-8",
    )

    def fail_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        raise AssertionError(
            f"_http_get_bytes should not be called for existing monthly fragment: {u}"
        )

    monkeypatch.setattr("ingest.gdelt_raw._http_get_bytes", fail_get)
    r = fetch_arab_spring(
        event_start=dt.date(2011, 1, 1),
        event_end=dt.date(2011, 1, 31),
        out_dir=out_dir,
        workers=1,
        force=False,
        allow_partial=True,
    )
    assert r.get("annual_files_fetched") == 1
    assert r.get("daily_files_fetched") == 0
    assert r.get("days_completed") == 0


def test_fetch_arab_spring_force_daily_removes_stale_fragment_when_refetch_keeps_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "arab_spring_20130401.jsonl"
    stale.write_text(json.dumps({"stale": "row"}) + "\n", encoding="utf-8")
    us_row = _raw_v1_row(
        GLOBALEVENTID="9",
        SQLDATE="20130401",
        ActionGeo_CountryCode="FR",
        ActionGeo_ADM1Code="USCA",
    )
    z = _zip_for_rows([us_row], v1=True)

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        assert u == "http://data.gdeltproject.org/events/20130401.export.CSV.zip"
        return 200, z

    monkeypatch.setattr("ingest.gdelt_raw._http_get_bytes", fake_get)
    r = fetch_arab_spring(
        event_start=dt.date(2013, 4, 1),
        event_end=dt.date(2013, 4, 1),
        out_dir=out_dir,
        workers=1,
        force=True,
        allow_partial=True,
    )
    assert not stale.exists()
    assert r.get("rows_written") == 0


def test_fetch_arab_spring_force_monthly_removes_stale_fragment_when_refetch_keeps_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "arab_spring_201101_monthly.jsonl"
    stale.write_text(json.dumps({"stale": "row"}) + "\n", encoding="utf-8")
    us_row = _raw_v1_row(
        GLOBALEVENTID="9",
        SQLDATE="20110101",
        ActionGeo_CountryCode="FR",
        ActionGeo_ADM1Code="USCA",
    )
    z = _zip_for_rows([us_row], v1=True, v1_pre_daily_monthly=True)

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        assert u == "http://data.gdeltproject.org/events/201101.zip"
        return 200, z

    monkeypatch.setattr("ingest.gdelt_raw._http_get_bytes", fake_get)
    r = fetch_arab_spring(
        event_start=dt.date(2011, 1, 1),
        event_end=dt.date(2011, 1, 31),
        out_dir=out_dir,
        workers=1,
        force=True,
        allow_partial=True,
    )
    assert not stale.exists()
    assert r.get("rows_written") == 0


def _v1_monthly_zip_bytes(yyyymm: str, rows: list[dict[str, str]]) -> bytes:
    """Mirror GDELT pre-daily monthly zips: one ``{yyyymm}.csv`` member, 57 columns, no SOURCEURL."""
    cols: tuple[str, ...] = GDELT_V1_EVENT_COLUMNS_NO_SOURCEURL
    payload = "\n".join("\t".join(str(row.get(c, "")) for c in cols) for row in rows)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{yyyymm}.csv", payload + "\n")
    return buffer.getvalue()


def test_main_cli_fetch_arab_spring_march_2010_monthly_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ingest.gdelt_raw as gdelt_raw

    out_dir = tmp_path / "arab_spring_out"
    in_egypt = _raw_v1_row(
        GLOBALEVENTID="1",
        SQLDATE="20100310",
        ActionGeo_CountryCode="Cairo, Al Qahirah, Egypt",
        ActionGeo_ADM1Code="EGC1",
    )
    in_us = _raw_v1_row(
        GLOBALEVENTID="2",
        SQLDATE="20100310",
        ActionGeo_CountryCode="United States",
        ActionGeo_ADM1Code="USCA",
    )
    z = _v1_monthly_zip_bytes("201003", [in_egypt, in_us])

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        if u == "http://data.gdeltproject.org/events/201003.zip":
            return 200, z
        return 404, None

    monkeypatch.setattr(gdelt_raw, "_http_get_bytes", fake_get)
    ret = main(
        [
            "fetch-arab-spring",
            "--raw-dir",
            str(out_dir),
            "--event-start",
            "2010-03-01",
            "--event-end",
            "2010-03-31",
        ]
    )
    assert ret == 0
    frag = out_dir / "arab_spring_201003_monthly.jsonl"
    assert frag.is_file()
    lines = [json.loads(s) for s in frag.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["GLOBALEVENTID"] == "1"
    man = json.loads((out_dir / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert man["rows_written"] == 1
    assert man.get("not_found_404_count", 0) == 0


def test_main_cli_fetch_arab_spring_april_2013_daily_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ingest.gdelt_raw as gdelt_raw

    out_dir = tmp_path / "arab_daily_out"
    eg1 = _raw_v1_row(
        GLOBALEVENTID="1",
        SQLDATE="20130401",
        ActionGeo_CountryCode="EG",
        ActionGeo_ADM1Code="EGC1",
    )
    eg2 = _raw_v1_row(
        GLOBALEVENTID="2",
        SQLDATE="20130401",
        ActionGeo_CountryCode="EG",
        ActionGeo_ADM1Code="EGC1",
    )
    fr = _raw_v1_row(
        GLOBALEVENTID="3",
        SQLDATE="20130401",
        ActionGeo_CountryCode="FR",
        ActionGeo_ADM1Code="USCA",
    )
    z = _zip_for_rows([eg1, eg2, fr], v1=True)

    def fake_get(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        if u == "http://data.gdeltproject.org/events/20130401.export.CSV.zip":
            return 200, z
        return 404, None

    monkeypatch.setattr(gdelt_raw, "_http_get_bytes", fake_get)
    ret = main(
        [
            "fetch-arab-spring",
            "--raw-dir",
            str(out_dir),
            "--event-start",
            "2013-04-01",
            "--event-end",
            "2013-04-01",
        ]
    )
    assert ret == 0
    frag = out_dir / "arab_spring_20130401.jsonl"
    assert frag.is_file()
    assert len(frag.read_text().strip().splitlines()) == 2


def test_main_cli_fetch_arab_spring_exits_1_on_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ingest.gdelt_raw as gdelt_raw

    def always_fail(
        u: str, *, max_retries: int, retry_backoff_seconds: float
    ) -> tuple[int, bytes | None]:
        return 503, None

    monkeypatch.setattr(gdelt_raw, "_http_get_bytes", always_fail)
    ret = main(
        [
            "fetch-arab-spring",
            "--raw-dir",
            str(tmp_path / "out"),
            "--event-start",
            "2010-03-01",
            "--event-end",
            "2010-03-31",
        ]
    )
    assert ret == 1


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gdelt_live
def test_subprocess_fetch_arab_spring_one_real_month_produces_non_empty_fragment(
    tmp_path: Path,
) -> None:
    """``python -m ingest.gdelt_raw fetch-arab-spring`` against one public monthly zip; opt-in (slow)."""
    out = tmp_path / "gdelt_arab_spring"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ingest.gdelt_raw",
            "fetch-arab-spring",
            "--raw-dir",
            str(out),
            "--event-start",
            "2010-03-01",
            "--event-end",
            "2010-03-31",
            "--allow-partial",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, f"stderr={r.stderr!r} stdout={r.stdout!r}"
    frag = out / "arab_spring_201003_monthly.jsonl"
    assert frag.is_file()
    n_kept = sum(1 for line in frag.open(encoding="utf-8") if line.strip())
    assert n_kept > 0, "expected at least one filtered row from real 201003.zip"
