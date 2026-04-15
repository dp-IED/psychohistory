from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile
from pathlib import Path

from ingest.gdelt_raw import (
    GDELT_V2_EVENT_COLUMNS,
    fetch_france_protests,
    parse_gdelt_zip_bytes,
    parse_masterfilelist,
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


def _zip_for_rows(rows: list[dict[str, str]]) -> bytes:
    payload = "\n".join(
        "\t".join(row[column] for column in GDELT_V2_EVENT_COLUMNS) for row in rows
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("events.CSV", payload + "\n")
    return buffer.getvalue()


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
    )

    assert metadata["kept_row_count"] == 0
    assert not fragment.exists()
