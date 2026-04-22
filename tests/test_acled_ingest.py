from __future__ import annotations

import csv
import datetime as dt
import io
import json
from pathlib import Path

from ingest.acled_raw import (
    ACLED_FIELDS,
    AcledCredentials,
    _expand_acled_csv_country_allow,
    fetch_france_protests,
    ingest_acled_csv,
)
from ingest.acled_tape import (
    acled_event_date_end_utc,
    acled_admin_to_gdelt_admin1_code,
    acled_source_available_at,
    normalize_acled_csv_row,
    normalize_acled_row,
    write_acled_event_tape,
)
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import query_records
from ingest.event_jsonl_merge import merge_event_jsonl


def _acled_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id_cnty": "FRA123",
        "event_date": "2023-03-25",
        "year": "2023",
        "time_precision": "1",
        "disorder_type": "Demonstrations",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "actor1": "Protesters (France)",
        "assoc_actor_1": "",
        "inter1": "6",
        "actor2": "",
        "assoc_actor_2": "",
        "inter2": "",
        "interaction": "",
        "civilian_targeting": "",
        "iso": "250",
        "region": "Europe",
        "country": "France",
        "admin1": "Nouvelle-Aquitaine",
        "admin2": "Deux-Sèvres",
        "admin3": "",
        "location": "Sainte-Soline",
        "latitude": "46.3580",
        "longitude": "-0.0820",
        "geo_precision": "1",
        "source": "Example Source",
        "source_scale": "National",
        "notes": "Example note",
        "fatalities": "0",
        "tags": "",
        "timestamp": "1679875200",
        "_retrieved_at": "2023-03-27T00:00:00Z",
    }
    row.update(overrides)
    return row


def test_acled_admin_mapping_prefers_department_names() -> None:
    assert acled_admin_to_gdelt_admin1_code("Nouvelle-Aquitaine", "Deux-Sèvres") == "FRB7"
    assert acled_admin_to_gdelt_admin1_code("Île-de-France", "Paris") == "FRA8"
    assert acled_admin_to_gdelt_admin1_code("Not Mapped", "Not Mapped") == "FR_UNKNOWN"


def test_normalize_acled_row_outputs_event_tape_record() -> None:
    record = normalize_acled_row(_acled_row())

    assert record is not None
    assert record.source_name == "acled"
    assert record.source_event_id == "acled:FRA123"
    assert record.admin1_code == "FRB7"
    assert record.event_class == "protest"
    assert record.quad_class is None
    assert record.source_available_at == dt.datetime(2023, 4, 1, tzinfo=dt.timezone.utc)


def test_acled_availability_policies_are_explicit() -> None:
    row = _acled_row(timestamp="1679875200")
    event_date = dt.date(2023, 3, 25)
    retrieved_at = dt.datetime(2026, 4, 16, tzinfo=dt.timezone.utc)

    assert acled_source_available_at(
        row,
        event_date=event_date,
        retrieved_at=retrieved_at,
        availability_policy="timestamp",
        availability_lag_days=7,
    ) == dt.datetime(2023, 3, 27, tzinfo=dt.timezone.utc)
    assert acled_source_available_at(
        row,
        event_date=event_date,
        retrieved_at=retrieved_at,
        availability_policy="event_date_lag",
        availability_lag_days=7,
    ) == dt.datetime(2023, 4, 1, tzinfo=dt.timezone.utc)
    assert acled_source_available_at(
        row,
        event_date=event_date,
        retrieved_at=retrieved_at,
        availability_policy="retrieved_at",
        availability_lag_days=7,
    ) == retrieved_at


def test_write_acled_event_tape_reads_manifest_fragments(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    fragment = raw_dir / "fragments" / "page_000001.jsonl"
    fragment.parent.mkdir(parents=True)
    fragment.write_text(json.dumps(_acled_row()) + "\n", encoding="utf-8")
    (raw_dir / "fetch_metadata.json").write_text(
        json.dumps({"run_id": "run-1", "failed_page_count": 0}) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "fetch_manifest.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "ok",
                "fragment_path": "fragments/page_000001.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "tape" / "events.jsonl"
    audit = write_acled_event_tape(raw_dir=raw_dir, out_path=out_path)

    records = [EventTapeRecord.model_validate_json(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0].source_name == "acled"
    assert audit["output_row_count"] == 1
    assert audit["availability_policy"] == "event_date_lag"
    assert audit["availability_lag_days"] == 7
    assert audit["unresolved_admin1_counts"] == {}


def test_acled_raw_fetch_writes_pages_without_secrets(tmp_path: Path, monkeypatch) -> None:
    pages = {
        1: [{"event_id_cnty": "FRA1"}],
        2: [],
    }

    def fake_credentials_from_env() -> AcledCredentials:
        return AcledCredentials(username="user@example.test", password="secret")

    def fake_get_access_token(**_kwargs) -> str:
        return "token"

    def fake_fetch_page(*, params: dict[str, str], **_kwargs):
        page = int(params["page"])
        return {"status": 200, "success": True, "last_update": 1, "data": pages[page]}

    monkeypatch.setattr("ingest.acled_raw.credentials_from_env", fake_credentials_from_env)
    monkeypatch.setattr("ingest.acled_raw.get_access_token", fake_get_access_token)
    monkeypatch.setattr("ingest.acled_raw._fetch_page", fake_fetch_page)

    audit = fetch_france_protests(
        event_start=dt.date(2023, 3, 20),
        event_end=dt.date(2023, 4, 3),
        out_dir=tmp_path / "raw",
        limit=1,
        max_pages=5,
    )

    manifest_text = (tmp_path / "raw" / "fetch_manifest.jsonl").read_text(encoding="utf-8")
    metadata_text = (tmp_path / "raw" / "fetch_metadata.json").read_text(encoding="utf-8")
    assert audit["row_count"] == 1
    assert "secret" not in manifest_text
    assert "token" not in manifest_text
    assert "secret" not in metadata_text
    assert "token" not in metadata_text


def test_acled_raw_fetch_normalizes_to_warehouse_without_fragments(tmp_path: Path, monkeypatch) -> None:
    pages = {
        1: [_acled_row(event_id_cnty="FRA1")],
        2: [],
    }

    def fake_credentials_from_env() -> AcledCredentials:
        return AcledCredentials(username="user@example.test", password="secret")

    def fake_get_access_token(**_kwargs) -> str:
        return "token"

    def fake_fetch_page(*, params: dict[str, str], **_kwargs):
        page = int(params["page"])
        return {"status": 200, "success": True, "last_update": 1, "data": pages[page]}

    monkeypatch.setattr("ingest.acled_raw.credentials_from_env", fake_credentials_from_env)
    monkeypatch.setattr("ingest.acled_raw.get_access_token", fake_get_access_token)
    monkeypatch.setattr("ingest.acled_raw._fetch_page", fake_fetch_page)

    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse" / "events.duckdb"
    audit = fetch_france_protests(
        event_start=dt.date(2023, 3, 20),
        event_end=dt.date(2023, 4, 3),
        out_dir=raw_dir,
        limit=1,
        max_pages=5,
        raw_retention="none",
        normalize_to_warehouse=True,
        warehouse_path=db_path,
        availability_policy="event_date_lag",
        availability_lag_days=7,
    )

    manifest_text = (raw_dir / "fetch_manifest.jsonl").read_text(encoding="utf-8")
    records = query_records(db_path=db_path, source_names={"acled"})
    assert audit["normalized_row_count"] == 1
    assert not (raw_dir / "fragments").exists()
    assert [record.source_event_id for record in records] == ["acled:FRA1"]
    assert "secret" not in manifest_text
    assert "token" not in manifest_text


def test_merge_event_jsonl_writes_source_counts(tmp_path: Path) -> None:
    gdelt = EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id="gdelt:1",
        event_date=dt.date(2023, 3, 25),
        source_available_at=dt.datetime(2023, 3, 26, tzinfo=dt.timezone.utc),
        retrieved_at=dt.datetime(2023, 3, 26, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code="FRB7",
        location_name=None,
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
    acled = normalize_acled_row(_acled_row())
    assert acled is not None
    gdelt_path = tmp_path / "gdelt.jsonl"
    acled_path = tmp_path / "acled.jsonl"
    out_path = tmp_path / "mixed" / "events.jsonl"
    gdelt_path.write_text(gdelt.model_dump_json() + "\n", encoding="utf-8")
    acled_path.write_text(acled.model_dump_json() + "\n", encoding="utf-8")

    audit = merge_event_jsonl(jsonl_paths=[gdelt_path, acled_path], out_path=out_path)

    assert audit["source_counts"] == {"acled": 1, "gdelt_v2_events": 1}
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 2


def _empty_acled_csv_row() -> dict[str, str]:
    return {k: "" for k in ACLED_FIELDS}


def test_expand_acled_csv_country_allow_libya_alias() -> None:
    a = _expand_acled_csv_country_allow(["Libyan Arab Jamahiriya", "Egypt"])
    assert "Libya" in a and "Libyan Arab Jamahiriya" in a
    b = _expand_acled_csv_country_allow(["Libya"])
    assert "Libyan Arab Jamahiriya" in b


def test_ingest_acled_csv_utf8_bom_header(tmp_path: Path) -> None:
    """BOM on first column name must not break ``event_id_cnty`` (utf-8-sig)."""
    csv_path = tmp_path / "bom.csv"
    one = _empty_acled_csv_row()
    one.update(
        {
            "event_id_cnty": "EGYBOM",
            "event_date": "2011-01-10",
            "year": "2011",
            "time_precision": "1",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "inter1": "1",
            "iso": "818",
            "region": "NA",
            "country": "Egypt",
            "admin1": "Cairo",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    import csv as csv_mod

    body = io.StringIO()
    w = csv_mod.DictWriter(body, fieldnames=list(ACLED_FIELDS), lineterminator="\n")
    w.writeheader()
    w.writerow(one)
    raw = body.getvalue().encode("utf-8-sig")
    csv_path.write_bytes(raw)
    out = tmp_path / "out"
    meta = ingest_acled_csv(
        csv_path,
        out_dir=out,
        countries=["Egypt"],
        normalize_to_warehouse=False,
    )
    assert meta["accepted_row_count"] == 1
    line = (out / "fragments" / "page_000001.jsonl").read_text(encoding="utf-8").strip()
    assert "EGYBOM" in line


def test_normalize_acled_csv_row_mapping() -> None:
    row = _empty_acled_csv_row()
    row.update(
        {
            "event_id_cnty": "EGY9",
            "event_date": "2013-06-15",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "iso": "818",
            "country": "Egypt",
            "admin1": "Cairo",
            "location": "Tahrir",
            "actor1": "Protesters",
        }
    )
    retrieved = dt.datetime(2013, 6, 20, 12, 0, 0, tzinfo=dt.timezone.utc)
    rec = normalize_acled_csv_row(
        row,
        retrieved_at=retrieved,
        input_basename="example.csv",
        csv_row_index=2,
    )
    assert rec.source_name == "acled_v3"
    assert rec.source_event_id == "acled:EGY9"
    assert rec.event_date == dt.date(2013, 6, 15)
    assert rec.source_available_at == acled_event_date_end_utc(dt.date(2013, 6, 15))
    assert rec.retrieved_at == retrieved
    assert rec.country_code == "EG"
    assert rec.admin1_code == "Cairo"
    assert rec.event_root_code == "Protests"
    assert rec.event_code == "Peaceful protest"
    assert rec.event_base_code == ""
    assert rec.event_class == "protest"
    assert rec.raw["_csv_input_file"] == "example.csv"
    assert rec.raw["_csv_row"] == 2


def test_ingest_acled_csv_stringio(tmp_path: Path) -> None:
    out_dir = tmp_path / "arab_spring"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(ACLED_FIELDS), lineterminator="\n")

    def add_row(r: dict[str, str]) -> None:
        base = _empty_acled_csv_row()
        base.update(r)
        w.writerow(base)

    w.writeheader()
    add_row(
        {
            "event_id_cnty": "EGY1",
            "event_date": "2013-06-01",
            "year": "2013",
            "time_precision": "1",
            "disorder_type": "Demonstrations",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "actor1": "A",
            "inter1": "1",
            "iso": "818",
            "region": "Northern Africa",
            "country": "Egypt",
            "admin1": "Cairo",
            "latitude": "30.0",
            "longitude": "31.0",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    add_row(
        {
            "event_id_cnty": "TUN1",
            "event_date": "2013-06-02",
            "year": "2013",
            "time_precision": "1",
            "event_type": "Riots",
            "sub_event_type": "Mob violence",
            "inter1": "1",
            "iso": "788",
            "region": "NA",
            "country": "Tunisia",
            "admin1": "Sousse",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    add_row(
        {
            "event_id_cnty": "LBY1",
            "event_date": "2013-06-03",
            "year": "2013",
            "time_precision": "1",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "inter1": "1",
            "iso": "434",
            "region": "NA",
            "country": "Libyan Arab Jamahiriya",
            "admin1": "Tripoli",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    add_row(
        {
            "event_id_cnty": "FRA1",
            "event_date": "2013-06-04",
            "year": "2013",
            "time_precision": "1",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "inter1": "1",
            "iso": "250",
            "region": "Europe",
            "country": "France",
            "admin1": "Paris",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    add_row(
        {
            "event_id_cnty": "USA1",
            "event_date": "2013-06-05",
            "year": "2013",
            "time_precision": "1",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "inter1": "1",
            "iso": "840",
            "region": "Americas",
            "country": "United States",
            "admin1": "DC",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    buf.seek(0)
    meta = ingest_acled_csv(
        buf,
        out_dir=out_dir,
        countries=["Egypt", "Tunisia", "Libyan Arab Jamahiriya"],
    )
    assert meta["accepted_row_count"] == 3
    assert meta["country_skipped_count"] == 2
    assert meta["completed_page_count"] == 1
    frag = out_dir / "fragments" / "page_000001.jsonl"
    assert frag.exists()
    lines = frag.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["event_id_cnty"] == "EGY1"
    assert first["_csv_input_file"] == "inline"
    assert first["country"] == "Egypt"
    man_lines = (out_dir / "fetch_manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(man_lines) == 1
    m0 = json.loads(man_lines[0])
    assert m0["source"] == "csv"
    assert m0["input_file"] == "inline"
    assert m0["fragment_path"] == "fragments/page_000001.jsonl"
    assert m0["row_count"] == 3
    mdoc = json.loads((out_dir / "fetch_metadata.json").read_text(encoding="utf-8"))
    assert mdoc["source"] == "csv"
    assert mdoc["accepted_row_count"] == 3


def test_ingest_acled_csv_warehouse(tmp_path: Path) -> None:
    out_dir = tmp_path / "raw"
    db_path = tmp_path / "wh" / "events.duckdb"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(ACLED_FIELDS), lineterminator="\n")
    w.writeheader()
    wh_row = _empty_acled_csv_row()
    wh_row.update(
        {
            "event_id_cnty": "EGYWH",
            "event_date": "2013-07-01",
            "year": "2013",
            "time_precision": "1",
            "event_type": "Protests",
            "sub_event_type": "Peaceful protest",
            "inter1": "1",
            "iso": "818",
            "region": "NA",
            "country": "Egypt",
            "admin1": "Cairo",
            "source": "S",
            "source_scale": "N",
            "notes": "n",
            "fatalities": "0",
            "timestamp": "0",
        }
    )
    w.writerow(wh_row)
    buf.seek(0)
    meta = ingest_acled_csv(
        buf,
        out_dir=out_dir,
        countries=["Egypt"],
        normalize_to_warehouse=True,
        warehouse_path=db_path,
    )
    assert meta["normalized_row_count"] == 1
    assert meta["normalize_to_warehouse"] is True
    records = query_records(db_path=db_path, source_names={"acled_v3"})
    assert len(records) == 1
    assert records[0].source_event_id == "acled:EGYWH"
    assert records[0].source_name == "acled_v3"
