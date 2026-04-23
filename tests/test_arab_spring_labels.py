from __future__ import annotations

import json
import tempfile
import datetime as dt
from datetime import date
from pathlib import Path

import pytest

from baselines.arab_spring_labels import build_arab_spring_labels
from ingest.event_warehouse import init_warehouse, upsert_records
from ingest.event_tape import EventTapeRecord

UTC = dt.timezone.utc


def _acled_eg(
    eid: str,
    ed: date,
    fatalities: int,
) -> EventTapeRecord:
    stamp = dt.datetime(2011, 1, 1, 12, 0, tzinfo=UTC)
    row = {
        "event_id_cnty": eid,
        "event_date": ed.isoformat(),
        "event_type": "Protests",
        "sub_event_type": "x",
        "actor1": "p",
        "actor2": "",
        "admin1": "Cairo",
        "location": "Cairo",
        "iso": "818",
        "fatalities": str(fatalities),
    }
    return EventTapeRecord(
        source_name="acled_v3",
        source_event_id=f"acled:{eid}",
        event_date=ed,
        source_available_at=stamp,
        retrieved_at=stamp,
        country_code="EG",
        admin1_code="EGC1",
        location_name="Cairo",
        latitude=30.0,
        longitude=31.0,
        event_class="protest",
        event_code="x",
        event_base_code="x",
        event_root_code="Protests",
        quad_class=None,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name="P",
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw=row,
    )


def _write_warehouse(path: Path, recs: list[EventTapeRecord]) -> None:
    init_warehouse(path)
    upsert_records(db_path=path, records=recs)


def test_label_fat4_zero_fat5_one(tmp_path: Path) -> None:
    d0 = date(2011, 1, 1)
    p = tmp_path / "w.duckdb"
    _write_warehouse(p, [_acled_eg("A", date(2011, 1, 2), 4)])
    m = build_arab_spring_labels(
        p,
        as_of=date(2011, 12, 31),
        country="EG",
        data_start=date(2011, 1, 1),
        data_end=date(2011, 12, 31),
    )
    assert m[d0] == 0
    _write_warehouse(p, [_acled_eg("B", date(2011, 1, 2), 5)])
    m2 = build_arab_spring_labels(
        p,
        as_of=date(2011, 12, 31),
        country="EG",
        data_start=date(2011, 1, 1),
        data_end=date(2011, 12, 31),
    )
    assert m2[d0] == 1


def test_forward_excludes_event_on_query_day(tmp_path: Path) -> None:
    d0 = date(2011, 1, 1)
    _write_warehouse(tmp_path / "w.duckdb", [_acled_eg("S", d0, 10)])
    m = build_arab_spring_labels(
        tmp_path / "w.duckdb",
        as_of=date(2011, 12, 31),
        country="EG",
        data_start=date(2011, 1, 1),
        data_end=date(2011, 12, 31),
    )
    assert m.get(d0, 0) == 0


def test_event_beyond_seven_day_window_not_in_label(tmp_path: Path) -> None:
    d0 = date(2011, 1, 1)
    _write_warehouse(
        tmp_path / "w.duckdb",
        [_acled_eg("E", date(2011, 1, 9), 5)],
    )
    m = build_arab_spring_labels(
        tmp_path / "w.duckdb",
        as_of=date(2011, 12, 31),
        country="EG",
        data_start=date(2011, 1, 1),
        data_end=date(2011, 12, 31),
    )
    assert m[d0] == 0


def test_syria_excluded_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "w.duckdb"
    _write_warehouse(p, [_acled_eg("S", date(2011, 1, 3), 10)])
    assert build_arab_spring_labels(
        p,
        as_of=date(2011, 6, 1),
        country="SY",
    ) == {}


def test_raw_json_roundtrip_fatality(tmp_path: Path) -> None:
    p = tmp_path / "w.duckdb"
    _write_warehouse(p, [_acled_eg("X", date(2011, 1, 2), 5)])
    import duckdb

    c = duckdb.connect(str(p), read_only=True)
    rj = c.execute("select raw_json from events limit 1").fetchone()[0]
    c.close()
    d = json.loads(rj)
    assert int(str(d.get("fatalities", 0))) == 5
