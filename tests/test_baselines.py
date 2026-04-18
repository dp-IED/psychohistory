from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from baselines.backtest import run_recurrence_backtest
from baselines.metrics import brier_score, mean_absolute_error, recall_at_k, top_k_hit_rate
from baselines.recurrence import ForecastRow, build_recurrence_forecasts_for_origin
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import init_warehouse, upsert_records
from ingest.snapshot_export import build_snapshot_payload


def _record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=dt.date.fromisoformat(event_date),
        source_available_at=dt.datetime.fromisoformat(source_available_at.replace("Z", "+00:00")),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
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


def _snapshot_targets(records: list[EventTapeRecord], origin: dt.date) -> dict[str, tuple[int, bool]]:
    payload = build_snapshot_payload(records=records, origin_date=origin)
    counts: dict[str, int] = {}
    occurs: dict[str, bool] = {}
    for row in payload["target_table"]:
        admin1_code = row["metadata"]["admin1_code"]
        if row["name"] == "target_count_next_7d":
            counts[admin1_code] = row["value"]
        elif row["name"] == "target_occurs_next_7d":
            occurs[admin1_code] = row["value"]
    return {admin1_code: (counts[admin1_code], occurs[admin1_code]) for admin1_code in counts}


def test_recurrence_uses_point_in_time_features_and_snapshot_targets() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:feature", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
        _record("gdelt:at-origin", event_date="2021-01-05", source_available_at="2021-01-11T00:00:00Z", admin1_code="FR22"),
        _record("gdelt:target", event_date="2021-01-12", source_available_at="2021-01-13T00:00:00Z"),
    ]

    rows = build_recurrence_forecasts_for_origin(records=records, forecast_origin=origin)
    previous_week = {
        row.admin1_code: row
        for row in rows
        if row.model_name == "previous_week_count"
    }

    assert sorted(previous_week) == ["FR11", "FR22"]
    assert previous_week["FR11"].predicted_count == 1.0
    assert previous_week["FR11"].predicted_occurrence_probability == 1.0
    assert previous_week["FR11"].target_count_next_7d == 1
    assert previous_week["FR11"].target_occurs_next_7d is True
    assert previous_week["FR22"].predicted_count == 0.0
    assert previous_week["FR11"].metadata["feature_record_count"] == 1
    assert {row.admin1_code for row in rows} == set(_snapshot_targets(records, origin))


def test_recurrence_excludes_origin_day_from_features_but_uses_it_as_target() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:prior", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
        _record("gdelt:origin-day", event_date="2021-01-11", source_available_at="2021-01-10T00:00:00Z"),
    ]

    rows = build_recurrence_forecasts_for_origin(records=records, forecast_origin=origin)
    previous_week_fr11 = next(
        row
        for row in rows
        if row.model_name == "previous_week_count" and row.admin1_code == "FR11"
    )

    assert previous_week_fr11.predicted_count == 1.0
    assert previous_week_fr11.target_count_next_7d == 1
    assert previous_week_fr11.metadata["feature_record_count"] == 1


def test_metrics_for_forecast_rows() -> None:
    rows = [
        ForecastRow(
            forecast_origin=dt.date(2021, 1, 4),
            admin1_code="FR11",
            model_name="m",
            predicted_count=0.25,
            predicted_occurrence_probability=0.25,
            target_count_next_7d=1,
            target_occurs_next_7d=True,
        ),
        ForecastRow(
            forecast_origin=dt.date(2021, 1, 4),
            admin1_code="FR22",
            model_name="m",
            predicted_count=0.75,
            predicted_occurrence_probability=0.75,
            target_count_next_7d=0,
            target_occurs_next_7d=False,
        ),
    ]

    assert brier_score(rows, "m") == pytest.approx((0.75**2 + 0.75**2) / 2)
    assert mean_absolute_error(rows, "m") == pytest.approx((0.75 + 0.75) / 2)
    assert top_k_hit_rate(rows, "m", k=1) == 0.0
    assert top_k_hit_rate(rows, "m", k=2) == 1.0


def test_recall_at_k_captures_fraction_of_positives() -> None:
    rows = [
        ForecastRow(
            forecast_origin=dt.date(2021, 1, 4),
            admin1_code=f"FR{i:02d}",
            model_name="m",
            predicted_count=float(10 - i),
            predicted_occurrence_probability=float(10 - i) / 10,
            target_count_next_7d=1 if i < 4 else 0,
            target_occurs_next_7d=i < 4,
        )
        for i in range(10)
    ]

    assert recall_at_k(rows, "m", k=5) == pytest.approx(1.0)
    assert recall_at_k(rows, "m", k=2) == pytest.approx(0.5)
    assert recall_at_k(rows, "m", k=0) == pytest.approx(0.0)


def test_recurrence_backtest_writes_jsonl_and_audit(tmp_path: Path) -> None:
    out_path = tmp_path / "baselines" / "recurrence_predictions.jsonl"
    db = tmp_path / "warehouse" / "events.duckdb"
    init_warehouse(db)
    upsert_records(
        db_path=db,
        records=[
            _record("gdelt:feature", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
            _record("gdelt:target", event_date="2021-01-12", source_available_at="2021-01-13T00:00:00Z"),
        ],
    )

    audit = run_recurrence_backtest(
        warehouse_path=db,
        origin_start=dt.date(2021, 1, 11),
        origin_end=dt.date(2021, 1, 11),
        out_path=out_path,
    )

    written_rows = [
        ForecastRow.model_validate_json(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    written_audit = json.loads(out_path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    assert len(written_rows) == 3
    assert audit["row_count"] == 3
    assert written_audit["row_count"] == 3
    assert written_audit["admin1_count"] == 1
    assert written_audit["model_names"] == [
        "previous_week_count",
        "trailing_4_week_mean",
        "trailing_12_week_mean",
    ]
