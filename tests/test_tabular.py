from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.backtest import run_tabular_backtest
from baselines.features import FEATURE_NAMES, FeatureRow, build_feature_matrix, extract_features_for_origin
from baselines.tabular import (
    TabularForecastRow,
    predict_tabular,
    train_tabular_model,
)
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import init_warehouse, upsert_records


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
        source_available_at=dt.datetime.fromisoformat(
            source_available_at.replace("Z", "+00:00")
        ),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
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


def _make_feature_rows(n: int, n_pos: int, universe: list[str]) -> list[FeatureRow]:
    rows = []
    for i in range(n):
        code = universe[i % len(universe)]
        rows.append(
            FeatureRow(
                forecast_origin=dt.date(2021, 1, 4) + dt.timedelta(weeks=i),
                admin1_code=code,
                features={name: float(i % 5) for name in FEATURE_NAMES},
            )
        )
    return rows


def test_train_tabular_model_returns_callable() -> None:
    universe = ["FR11", "FR22"]
    n = 40
    feature_rows = _make_feature_rows(n, n_pos=20, universe=universe)
    targets = {(r.forecast_origin, r.admin1_code): (i % 2 == 0) for i, r in enumerate(feature_rows)}

    model = train_tabular_model(feature_rows=feature_rows, targets=targets)

    assert callable(model)


def test_train_tabular_model_handles_single_class_smoke_windows() -> None:
    universe = ["FR11", "FR22"]
    feature_rows = _make_feature_rows(4, n_pos=4, universe=universe)
    targets = {(row.forecast_origin, row.admin1_code): True for row in feature_rows}

    model = train_tabular_model(feature_rows=feature_rows, targets=targets)
    proba = model(build_feature_matrix(feature_rows)[0])

    assert proba.shape == (4, 2)
    assert np.allclose(proba[:, 1], 1.0)


def test_predict_tabular_returns_one_row_per_admin1() -> None:
    universe = ["FR11", "FR22", "FR33"]
    n = 60
    feature_rows = _make_feature_rows(n, n_pos=30, universe=universe)
    targets = {(r.forecast_origin, r.admin1_code): (r.admin1_code in {"FR11", "FR33"}) for r in feature_rows}
    model = train_tabular_model(feature_rows=feature_rows, targets=targets)

    origin = dt.date(2021, 3, 1)
    eval_rows = [
        FeatureRow(
            forecast_origin=origin,
            admin1_code=code,
            features={name: 1.0 for name in FEATURE_NAMES},
        )
        for code in universe
    ]
    predictions = predict_tabular(model=model, feature_rows=eval_rows)

    assert len(predictions) == 3
    for pred in predictions:
        assert isinstance(pred, TabularForecastRow)
        assert 0.0 <= pred.predicted_occurrence_probability <= 1.0
        assert pred.model_name == "xgboost_tabular"


def test_tabular_forecast_row_serializes_to_json() -> None:
    row = TabularForecastRow(
        forecast_origin=dt.date(2021, 1, 4),
        admin1_code="FR11",
        model_name="xgboost_tabular",
        predicted_count=0.5,
        predicted_occurrence_probability=0.5,
        target_count_next_7d=1,
        target_occurs_next_7d=True,
    )
    serialized = row.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["model_name"] == "xgboost_tabular"
    assert parsed["admin1_code"] == "FR11"


def test_tabular_backtest_writes_output(tmp_path: Path) -> None:
    out_path = tmp_path / "tabular_predictions.jsonl"

    records = []
    for week in range(20):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        # Alternate regions so training labels include both positive and negative
        # rows (XGBoost requires both classes).
        admin1_code = "FR11" if week % 2 == 0 else "FR22"
        records.append(
            EventTapeRecord(
                source_name="gdelt_v2_events",
                source_event_id=f"gdelt:{week}",
                event_date=origin - dt.timedelta(days=3),
                source_available_at=dt.datetime.combine(
                    origin - dt.timedelta(days=2), dt.time(), tzinfo=dt.timezone.utc
                ),
                retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
                country_code="FR",
                admin1_code=admin1_code,
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
        )

    db = tmp_path / "warehouse" / "events.duckdb"
    init_warehouse(db)
    upsert_records(db_path=db, records=records)

    audit = run_tabular_backtest(
        warehouse_path=db,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 4, 26),
        eval_origin_start=dt.date(2021, 5, 3),
        eval_origin_end=dt.date(2021, 5, 17),
        out_path=out_path,
    )

    assert out_path.exists()
    assert audit["eval_row_count"] > 0
    assert "brier" in audit
    assert "recall_at_5" in audit
