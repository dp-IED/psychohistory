"""Recurrence baselines over point-in-time normalized event records."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from ingest.event_tape import EventTapeRecord
from ingest.snapshot_export import build_snapshot_payload


UTC = dt.timezone.utc
WINDOW_DAYS = 7
RECURRENCE_MODEL_NAMES = (
    "previous_week_count",
    "trailing_4_week_mean",
    "trailing_12_week_mean",
)
RecurrenceModelName = Literal[
    "previous_week_count",
    "trailing_4_week_mean",
    "trailing_12_week_mean",
]


class ForecastRow(BaseModel):
    forecast_origin: dt.date
    admin1_code: str
    model_name: str
    predicted_count: float
    predicted_occurrence_probability: float
    target_count_next_7d: int
    target_occurs_next_7d: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


def _origin_datetime(forecast_origin: dt.date) -> dt.datetime:
    return dt.datetime.combine(forecast_origin, dt.time(), tzinfo=UTC)


def point_in_time_feature_events(
    records: Iterable[EventTapeRecord],
    *,
    forecast_origin: dt.date,
    source_names: set[str] | None = None,
) -> list[EventTapeRecord]:
    """Return events visible strictly before the forecast origin."""

    origin_dt = _origin_datetime(forecast_origin)
    return [
        record
        for record in records
        if (source_names is None or record.source_name in source_names)
        if record.source_available_at.astimezone(UTC) < origin_dt
        and record.event_date < forecast_origin
    ]


def _count_events_by_admin(
    records: Iterable[EventTapeRecord],
    *,
    start: dt.date,
    end: dt.date,
) -> Counter[str]:
    return Counter(
        record.admin1_code
        for record in records
        if start <= record.event_date < end
    )


def _targets_from_snapshot(
    records: list[EventTapeRecord],
    *,
    forecast_origin: dt.date,
    source_names: set[str] | None = None,
) -> dict[str, tuple[int, bool]]:
    payload = build_snapshot_payload(
        records=records,
        origin_date=forecast_origin,
        source_names=source_names,
    )
    counts: dict[str, int] = {}
    occurrences: dict[str, bool] = {}

    for row in payload["target_table"]:
        admin1_code = str(row["metadata"]["admin1_code"])
        if row["name"] == "target_count_next_7d":
            counts[admin1_code] = int(row["value"])
        elif row["name"] == "target_occurs_next_7d":
            occurrences[admin1_code] = bool(row["value"])

    missing = sorted(set(counts) ^ set(occurrences))
    if missing:
        raise ValueError(f"incomplete target rows for admin1 codes: {missing}")
    return {
        admin1_code: (counts[admin1_code], occurrences[admin1_code])
        for admin1_code in sorted(counts)
    }


def _predicted_counts_by_model(
    feature_events: list[EventTapeRecord],
    *,
    forecast_origin: dt.date,
) -> dict[str, Counter[str]]:
    return {
        "previous_week_count": _count_events_by_admin(
            feature_events,
            start=forecast_origin - dt.timedelta(days=WINDOW_DAYS),
            end=forecast_origin,
        ),
        "trailing_4_week_mean": Counter(
            {
                admin1_code: count / 4.0
                for admin1_code, count in _count_events_by_admin(
                    feature_events,
                    start=forecast_origin - dt.timedelta(days=WINDOW_DAYS * 4),
                    end=forecast_origin,
                ).items()
            }
        ),
        "trailing_12_week_mean": Counter(
            {
                admin1_code: count / 12.0
                for admin1_code, count in _count_events_by_admin(
                    feature_events,
                    start=forecast_origin - dt.timedelta(days=WINDOW_DAYS * 12),
                    end=forecast_origin,
                ).items()
            }
        ),
    }


def build_recurrence_forecasts_for_origin(
    *,
    records: list[EventTapeRecord],
    forecast_origin: dt.date,
    model_names: Iterable[str] = RECURRENCE_MODEL_NAMES,
    source_names: set[str] | None = None,
) -> list[ForecastRow]:
    feature_events = point_in_time_feature_events(
        records,
        forecast_origin=forecast_origin,
        source_names=source_names,
    )
    targets = _targets_from_snapshot(
        records,
        forecast_origin=forecast_origin,
        source_names=source_names,
    )
    predictions_by_model = _predicted_counts_by_model(
        feature_events,
        forecast_origin=forecast_origin,
    )

    rows: list[ForecastRow] = []
    for model_name in model_names:
        if model_name not in predictions_by_model:
            raise ValueError(f"unknown recurrence model: {model_name}")
        predictions = predictions_by_model[model_name]
        for admin1_code, (target_count, target_occurs) in targets.items():
            predicted_count = float(predictions.get(admin1_code, 0.0))
            rows.append(
                ForecastRow(
                    forecast_origin=forecast_origin,
                    admin1_code=admin1_code,
                    model_name=model_name,
                    predicted_count=predicted_count,
                    predicted_occurrence_probability=min(1.0, predicted_count),
                    target_count_next_7d=target_count,
                    target_occurs_next_7d=target_occurs,
                    metadata={
                        "feature_record_count": len(feature_events),
                        "feature_rule": (
                            "source_available_at < forecast_origin and "
                            "event_date < forecast_origin_date"
                        ),
                        "target_source": "ingest.snapshot_export.build_snapshot_payload",
                    },
                )
            )
    return rows
