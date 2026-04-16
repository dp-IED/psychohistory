from __future__ import annotations

import datetime as dt

from baselines.features import (
    FEATURE_NAMES,
    build_feature_matrix,
    extract_features_for_origin,
)
from ingest.event_tape import EventTapeRecord


def _record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
    source_name: str = "gdelt_v2_events",
    actor1_name: str | None = None,
    avg_tone: float | None = None,
    goldstein_scale: float | None = None,
    num_mentions: int | None = None,
    num_articles: int | None = None,
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name=source_name,
        source_event_id=source_event_id,
        event_date=dt.date.fromisoformat(event_date),
        source_available_at=dt.datetime.fromisoformat(
            source_available_at.replace("Z", "+00:00")
        ),
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
        goldstein_scale=goldstein_scale,
        num_mentions=num_mentions,
        num_sources=None,
        num_articles=num_articles,
        avg_tone=avg_tone,
        actor1_name=actor1_name,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def test_feature_names_are_stable() -> None:
    assert "event_count_prev_1w" in FEATURE_NAMES
    assert "rate_change_1w_vs_4w" in FEATURE_NAMES
    assert "mean_goldstein_prev_4w" in FEATURE_NAMES
    assert "admin1_code_idx" in FEATURE_NAMES


def test_extract_features_counts_visible_events_only() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:vis", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
        _record("gdelt:future", event_date="2021-01-05", source_available_at="2021-01-12T00:00:00Z"),
        _record("gdelt:old", event_date="2020-11-01", source_available_at="2020-11-02T00:00:00Z"),
    ]
    scoring_universe = ["FR11", "FR22"]

    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=scoring_universe)

    fr11 = next(r for r in rows if r.admin1_code == "FR11")
    assert fr11.features["event_count_prev_1w"] == 1.0
    assert fr11.features["event_count_prev_4w"] == 1.0
    assert fr11.features["event_count_prev_12w"] == 2.0

    fr22 = next(r for r in rows if r.admin1_code == "FR22")
    assert fr22.features["event_count_prev_1w"] == 0.0


def test_rate_change_zero_when_no_4w_baseline() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:r1", event_date="2021-01-05", source_available_at="2021-01-12T00:00:00Z"),
    ]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    fr11 = rows[0]
    assert fr11.features["rate_change_1w_vs_4w"] == 0.0


def test_mean_tone_zero_when_no_events() -> None:
    origin = dt.date(2021, 1, 11)
    records = []
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    assert rows[0].features["mean_avg_tone_prev_4w"] == 0.0


def test_weeks_since_last_event_is_capped_at_52() -> None:
    origin = dt.date(2022, 1, 10)
    records = [
        _record("gdelt:old", event_date="2019-01-01", source_available_at="2019-01-02T00:00:00Z"),
    ]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    assert rows[0].features["weeks_since_last_event"] == 52.0


def test_admin1_code_idx_matches_sorted_position() -> None:
    origin = dt.date(2021, 1, 11)
    records = []
    universe = ["FRA1", "FRB2", "FRC3"]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=universe)
    idxs = {r.admin1_code: r.features["admin1_code_idx"] for r in rows}
    assert idxs == {"FRA1": 0.0, "FRB2": 1.0, "FRC3": 2.0}


def test_build_feature_matrix_shape_and_order() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:vis", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
    ]
    universe = ["FR11", "FR22"]
    feature_rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=universe)
    X, admin1_codes = build_feature_matrix(feature_rows)

    assert X.shape == (2, len(FEATURE_NAMES))
    assert list(admin1_codes) == ["FR11", "FR22"]


def test_extract_features_filters_to_gdelt_source() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record(
            "gdelt:vis",
            event_date="2021-01-05",
            source_available_at="2021-01-06T00:00:00Z",
        ),
        _record(
            "acled:vis",
            event_date="2021-01-06",
            source_available_at="2021-01-07T00:00:00Z",
            source_name="acled",
        ),
    ]

    rows = extract_features_for_origin(
        records=records,
        origin_date=origin,
        scoring_universe=["FR11"],
        source_names={"gdelt_v2_events"},
    )

    assert rows[0].features["event_count_prev_1w"] == 1.0


def test_extract_features_filters_to_acled_source() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record(
            "gdelt:vis",
            event_date="2021-01-05",
            source_available_at="2021-01-06T00:00:00Z",
        ),
        _record(
            "acled:vis",
            event_date="2021-01-06",
            source_available_at="2021-01-07T00:00:00Z",
            source_name="acled",
        ),
    ]

    rows = extract_features_for_origin(
        records=records,
        origin_date=origin,
        scoring_universe=["FR11"],
        source_names={"acled"},
    )

    assert rows[0].features["event_count_prev_1w"] == 1.0


def test_extract_features_uses_all_sources_by_default() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record(
            "gdelt:vis",
            event_date="2021-01-05",
            source_available_at="2021-01-06T00:00:00Z",
        ),
        _record(
            "acled:vis",
            event_date="2021-01-06",
            source_available_at="2021-01-07T00:00:00Z",
            source_name="acled",
        ),
    ]

    rows = extract_features_for_origin(
        records=records,
        origin_date=origin,
        scoring_universe=["FR11"],
    )

    assert rows[0].features["event_count_prev_1w"] == 2.0
