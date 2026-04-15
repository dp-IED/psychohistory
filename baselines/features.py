"""Per-origin per-admin1 feature extraction from the event tape."""

from __future__ import annotations

import datetime as dt

import numpy as np

from ingest.event_tape import EventTapeRecord

UTC = dt.timezone.utc
WINDOW_DAYS = 7
NATIONAL_CODES = frozenset({"FR", "FR00"})

FEATURE_NAMES: list[str] = [
    "event_count_prev_1w",
    "event_count_prev_4w",
    "event_count_prev_12w",
    "rate_change_1w_vs_4w",
    "mean_goldstein_prev_4w",
    "mean_avg_tone_prev_4w",
    "mean_num_mentions_prev_4w",
    "mean_num_articles_prev_4w",
    "distinct_actor_count_prev_4w",
    "national_event_count_prev_1w",
    "national_event_count_prev_4w",
    "weeks_since_last_event",
    "admin1_code_idx",
]


class FeatureRow:
    __slots__ = ("forecast_origin", "admin1_code", "features")

    def __init__(
        self,
        forecast_origin: dt.date,
        admin1_code: str,
        features: dict[str, float],
    ) -> None:
        self.forecast_origin = forecast_origin
        self.admin1_code = admin1_code
        self.features = features


def _visible_before(records: list[EventTapeRecord], origin_dt: dt.datetime) -> list[EventTapeRecord]:
    return [
        r
        for r in records
        if r.source_available_at.astimezone(UTC) < origin_dt and r.event_date < origin_dt.date()
    ]


def extract_features_for_origin(
    *,
    records: list[EventTapeRecord],
    origin_date: dt.date,
    scoring_universe: list[str],
) -> list[FeatureRow]:
    origin_dt = dt.datetime.combine(origin_date, dt.time(), tzinfo=UTC)
    visible = _visible_before(records, origin_dt)

    w1_start = origin_date - dt.timedelta(days=WINDOW_DAYS)
    w4_start = origin_date - dt.timedelta(days=WINDOW_DAYS * 4)
    w12_start = origin_date - dt.timedelta(days=WINDOW_DAYS * 12)

    national_1w = [r for r in visible if r.admin1_code in NATIONAL_CODES and r.event_date >= w1_start]
    national_4w = [r for r in visible if r.admin1_code in NATIONAL_CODES and r.event_date >= w4_start]
    nat_1w_count = float(len(national_1w))
    nat_4w_count = float(len(national_4w))

    code_to_idx = {code: float(i) for i, code in enumerate(scoring_universe)}

    rows: list[FeatureRow] = []
    for admin1_code in scoring_universe:
        prev_1w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w1_start]
        prev_4w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w4_start]
        prev_12w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w12_start]
        all_prev = [r for r in visible if r.admin1_code == admin1_code]

        c1w = float(len(prev_1w))
        c4w = float(len(prev_4w))
        c12w = float(len(prev_12w))

        mean_4w = c4w / 4.0
        rate_change = (c1w / mean_4w) if mean_4w > 0 else 0.0

        goldsteins = [r.goldstein_scale for r in prev_4w if r.goldstein_scale is not None]
        tones = [r.avg_tone for r in prev_4w if r.avg_tone is not None]
        mentions = [r.num_mentions for r in prev_4w if r.num_mentions is not None]
        articles = [r.num_articles for r in prev_4w if r.num_articles is not None]
        actors = {r.actor1_name for r in prev_4w if r.actor1_name}

        if all_prev:
            most_recent = max(r.event_date for r in all_prev)
            weeks_gap = min(52.0, (origin_date - most_recent).days / 7.0)
        else:
            weeks_gap = 52.0

        rows.append(
            FeatureRow(
                forecast_origin=origin_date,
                admin1_code=admin1_code,
                features={
                    "event_count_prev_1w": c1w,
                    "event_count_prev_4w": c4w,
                    "event_count_prev_12w": c12w,
                    "rate_change_1w_vs_4w": rate_change,
                    "mean_goldstein_prev_4w": float(np.mean(goldsteins)) if goldsteins else 0.0,
                    "mean_avg_tone_prev_4w": float(np.mean(tones)) if tones else 0.0,
                    "mean_num_mentions_prev_4w": float(np.mean(mentions)) if mentions else 0.0,
                    "mean_num_articles_prev_4w": float(np.mean(articles)) if articles else 0.0,
                    "distinct_actor_count_prev_4w": float(len(actors)),
                    "national_event_count_prev_1w": nat_1w_count,
                    "national_event_count_prev_4w": nat_4w_count,
                    "weeks_since_last_event": weeks_gap,
                    "admin1_code_idx": code_to_idx[admin1_code],
                },
            )
        )
    return rows


def build_feature_matrix(
    feature_rows: list[FeatureRow],
) -> tuple[np.ndarray, list[str]]:
    X = np.array(
        [[row.features[name] for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float32,
    )
    admin1_codes = [row.admin1_code for row in feature_rows]
    return X, admin1_codes
