"""Metrics for baseline forecast rows."""

from __future__ import annotations

from collections import defaultdict

from baselines.recurrence import ForecastRow


def _rows_for_model(rows: list[ForecastRow], model_name: str) -> list[ForecastRow]:
    selected = [row for row in rows if row.model_name == model_name]
    if not selected:
        raise ValueError(f"no forecast rows for model: {model_name}")
    return selected


def brier_score(rows: list[ForecastRow], model_name: str) -> float:
    selected = _rows_for_model(rows, model_name)
    return sum(
        (
            row.predicted_occurrence_probability
            - float(row.target_occurs_next_7d)
        )
        ** 2
        for row in selected
    ) / len(selected)


def mean_absolute_error(rows: list[ForecastRow], model_name: str) -> float:
    selected = _rows_for_model(rows, model_name)
    return sum(
        abs(row.predicted_count - row.target_count_next_7d)
        for row in selected
    ) / len(selected)


def top_k_hit_rate(rows: list[ForecastRow], model_name: str, k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    selected = _rows_for_model(rows, model_name)
    by_origin: dict[object, list[ForecastRow]] = defaultdict(list)
    for row in selected:
        by_origin[row.forecast_origin].append(row)

    hits = 0
    for origin_rows in by_origin.values():
        ranked = sorted(
            origin_rows,
            key=lambda row: (-row.predicted_count, row.admin1_code),
        )
        hits += int(any(row.target_occurs_next_7d for row in ranked[:k]))
    return hits / len(by_origin)
