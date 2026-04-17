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


def recall_at_k(rows: list[ForecastRow], model_name: str, k: int = 5) -> float:
    if k <= 0:
        return 0.0
    selected = _rows_for_model(rows, model_name)
    by_origin: dict[object, list[ForecastRow]] = defaultdict(list)
    for row in selected:
        by_origin[row.forecast_origin].append(row)

    total_positives = 0
    captured = 0
    for origin_rows in by_origin.values():
        ranked = sorted(
            origin_rows,
            key=lambda row: (-row.predicted_count, row.admin1_code),
        )
        top_k = ranked[:k]
        captured += sum(row.target_occurs_next_7d for row in top_k)
        total_positives += sum(row.target_occurs_next_7d for row in origin_rows)
    if total_positives == 0:
        return 0.0
    return captured / total_positives


def positive_rate(rows: list[ForecastRow], model_name: str) -> float:
    selected = _rows_for_model(rows, model_name)
    positives = sum(1 for row in selected if row.target_occurs_next_7d)
    return positives / len(selected)


def balanced_accuracy(
    rows: list[ForecastRow],
    model_name: str,
    *,
    threshold: float = 0.5,
) -> float:
    """Mean of sensitivity (TPR) and specificity (TNR) at a probability threshold.

    Returns 0.0 when every label is positive or every label is negative (degenerate).
    """

    selected = _rows_for_model(rows, model_name)
    pos = [row for row in selected if row.target_occurs_next_7d]
    neg = [row for row in selected if not row.target_occurs_next_7d]
    if not pos or not neg:
        return 0.0

    tp = sum(
        1
        for row in pos
        if row.predicted_occurrence_probability >= threshold
    )
    fn = len(pos) - tp
    tn = sum(
        1
        for row in neg
        if row.predicted_occurrence_probability < threshold
    )
    fp = len(neg) - tn
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (sensitivity + specificity) / 2.0


def pr_auc(rows: list[ForecastRow], model_name: str) -> float:
    """Average precision (area under the precision–recall curve) for binary labels."""

    selected = _rows_for_model(rows, model_name)
    n_pos = sum(1 for row in selected if row.target_occurs_next_7d)
    if n_pos == 0:
        return 0.0
    sorted_rows = sorted(
        selected,
        key=lambda row: -row.predicted_occurrence_probability,
    )
    tp = 0
    ap = 0.0
    for i, row in enumerate(sorted_rows, start=1):
        if row.target_occurs_next_7d:
            tp += 1
            precision_at_i = tp / i
            ap += precision_at_i
    return ap / n_pos
