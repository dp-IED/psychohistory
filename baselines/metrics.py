"""Metrics for baseline forecast rows."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from baselines.recurrence import ForecastRow


@runtime_checkable
class _BinaryForecastLike(Protocol):
    forecast_origin: Any
    admin1_code: str
    model_name: str
    predicted_occurrence_probability: float
    target_occurs_next_7d: bool


def _rows_for_model(
    rows: list[_BinaryForecastLike], model_name: str
) -> list[_BinaryForecastLike]:
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


def binary_log_loss(
    rows: list[_BinaryForecastLike],
    model_name: str,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean binary cross-entropy on predicted probabilities vs ``target_occurs_next_7d``."""

    selected = _rows_for_model(rows, model_name)
    total = 0.0
    for row in selected:
        p = float(row.predicted_occurrence_probability)
        p = min(max(p, eps), 1.0 - eps)
        y = 1.0 if row.target_occurs_next_7d else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(selected)


def brier_skill_score(
    rows: list[_BinaryForecastLike],
    model_name: str,
    *,
    reference_prevalence: float,
) -> float:
    """BSS = 1 - Brier / (p(1-p)) using a fixed label prevalence reference (e.g. holdout)."""

    p = float(reference_prevalence)
    denom = p * (1.0 - p)
    if denom <= 0.0:
        return float("nan")
    return 1.0 - brier_score(rows, model_name) / denom  # type: ignore[arg-type]


def holdout_mask_identity(rows: list[_BinaryForecastLike]) -> dict[str, Any]:
    """Stable fingerprint of scored holdout (origin, admin1) keys for audit parity across variants."""

    keys = sorted(
        (row.forecast_origin.isoformat(), row.admin1_code) for row in rows
    )
    payload = "\n".join(f"{o}\t{c}" for o, c in keys)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "scored_row_count": len(rows),
        "keys_sha256": digest,
    }


def wm_holdout_metrics_dict(
    rows: list[_BinaryForecastLike],
    model_name: str,
    *,
    balanced_accuracy_threshold: float = 0.5,
) -> dict[str, float]:
    """Required WM ablation contract: one bundle from a single holdout forward pass."""

    if not rows:
        return {
            "brier": float("nan"),
            "log_loss": float("nan"),
            "pr_auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "label_prevalence": float("nan"),
            "mean_prediction": float("nan"),
            "brier_skill_score": float("nan"),
        }
    sel = _rows_for_model(rows, model_name)
    n = len(sel)
    prev = sum(1.0 for row in sel if row.target_occurs_next_7d) / float(n)
    mean_p = sum(float(row.predicted_occurrence_probability) for row in sel) / float(n)
    br = (
        sum(
            (float(row.predicted_occurrence_probability) - float(row.target_occurs_next_7d)) ** 2
            for row in sel
        )
        / float(n)
    )
    ll = binary_log_loss(rows, model_name)
    pr = pr_auc(rows, model_name)  # type: ignore[arg-type]
    bacc = balanced_accuracy(rows, model_name, threshold=balanced_accuracy_threshold)  # type: ignore[arg-type]
    denom = prev * (1.0 - prev)
    bss = float("nan") if denom <= 0.0 else 1.0 - br / denom
    return {
        "brier": br,
        "log_loss": ll,
        "pr_auc": pr,
        "balanced_accuracy": bacc,
        "label_prevalence": prev,
        "mean_prediction": mean_p,
        "brier_skill_score": bss,
    }
