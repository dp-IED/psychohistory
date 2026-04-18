from __future__ import annotations

import datetime as dt

import pytest

from baselines.metrics import (
    balanced_accuracy,
    brier_score,
    positive_rate,
    pr_auc,
    wm_holdout_metrics_dict,
)
from baselines.recurrence import ForecastRow


def _row(
    *,
    target: bool,
    prob: float,
    model: str = "m",
) -> ForecastRow:
    return ForecastRow(
        forecast_origin=dt.date(2025, 1, 6),
        admin1_code="FR11",
        model_name=model,
        predicted_count=0.0,
        predicted_occurrence_probability=prob,
        target_count_next_7d=1 if target else 0,
        target_occurs_next_7d=target,
    )


def test_positive_rate_balanced() -> None:
    rows = [
        _row(target=True, prob=0.9),
        _row(target=True, prob=0.8),
        _row(target=False, prob=0.1),
        _row(target=False, prob=0.2),
    ]
    assert positive_rate(rows, "m") == 0.5


def test_positive_rate_all_negative() -> None:
    rows = [_row(target=False, prob=0.1) for _ in range(10)]
    assert positive_rate(rows, "m") == 0.0


def test_balanced_accuracy_perfect() -> None:
    rows = [
        _row(target=True, prob=0.9),
        _row(target=False, prob=0.1),
    ]
    assert balanced_accuracy(rows, "m") == pytest.approx(1.0)


def test_balanced_accuracy_degenerate_all_negative_labels() -> None:
    rows = [_row(target=False, prob=0.9)]
    assert balanced_accuracy(rows, "m") == 0.0


def test_balanced_accuracy_degenerate_all_positive_labels() -> None:
    rows = [_row(target=True, prob=0.1)]
    assert balanced_accuracy(rows, "m") == 0.0


def test_pr_auc_perfect_ranking() -> None:
    rows = [
        _row(target=True, prob=1.0),
        _row(target=True, prob=0.9),
        _row(target=False, prob=0.1),
        _row(target=False, prob=0.0),
    ]
    assert pr_auc(rows, "m") == pytest.approx(1.0)


def test_pr_auc_no_positives() -> None:
    rows = [_row(target=False, prob=0.5)]
    assert pr_auc(rows, "m") == 0.0


def test_wm_holdout_metrics_bundle_matches_brier() -> None:
    rows = [
        _row(target=True, prob=0.9),
        _row(target=False, prob=0.1),
    ]
    hm = wm_holdout_metrics_dict(rows, "m")
    assert hm["brier"] == pytest.approx(brier_score(rows, "m"))


def test_pr_auc_constant_scores_bounded() -> None:
    rows = [_row(target=True, prob=0.5), _row(target=False, prob=0.5)]
    auc = pr_auc(rows, "m")
    assert 0.0 <= auc <= 1.0
    assert positive_rate(rows, "m") == 0.5
