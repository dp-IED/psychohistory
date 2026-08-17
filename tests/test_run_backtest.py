from __future__ import annotations

from datetime import date

import pytest

from harness.corpus.backtest_corpus import BacktestQuestion
from scripts.run_backtest import (
    BacktestResult,
    _is_duplicate,
    _normalise,
    _rollup_summary,
)


def test_normalise_strips_punctuation() -> None:
    assert _normalise("  Will X happen? ") == "will x happen"


def test_is_duplicate_matches_id_or_text() -> None:
    assert _is_duplicate("Will X happen?", set(), {"will x happen"}, "")
    assert _is_duplicate("other", {"q-1"}, set(), "q-1")
    assert not _is_duplicate("Will Y happen?", set(), {"will x happen"}, "q-2")


def test_rollup_summary_mean_brier() -> None:
    results = [
        BacktestResult(question_id="a", p_yes=0.7, brier_score=0.09),
        BacktestResult(question_id="b", p_yes=0.2, brier_score=None),
        BacktestResult(question_id="c", p_yes=0.4, brier_score=0.01),
    ]
    summary = _rollup_summary(results)
    assert summary["total"] == 3
    assert summary["resolved"] == 2
    assert summary["mean_brier"] == pytest.approx(0.05)


def test_sample_question_shape() -> None:
    q = BacktestQuestion(
        question_id="q-1",
        source="polymarket",
        question_text="Unit test",
        open_date=date(2024, 1, 1),
        close_date=date(2024, 6, 1),
        resolution=True,
        market_price_at_open=0.5,
        category=None,
    )
    assert q.question_id == "q-1"
