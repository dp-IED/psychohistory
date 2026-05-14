from __future__ import annotations

import json
from datetime import date

import pytest

from harness.corpus.backtest_corpus import (
    BacktestQuestion,
    normalize_manifold_market,
    normalize_polymarket_market,
)


def test_polymarket_field_mapping() -> None:
    raw = {
        "conditionId": "0xabc",
        "question": "Will tests pass?",
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-02-01T00:00:00Z",
        "volumeNum": 12_000.0,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "groupItemTagUUIDs": ["tag-1", "tag-2"],
    }

    record = normalize_polymarket_market(dict[str, object](raw), min_close_date=date(2023, 1, 1), min_volume=5000.0)

    assert record is not None
    assert record.resolution is True
    assert record.market_price_at_open is None  # Terminal prices, not opening
    assert record.category == "general"


def test_polymarket_resolution_no() -> None:
    raw = {
        "conditionId": "0xdef",
        "question": "Will tests fail?",
        "startDate": "2024-03-01T00:00:00Z",
        "endDate": "2024-04-01T00:00:00Z",
        "volumeNum": 8_000.0,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0", "1"]),
    }

    record = normalize_polymarket_market(dict[str, object](raw), min_close_date=date(2023, 1, 1), min_volume=5000.0)

    assert record is not None
    assert record.resolution is False


def test_polymarket_category_extraction() -> None:
    """Categories should be extracted from tags or category field."""

    raw_tags = {
        "conditionId": "0xtest",
        "question": "Category test",
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-02-01T00:00:00Z",
        "volumeNum": 10_000.0,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "tags": ["politics", "us-elections"],
    }
    record = normalize_polymarket_market(
        dict[str, object](raw_tags), min_close_date=date(2023, 1, 1), min_volume=5000.0
    )
    assert record is not None
    assert record.category == "politics"

    raw_cat = {
        "conditionId": "0xcat",
        "question": "Category test 2",
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-02-01T00:00:00Z",
        "volumeNum": 10_000.0,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0", "1"]),
        "category": "crypto",
    }
    record2 = normalize_polymarket_market(
        dict[str, object](raw_cat), min_close_date=date(2023, 1, 1), min_volume=5000.0
    )
    assert record2 is not None
    assert record2.category == "crypto"


def test_manifold_opening_probability_parameter() -> None:
    raw = {
        "id": "manifold-open",
        "question": "Opening override?",
        "createdTime": 1_700_000_000_000,
        "closeTime": 1_701_000_000_000,
        "isResolved": True,
        "outcomeType": "BINARY",
        "resolution": "YES",
        "probability": 0.10,
        "tags": ["science"],
    }

    record = normalize_manifold_market(
        dict[str, object](raw),
        min_close_time_ms=0,
        opening_probability=0.88,
    )

    assert record is not None
    assert record.market_price_at_open == pytest.approx(0.88)


def test_manifold_field_mapping() -> None:
    raw = {
        "id": "manifold-1",
        "question": "Will the harness finish?",
        "createdTime": 1_700_000_000_000,
        "closeTime": 1_701_000_000_000,
        "isResolved": True,
        "outcomeType": "BINARY",
        "resolution": "YES",
        "probability": 0.42,
        "tags": ["science", "meta"],
    }

    record = normalize_manifold_market(dict[str, object](raw), min_close_time_ms=0)

    assert record is not None
    assert record.resolution is True
    assert record.market_price_at_open == pytest.approx(0.42)
    assert record.category == "science"


def test_skips_unresolved() -> None:
    raw = {
        "conditionId": "0xmissing",
        "question": "Unresolved market",
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-02-01T00:00:00Z",
        "volumeNum": 20_000.0,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.5", "0.5"]),
    }

    assert (
        normalize_polymarket_market(dict[str, object](raw), min_close_date=date(2023, 1, 1), min_volume=1000.0)
        is None
    )


def test_skips_non_binary() -> None:
    raw = {
        "conditionId": "0xpoly",
        "question": "Who wins?",
        "startDate": "2024-01-01T00:00:00Z",
        "endDate": "2024-02-01T00:00:00Z",
        "volumeNum": 20_000.0,
        "outcomes": json.dumps(["Alice", "Bob"]),
        "outcomePrices": json.dumps(["1", "0"]),
    }

    assert (
        normalize_polymarket_market(dict[str, object](raw), min_close_date=date(2023, 1, 1), min_volume=1000.0)
        is None
    )


def test_validation_date_order() -> None:
    with pytest.raises(ValueError):
        BacktestQuestion(
            question_id="bad-dates",
            source="polymarket",
            question_text="invalid window",
            open_date=date(2024, 2, 1),
            close_date=date(2024, 1, 1),
            resolution=True,
            market_price_at_open=0.5,
            category=None,
        )


def test_validation_price_range() -> None:
    with pytest.raises(ValueError):
        BacktestQuestion(
            question_id="bad-price",
            source="polymarket",
            question_text="invalid price",
            open_date=date(2024, 1, 1),
            close_date=date(2024, 2, 1),
            resolution=True,
            market_price_at_open=1.2,
            category=None,
        )
