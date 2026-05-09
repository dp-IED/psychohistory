import json

from ingest.polymarket_resolved import normalize_market


def test_normalize_resolved_binary_market() -> None:
    raw = {
        "id": "123",
        "slug": "will-x-happen",
        "question": "Will X happen?",
        "description": "Resolution text",
        "conditionId": "0xabc",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "clobTokenIds": json.dumps(["yes-token", "no-token"]),
        "volumeNum": 42.5,
    }

    record = normalize_market(raw)

    assert record is not None
    assert record.resolved_outcome == "Yes"
    assert record.terminal_outcome_prices == [1.0, 0.0]
    assert record.url == "https://polymarket.com/market/will-x-happen"


def test_normalize_rejects_non_terminal_prices() -> None:
    raw = {
        "id": "123",
        "slug": "will-x-happen",
        "question": "Will X happen?",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.6", "0.4"]),
    }

    assert normalize_market(raw) is None


def test_normalize_rejects_non_yes_no_market() -> None:
    raw = {
        "id": "123",
        "slug": "winner-market",
        "question": "Who wins?",
        "outcomes": json.dumps(["Alice", "Bob"]),
        "outcomePrices": json.dumps(["1", "0"]),
    }

    assert normalize_market(raw) is None
