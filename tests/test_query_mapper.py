from __future__ import annotations

from datetime import date, timedelta

import pytest

from harness.query_mapper import (
    MarketFrame,
    PITViolationError,
    UnknownCheckError,
    WebSearchRequest,
    blind_spot_to_query,
)
from harness.query_templates import TEMPLATE_REGISTRY


def _frame() -> MarketFrame:
    return MarketFrame(
        market_family="event_negotiation",
        question="Will a ceasefire be announced before June 1?",
        cutoff_date=date(2026, 5, 20),
        key_actors=["Mediator X", "Faction Y"],
        region="MENA",
    )


@pytest.mark.parametrize("check", sorted(TEMPLATE_REGISTRY.keys()))
def test_template_matching_returns_expected_contract(check: str) -> None:
    frame = _frame()

    request = blind_spot_to_query(check, frame)

    assert isinstance(request, WebSearchRequest)
    assert request.blind_spot_check == check
    assert request.market_family == frame.market_family
    assert request.as_of_date <= frame.cutoff_date
    assert request.query


def test_llm_fallback_used_for_unknown_check() -> None:
    frame = _frame()
    seen: list[tuple[str, MarketFrame]] = []

    def fallback(check: str, passed_frame: MarketFrame) -> WebSearchRequest:
        seen.append((check, passed_frame))
        return WebSearchRequest(
            query="fallback query",
            as_of_date=passed_frame.cutoff_date,
            market_family=passed_frame.market_family,
            blind_spot_check=check,
        )

    request = blind_spot_to_query("unseen_check", frame, llm_fallback=fallback)

    assert seen == [("unseen_check", frame)]
    assert request.query == "fallback query"
    assert request.blind_spot_check == "unseen_check"


def test_unknown_check_without_fallback_raises() -> None:
    frame = _frame()

    with pytest.raises(UnknownCheckError, match="unknown_check"):
        blind_spot_to_query("unknown_check", frame)


def test_pit_guard_rejects_post_cutoff_fallback_result() -> None:
    frame = _frame()

    def violating_fallback(check: str, passed_frame: MarketFrame) -> WebSearchRequest:
        return WebSearchRequest(
            query="violating query",
            as_of_date=passed_frame.cutoff_date + timedelta(days=1),
            market_family=passed_frame.market_family,
            blind_spot_check=check,
        )

    with pytest.raises(PITViolationError, match="exceeds cutoff"):
        blind_spot_to_query("unknown_check", frame, llm_fallback=violating_fallback)


def test_pit_guard_rejects_post_cutoff_template_result(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()

    def violating_template(passed_frame: MarketFrame) -> WebSearchRequest:
        return WebSearchRequest(
            query="violating template query",
            as_of_date=passed_frame.cutoff_date + timedelta(days=1),
            market_family=passed_frame.market_family,
            blind_spot_check="electoral_legitimacy_check",
        )

    monkeypatch.setitem(TEMPLATE_REGISTRY, "electoral_legitimacy_check", violating_template)

    with pytest.raises(PITViolationError, match="exceeds cutoff"):
        blind_spot_to_query("electoral_legitimacy_check", frame)


def test_market_frame_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="market_family"):
        MarketFrame(
            market_family="",
            question="Will X happen?",
            cutoff_date=date(2026, 5, 20),
            key_actors=["Actor A"],
            region=None,
        )

    with pytest.raises(ValueError, match="question"):
        MarketFrame(
            market_family="event_negotiation",
            question="",
            cutoff_date=date(2026, 5, 20),
            key_actors=["Actor A"],
            region=None,
        )

    with pytest.raises(ValueError, match="key_actors"):
        MarketFrame(
            market_family="event_negotiation",
            question="Will X happen?",
            cutoff_date=date(2026, 5, 20),
            key_actors=["Actor A", 42],  # type: ignore[list-item]
            region=None,
        )


def test_web_search_request_validates_non_empty_query() -> None:
    with pytest.raises(ValueError, match="query"):
        WebSearchRequest(
            query="",
            as_of_date=date(2026, 5, 20),
            market_family="event_negotiation",
            blind_spot_check="coalition_stability_check",
        )
