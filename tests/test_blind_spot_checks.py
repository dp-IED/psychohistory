from __future__ import annotations

from datetime import date

from harness.policy.blind_spot_checks import (
    base_rate_check,
    current_state_check,
    markets_price_check,
    policy_decision_check,
    resolution_criteria_check,
)
from harness.query_mapper import MarketFrame
from harness.query_templates import TEMPLATE_REGISTRY


def _frame() -> MarketFrame:
    return MarketFrame(
        market_family="metaculus_binary",
        question="Will Alpha reach Beta milestone before July 2026?",
        cutoff_date=date(2026, 5, 20),
        key_actors=[],
        region=None,
    )


def test_base_rate_check_returns_web_search_request() -> None:
    frame = _frame()
    req = base_rate_check(frame)
    assert req.blind_spot_check == "base_rate_check"
    assert frame.question in req.query
    assert "base rate" in req.query.lower()
    assert req.as_of_date == frame.cutoff_date
    assert req.market_family == frame.market_family


def test_current_state_check_returns_web_search_request() -> None:
    frame = _frame()
    req = current_state_check(frame)
    assert req.blind_spot_check == "current_state_check"
    assert frame.question in req.query
    assert "current status" in req.query.lower()
    assert req.as_of_date == frame.cutoff_date


def test_resolution_criteria_check_returns_web_search_request() -> None:
    frame = _frame()
    req = resolution_criteria_check(frame)
    assert req.blind_spot_check == "resolution_criteria_check"
    assert frame.question in req.query
    assert "criteria" in req.query.lower()
    assert req.as_of_date == frame.cutoff_date


def test_markets_price_check_returns_web_search_request() -> None:
    frame = _frame()
    req = markets_price_check(frame)
    assert req.blind_spot_check == "markets_price_check"
    assert frame.question in req.query
    assert '"price"' in req.query or '"close above"' in req.query
    assert req.as_of_date == frame.cutoff_date


def test_policy_decision_check_returns_web_search_request() -> None:
    frame = _frame()
    req = policy_decision_check(frame)
    assert req.blind_spot_check == "policy_decision_check"
    assert frame.question in req.query
    assert "central bank" in req.query.lower()
    assert req.as_of_date == frame.cutoff_date


def test_five_checks_registered_in_template_registry() -> None:
    assert TEMPLATE_REGISTRY["base_rate_check"] is base_rate_check
    assert TEMPLATE_REGISTRY["current_state_check"] is current_state_check
    assert TEMPLATE_REGISTRY["resolution_criteria_check"] is resolution_criteria_check
    assert TEMPLATE_REGISTRY["markets_price_check"] is markets_price_check
    assert TEMPLATE_REGISTRY["policy_decision_check"] is policy_decision_check
