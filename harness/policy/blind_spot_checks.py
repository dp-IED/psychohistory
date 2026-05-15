"""Additional blind-spot templates (base rate, current state, resolution criteria)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.query_mapper import MarketFrame


def base_rate_check(frame: MarketFrame) -> WebSearchRequest:
    """How often does this type of event occur historically?"""
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'"{frame.question}" '
            '("probability" OR "historical frequency" OR "base rate" OR '
            '"prior probability")'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="base_rate_check",
    )


def current_state_check(frame: MarketFrame) -> WebSearchRequest:
    """What is the current observable state relevant to this question?"""
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'"{frame.question}" '
            '("current status" OR "latest" OR "recent development" OR "recent news")'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="current_state_check",
    )


def resolution_criteria_check(frame: MarketFrame) -> WebSearchRequest:
    """What conditions would need to be met for the question to resolve YES?"""
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'"{frame.question}" '
            '("criteria" OR "threshold" OR "condition" OR "resolution rules")'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="resolution_criteria_check",
    )


def markets_price_check(frame: MarketFrame) -> WebSearchRequest:
    """Price-level and market data for financial questions."""
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'"{frame.question}" '
            '("price" OR "close above" OR "trading at" OR "market data" OR "exchange rate" OR "yield")'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="markets_price_check",
    )


def policy_decision_check(frame: MarketFrame) -> WebSearchRequest:
    """Central bank and regulatory policy decisions."""
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'"{frame.question}" '
            '("rate decision" OR "central bank" OR "monetary policy" OR "FOMC" OR "ECB" OR '
            '"regulatory ruling" OR "policy announcement" OR "Fed" OR "interest rate")'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="policy_decision_check",
    )
