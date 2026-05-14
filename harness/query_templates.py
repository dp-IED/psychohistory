"""Deterministic blind-spot query templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from harness.policy.blind_spot_checks import (
    base_rate_check,
    current_state_check,
    markets_price_check,
    policy_decision_check,
    resolution_criteria_check,
)

if TYPE_CHECKING:
    from harness.query_mapper import MarketFrame, WebSearchRequest


def _actor_clause(frame: "MarketFrame") -> str:
    if not frame.key_actors:
        return ""
    return " " + " OR ".join(f'"{actor}"' for actor in frame.key_actors)


def _region_clause(frame: "MarketFrame") -> str:
    return f' "{frame.region}"' if frame.region else ""


def electoral_legitimacy_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("election" OR "turnout" OR "fraud" OR "observer report")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="electoral_legitimacy_check",
    )


def coalition_stability_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("coalition" OR "no-confidence" OR "cabinet reshuffle" OR "defection")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="coalition_stability_check",
    )


def sanctions_escalation_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("sanctions" OR "export controls" OR "asset freeze" OR "secondary sanctions")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="sanctions_escalation_check",
    )


def leadership_succession_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("succession" OR "health rumors" OR "interim leadership" OR "power struggle")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="leadership_succession_check",
    )


def treaty_compliance_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("treaty" OR "compliance" OR "verification" OR "inspections")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="treaty_compliance_check",
    )


TEMPLATE_REGISTRY: dict[str, Callable[["MarketFrame"], "WebSearchRequest"]] = {
    "electoral_legitimacy_check": electoral_legitimacy_template,
    "coalition_stability_check": coalition_stability_template,
    "sanctions_escalation_check": sanctions_escalation_template,
    "leadership_succession_check": leadership_succession_template,
    "treaty_compliance_check": treaty_compliance_template,
    "base_rate_check": base_rate_check,
    "current_state_check": current_state_check,
    "resolution_criteria_check": resolution_criteria_check,
    "markets_price_check": markets_price_check,
    "policy_decision_check": policy_decision_check,
}
