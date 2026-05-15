"""Deterministic blind-spot query templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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


def geopolitical_stability_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("geopolitical risk" OR "instability" OR "protests" OR "coup" OR "border" OR "sanctions")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="geopolitical_stability_check",
    )


def economic_condition_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("GDP" OR "inflation" OR "recession" OR "unemployment" OR "interest rates" OR '
            f'"central bank" OR "commodities" OR "trade deficit")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="economic_condition_check",
    )


def insufficient_evidence_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("{frame.question}" OR background OR forecast OR prediction OR outlook OR latest news OR analysis)'
            f"{_actor_clause(frame)}{_region_clause(frame)}"
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="insufficient_evidence_check",
    )


def default_prior_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("base rate" OR probability OR likelihood OR "prediction market" OR Polymarket)'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="default_prior_check",
    )


def sports_match_calendar_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("match preview" OR fixture OR schedule OR kickoff OR postponed OR '
            f'"official lineup" OR "starting XI")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="sports_match_calendar_check",
    )


def base_rate_league_prior_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'("home win percentage" OR "1X2" OR odds OR Elo OR "league table" OR '
            f'"historical results" OR "title odds")'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="base_rate_league_prior_check",
    )


def injury_suspension_lineup_template(frame: "MarketFrame") -> "WebSearchRequest":
    from harness.query_mapper import WebSearchRequest

    return WebSearchRequest(
        query=(
            f'(injury OR suspended OR "doubtful" OR "match fitness" OR "team news" OR '
            f'absence OR ban)'
            f"{_actor_clause(frame)}{_region_clause(frame)} "
            f'"{frame.question}"'
        ),
        as_of_date=frame.cutoff_date,
        market_family=frame.market_family,
        blind_spot_check="injury_suspension_lineup_check",
    )


TEMPLATE_REGISTRY: dict[str, Callable[["MarketFrame"], "WebSearchRequest"]] = {
    "electoral_legitimacy_check": electoral_legitimacy_template,
    "geopolitical_stability_check": geopolitical_stability_template,
    "economic_condition_check": economic_condition_template,
    "insufficient_evidence_check": insufficient_evidence_template,
    "default_prior_check": default_prior_template,
    "coalition_stability_check": coalition_stability_template,
    "sanctions_escalation_check": sanctions_escalation_template,
    "leadership_succession_check": leadership_succession_template,
    "treaty_compliance_check": treaty_compliance_template,
    "sports_match_calendar_check": sports_match_calendar_template,
    "base_rate_league_prior_check": base_rate_league_prior_template,
    "injury_suspension_lineup_check": injury_suspension_lineup_template,
}
