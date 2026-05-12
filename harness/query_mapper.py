"""Blind-spot check to web-search request mapping with PIT guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from harness.query_templates import TEMPLATE_REGISTRY


class UnknownCheckError(ValueError):
    """Raised when a blind-spot check has no deterministic template or fallback."""


class PITViolationError(ValueError):
    """Raised when a generated query leaks beyond the episode cutoff date."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class MarketFrame:
    market_family: str
    question: str
    cutoff_date: date
    key_actors: list[str]
    region: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.market_family, "market_family")
        _require_non_empty(self.question, "question")
        if not isinstance(self.key_actors, list) or not all(isinstance(actor, str) for actor in self.key_actors):
            raise ValueError("key_actors must be list[str]")


@dataclass(frozen=True)
class WebSearchRequest:
    query: str
    as_of_date: date
    market_family: str
    blind_spot_check: str

    def __post_init__(self) -> None:
        _require_non_empty(self.query, "query")


def blind_spot_to_query(
    check: str,
    frame: MarketFrame,
    llm_fallback: Callable[[str, MarketFrame], WebSearchRequest] | None = None,
) -> WebSearchRequest:
    template = TEMPLATE_REGISTRY.get(check)
    if template is not None:
        result = template(frame)
    elif llm_fallback is not None:
        result = llm_fallback(check, frame)
    else:
        raise UnknownCheckError(f"Unknown blind-spot check: {check}")

    if result.as_of_date > frame.cutoff_date:
        raise PITViolationError(
            f"Query as_of_date {result.as_of_date} exceeds cutoff {frame.cutoff_date}"
        )

    return result
