from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from harness.query_templates import TEMPLATE_REGISTRY


class PITViolationError(ValueError):
    """Raised when a query attempts to access information after a PIT cutoff."""


class UnknownCheckError(ValueError):
    """Raised when a blind-spot check is not in the template registry."""


@dataclass(frozen=True)
class WebSearchRequest:
    """A structured request for web evidence retrieval."""

    query: str
    as_of_date: date
    blind_spot_check: str | None = None
    market_family: str = "binary"

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")


@dataclass(frozen=True)
class MarketFrame:
    """Minimal market descriptor used by blind-spot checks."""

    market_family: str
    question: str
    cutoff_date: date
    key_actors: list[str] = field(default_factory=list)
    region: str | None = None

    def __post_init__(self) -> None:
        if not self.market_family.strip():
            raise ValueError("market_family must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        for i, actor in enumerate(self.key_actors):
            if not isinstance(actor, str):
                raise ValueError(
                    f"key_actors[{i}] must be str, got {type(actor).__name__}"
                )


# ── Blind-spot → query mapper ────────────────────────────────────────

LLMFallback = Callable[[str, MarketFrame], WebSearchRequest]


def blind_spot_to_query(
    check: str,
    frame: MarketFrame,
    *,
    llm_fallback: LLMFallback | None = None,
) -> WebSearchRequest:
    """Map a blind-spot check name to a WebSearchRequest.

    Uses TEMPLATE_REGISTRY if the check is registered, otherwise falls
    back to the llm_fallback callable (or raises UnknownCheckError).
    """
    template_fn = TEMPLATE_REGISTRY.get(check)

    if template_fn is not None:
        request = template_fn(frame)
    elif llm_fallback is not None:
        request = llm_fallback(check, frame)
    else:
        raise UnknownCheckError(
            f"Unknown blind-spot check {check!r} — not in "
            f"TEMPLATE_REGISTRY and no llm_fallback provided"
        )

    # PIT guard: request date must not exceed frame cutoff
    if request.as_of_date > frame.cutoff_date:
        raise PITViolationError(
            f"WebSearchRequest as_of_date ({request.as_of_date}) "
            f"exceeds cutoff ({frame.cutoff_date})"
        )

    return request
