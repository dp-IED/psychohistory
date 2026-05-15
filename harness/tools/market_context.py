"""Market context from Polymarket-style tags and keyword fallbacks."""

from __future__ import annotations

import hashlib
from datetime import date

from harness.agent_loop import MarketContext
from harness.corpus.backtest_corpus import TAG_TO_CATEGORY, _QUESTION_KEYWORD_HINTS


def _market_id_from_question(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"q-{digest}"


def _category_from_tags(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    for raw in tags:
        token = raw.strip().lower()
        if not token:
            continue
        if token in TAG_TO_CATEGORY:
            return TAG_TO_CATEGORY[token]
        for part in token.replace("/", "-").split("-"):
            p = part.strip()
            if p in TAG_TO_CATEGORY:
                return TAG_TO_CATEGORY[p]
    return None


def _category_from_question_text(question: str) -> str:
    lowered = question.lower()
    for keywords, cat in _QUESTION_KEYWORD_HINTS:
        if any(kw in lowered for kw in keywords):
            return cat
    return "general"


def resolve_market_context(
    question: str,
    cutoff_date: date,
    resolution_date: date,
    market_id: str | None = None,
    tags: list[str] | None = None,
) -> MarketContext:
    mid = market_id or _market_id_from_question(question)
    from_tags = _category_from_tags(tags)
    family = from_tags or _category_from_question_text(question)
    return MarketContext(
        market_id=mid,
        market_family=family,
        key_actors=[],
        region=None,
        cutoff_date=cutoff_date,
        resolution_date=resolution_date,
    )
