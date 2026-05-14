"""Deterministic v0 evidence classification from search snippets (no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from harness.tools.web_search import SearchResult

_POSITIVE_SIGNALS = frozenset(
    {
        "yes",
        "confirmed",
        "announce",
        "announced",
        "signed",
        "agreed",
        "approved",
        "passed",
        "success",
        "successful",
        "completed",
        "reached",
        "deal",
        "ratified",
        "adopted",
        "likely",
        "expected",
        "poised",
        "advance",
        "breakthrough",
        "won",
        "win",
        "victory",
        "triumph",
        "will",
    }
)

_NEGATIVE_SIGNALS = frozenset(
    {
        "no",
        "not",
        "denied",
        "denies",
        "failed",
        "failure",
        "blocked",
        "rejected",
        "unlikely",
        "abandoned",
        "cancelled",
        "canceled",
        "ruled",
        "collapse",
        "impasse",
        "stalled",
        "withdraw",
        "withdrawn",
        "oppose",
        "opposes",
        "halt",
        "scrap",
        "scuttled",
        "lost",
        "lose",
        "defeat",
    }
)


def _question_keyword_tokens(question: str) -> frozenset[str]:
    raw = re.findall(r"[a-z0-9]+", question.lower())
    return frozenset(w for w in raw if len(w) > 2)


def _count_word_signals(blob: str) -> tuple[int, int]:
    """Return (positive_hits, negative_hits) using whole-word matches."""
    lower = blob.lower()
    pos = 0
    neg = 0
    for w in _POSITIVE_SIGNALS:
        pos += len(re.findall(rf"\b{re.escape(w)}\b", lower))
    for w in _NEGATIVE_SIGNALS:
        neg += len(re.findall(rf"\b{re.escape(w)}\b", lower))
    return pos, neg


def _classify_result(
    item: SearchResult,
    q_keywords: frozenset[str],
) -> Literal["yes", "no", "ambiguous"]:
    text = f"{item.title} {item.summary}".strip()
    if not text:
        return "ambiguous"

    tokens = frozenset(re.findall(r"[a-z0-9]+", text.lower()))
    overlap = len(tokens & q_keywords)

    pos_hits, neg_hits = _count_word_signals(text)
    if overlap == 0 and pos_hits == 0 and neg_hits == 0:
        return "ambiguous"

    margin = 1
    if pos_hits > neg_hits + margin:
        return "yes"
    if neg_hits > pos_hits + margin:
        return "no"
    return "ambiguous"


def _label_for(item: SearchResult) -> str:
    label = item.title.strip() or item.url.strip() or item.summary.strip()[:80]
    return label if label else "(untitled)"


@dataclass(frozen=True)
class EvidenceAssessment:
    supports_yes: list[str] = field(default_factory=list)
    supports_no: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    evidence_strength: Literal["strong", "moderate", "weak", "none"] = "none"


def synthesise_evidence(
    results: list[SearchResult],
    question: str,
    prior: float,
) -> tuple[EvidenceAssessment, float]:
    """
    Classify search results as supporting YES/NO and compute adjusted p_yes.

    Strong YES: enough aligned snippets → pull probability up (capped).
    Strong NO: aligned negative snippets → pull down (floored).
    Otherwise retain the prior (weak or mixed evidence).
    """
    prior_f = float(prior)
    prior_f = max(0.0, min(1.0, prior_f))

    if not results:
        return (
            EvidenceAssessment(evidence_strength="none", uncertainty_flags=["no_search_results"]),
            prior_f,
        )

    q_keywords = _question_keyword_tokens(question)

    supports_yes: list[str] = []
    supports_no: list[str] = []
    ambiguous = 0

    for item in results:
        bucket = _classify_result(item, q_keywords)
        label = _label_for(item)
        if bucket == "yes":
            supports_yes.append(label)
        elif bucket == "no":
            supports_no.append(label)
        else:
            ambiguous += 1

    yes_n = len(supports_yes)
    no_n = len(supports_no)
    flags: list[str] = []
    if ambiguous:
        flags.append(f"ambiguous_snippets={ambiguous}")

    strong_yes = yes_n >= 2 and yes_n >= no_n + 2
    strong_no = no_n >= 2 and no_n >= yes_n + 2

    if strong_yes:
        p_yes = min(prior_f * 2.5, 0.85)
        strength: Literal["strong", "moderate", "weak", "none"] = "strong"
    elif strong_no:
        p_yes = max(prior_f * 0.3, 0.05)
        strength = "strong"
    else:
        p_yes = prior_f
        if yes_n == 0 and no_n == 0:
            strength = "weak"
            flags.append("no_directional_keyword_signal")
        elif abs(yes_n - no_n) <= 1 and (yes_n + no_n) >= 2:
            strength = "weak"
            flags.append("mixed_keyword_signal")
        elif max(yes_n, no_n) >= 1:
            strength = "moderate"
        else:
            strength = "weak"

    return (
        EvidenceAssessment(
            supports_yes=supports_yes,
            supports_no=supports_no,
            uncertainty_flags=flags,
            evidence_strength=strength,
        ),
        p_yes,
    )
