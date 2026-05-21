from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from harness.tools.web_search import SearchResult


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_strength: Literal["none", "weak", "moderate", "strong"]
    supports_yes: list[str] = field(default_factory=list)
    supports_no: list[str] = field(default_factory=list)


def synthesise_evidence(
    results: list[SearchResult],
    *,
    question: str,
    prior: float,
) -> tuple[EvidenceAssessment, float]:
    """Synthesise web search results into an evidence assessment and updated probability.

    Contract (from tests):
    - Empty results → prior unchanged, strength "none"
    - Strong yes evidence (2+ yes-keyword items) → p_yes = min(prior * 2.5, 0.85)
    - Strong no evidence (2+ no-keyword items) → p_yes = max(prior * 0.3, 0.05)
    - Weak evidence → returns prior
    - Mixed evidence → returns prior, strength != "strong"
    """
    if not results:
        return (
            EvidenceAssessment(evidence_strength="none"),
            prior,
        )

    # Classify each result as supporting yes or no via keyword matching
    yes_keywords = (
        "confirmed", "victory", "success", "signed", "agreement",
        "announced", "approved", "passed", "won", "achieved",
        "reached", "deal", "breakthrough",
    )
    no_keywords = (
        "failed", "rejected", "blocked", "denied", "collapse",
        "stalemate", "deadlock", "breakdown", "vetoed", "withdrew",
        "postponed", "delayed", "abandoned", "cancelled",
    )

    supports_yes: list[str] = []
    supports_no: list[str] = []

    for r in results:
        text = (r.title + " " + r.summary).lower()
        yes_hits = sum(1 for kw in yes_keywords if kw in text)
        no_hits = sum(1 for kw in no_keywords if kw in text)

        if yes_hits > no_hits:
            supports_yes.append(r.title)
        elif no_hits > yes_hits:
            supports_no.append(r.title)

    # Determine strength and update probability
    yes_count = len(supports_yes)
    no_count = len(supports_no)

    # Mixed evidence: equal or both present → prior
    if yes_count > 0 and no_count > 0:
        return (
            EvidenceAssessment(
                evidence_strength="moderate",
                supports_yes=supports_yes,
                supports_no=supports_no,
            ),
            prior,
        )

    # Single-directional evidence
    if yes_count >= 2:
        strength = "strong"
        p_yes = min(prior * 2.5, 0.85)
    elif yes_count == 1:
        strength = "moderate"
        p_yes = prior
    elif no_count >= 2:
        strength = "strong"
        p_yes = max(prior * 0.3, 0.05)
    elif no_count == 1:
        strength = "moderate"
        p_yes = prior
    else:
        strength = "weak"
        p_yes = prior

    return (
        EvidenceAssessment(
            evidence_strength=strength,
            supports_yes=supports_yes,
            supports_no=supports_no,
        ),
        p_yes,
    )
