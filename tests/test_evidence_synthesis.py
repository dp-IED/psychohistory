from __future__ import annotations

from datetime import date

from harness.policy.evidence_synthesis import synthesise_evidence
from harness.tools.web_search import SearchResult


def _item(title: str, summary: str = "", url: str = "https://example.com/a") -> SearchResult:
    return SearchResult(
        title=title,
        summary=summary,
        url=url,
        published_at=date(2026, 5, 1),
        source="src",
    )


def test_empty_results_returns_prior() -> None:
    assessment, p_yes = synthesise_evidence([], question="Will X happen?", prior=0.42)
    assert p_yes == 0.42
    assert assessment.evidence_strength == "none"
    assert assessment.supports_yes == []
    assert assessment.supports_no == []


def test_strong_yes_evidence_increases_probability() -> None:
    results = [
        _item("Ceasefire deal confirmed victory", url="https://e/1"),
        _item("Agreement signed success reached", url="https://e/2"),
    ]
    _, p_yes = synthesise_evidence(results, question="Will ceasefire be reached?", prior=0.4)
    assert p_yes == min(0.4 * 2.5, 0.85)


def test_strong_no_evidence_decreases_probability() -> None:
    results = [
        _item("Ceasefire talks failed rejected", url="https://e/1"),
        _item("Peace plan abandoned cancelled", url="https://e/2"),
    ]
    _, p_yes = synthesise_evidence(results, question="Will ceasefire be reached?", prior=0.5)
    assert p_yes == max(0.5 * 0.3, 0.05)


def test_weak_evidence_returns_prior() -> None:
    results = [_item("Routine agenda meeting notes", summary="no strong wording")]
    assessment, p_yes = synthesise_evidence(results, question="Will ceasefire be reached?", prior=0.37)
    assert p_yes == 0.37
    assert assessment.evidence_strength in {"weak", "moderate"}


def test_mixed_evidence_returns_prior() -> None:
    results = [
        _item("Deal confirmed by side A", url="https://e/1"),
        _item("Opposition rejected failed talks", url="https://e/2"),
    ]
    assessment, p_yes = synthesise_evidence(results, question="Will ceasefire be reached?", prior=0.55)
    assert p_yes == 0.55
    assert assessment.evidence_strength != "strong"
