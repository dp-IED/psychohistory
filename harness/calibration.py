from __future__ import annotations

from dataclasses import dataclass, field

from harness.memory_schema import EpisodicRecord
from harness.memory_store import MemoryStore


@dataclass(frozen=True)
class CalibrationReport:
    insufficient_data: bool
    resolved_count: int
    overconfidence_bias: float = 0.0
    suggested_shrinkage: float = 0.0
    overall_brier: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)


def compute_calibration(memory: MemoryStore) -> CalibrationReport:
    """Compute calibration diagnostics from resolved episodes in memory.

    Shrinkage is capped at 0.3 and only activates after >= 10 resolved
    episodes.
    """
    # Collect all resolved episodes (with Brier scores) across families
    resolved: list[EpisodicRecord] = []
    for family in (
        "event_negotiation",
        "macro_rates",
        "metaculus_binary",
        "binary",
        "polymarket_binary",
        "politics",
        "economics",
        "test_family",
    ):
        for ep in memory.read_recent_episodes(family, 1000):
            if ep.brier_score is not None:
                resolved.append(ep)

    if len(resolved) < 10:
        return CalibrationReport(
            insufficient_data=True,
            resolved_count=len(resolved),
        )

    # ── overall Brier ────────────────────────────────────────────
    overall_brier = sum(ep.brier_score for ep in resolved if ep.brier_score is not None) / len(resolved)  # type: ignore[arg-type]

    # ── by_category ──────────────────────────────────────────────
    by_category: dict[str, list[float]] = {}
    for ep in resolved:
        cat = ep.market_family
        if ep.brier_score is not None:
            by_category.setdefault(cat, []).append(ep.brier_score)  # type: ignore[arg-type]

    by_category_means = {
        cat: sum(scores) / len(scores)
        for cat, scores in by_category.items()
        if scores
    }

    # ── overconfidence bias ──────────────────────────────────────
    # Formula: sqrt(mean_brier) - 0.5
    # Well-calibrated (Brier=0.25) → bias=0; max overconfidence (Brier=1.0) → bias=0.5
    import math

    overconfidence_bias = math.sqrt(overall_brier) - 0.5
    suggested_shrinkage = min(0.3, overconfidence_bias * 0.6)

    return CalibrationReport(
        insufficient_data=False,
        resolved_count=len(resolved),
        overconfidence_bias=overconfidence_bias,
        suggested_shrinkage=suggested_shrinkage,
        overall_brier=overall_brier,
        by_category=by_category_means,
    )


def apply_shrinkage(p_yes: float, shrinkage: float) -> float:
    """Shrink a probability toward 0.5 by the given amount.

    p_yes=0.9, shrinkage=0.24 → 0.9 - (0.9-0.5)*0.24 = 0.804
    """
    return p_yes - (p_yes - 0.5) * shrinkage
