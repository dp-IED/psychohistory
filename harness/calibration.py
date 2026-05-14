"""Calibration metrics from resolved episodic memory and shrinkage for synthesis.

``market_baseline_brier`` and ``agent_edge`` are reserved for callers that have
corpus metadata (e.g. ``market_price_at_open``); they are not computable from
``EpisodicRecord`` alone, which only stores ``final_p_yes`` and ``brier_score``.
Use backtest summaries when those comparisons are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean

from harness.memory_store import MemoryStore

MIN_EPISODES = 10


@dataclass
class CalibrationReport:
    overall_brier: float
    by_category: dict[str, float]

    # Baseline vs agent edge need corpus fields (e.g. market_price_at_open) not
    # present on EpisodicRecord; left as None here by design.
    market_baseline_brier: float | None
    agent_edge: float | None

    overconfidence_bias: float
    suggested_shrinkage: float
    episode_count: int
    resolved_count: int
    insufficient_data: bool
    generated_at: datetime


def compute_calibration(memory: MemoryStore) -> CalibrationReport:
    """Read all resolved EpisodicRecords from memory and compute calibration metrics.

    Returns suggested_shrinkage=0.0 and insufficient_data=True if fewer than
    MIN_EPISODES resolved episodes exist.
    """
    episodes = memory.read_all_episodes()
    resolved = [e for e in episodes if e.brier_score is not None]
    episode_count = len(episodes)
    resolved_count = len(resolved)

    if resolved_count < MIN_EPISODES:
        return CalibrationReport(
            overall_brier=0.0,
            by_category={},
            market_baseline_brier=None,
            agent_edge=None,
            overconfidence_bias=0.0,
            suggested_shrinkage=0.0,
            episode_count=episode_count,
            resolved_count=resolved_count,
            insufficient_data=True,
            generated_at=datetime.now(timezone.utc),
        )

    overall_brier = mean(e.brier_score for e in resolved)  # type: ignore[arg-type]

    by_family: dict[str, list[float]] = {}
    for e in resolved:
        by_family.setdefault(e.market_family, []).append(e.brier_score)  # type: ignore[arg-type]
    by_category = {family: mean(scores) for family, scores in by_family.items()}

    # TODO(calibration): Prefer explicit outcome on EpisodicRecord and classify
    # "wrong" via round(final_p_yes) != outcome. Until then, brier > 0.25 implies
    # |p - truth| > 0.5 (conservative wrong set; no false positives).
    wrong = [e for e in resolved if (e.brier_score or 0.0) > 0.25]
    overconfidence_bias = mean(abs(e.final_p_yes - 0.5) for e in wrong) if wrong else 0.0

    suggested_shrinkage = min(overconfidence_bias * 0.6, 0.3)

    return CalibrationReport(
        overall_brier=overall_brier,
        by_category=by_category,
        market_baseline_brier=None,
        agent_edge=None,
        overconfidence_bias=overconfidence_bias,
        suggested_shrinkage=suggested_shrinkage,
        episode_count=episode_count,
        resolved_count=resolved_count,
        insufficient_data=False,
        generated_at=datetime.now(timezone.utc),
    )


def apply_shrinkage(p_yes: float, shrinkage: float) -> float:
    """Pull p_yes toward 0.5 by shrinkage factor.

    0.0 = no change. 1.0 = returns 0.5.
    Typical shrinkage is in [0, 0.3].
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    return 0.5 + (p_yes - 0.5) * (1.0 - shrinkage)


__all__ = [
    "MIN_EPISODES",
    "CalibrationReport",
    "apply_shrinkage",
    "compute_calibration",
]
