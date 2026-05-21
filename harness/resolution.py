from __future__ import annotations

from dataclasses import dataclass

from harness.agent_loop import AgentToolset
from harness.memory_store import MemoryStore


class AlreadyResolvedError(ValueError):
    """Raised when attempting to resolve an already-resolved episode."""


class EpisodeNotFoundError(KeyError):
    """Raised when attempting to resolve an unknown job_id."""


@dataclass(frozen=True)
class BrierUpdateResult:
    job_id: str
    market_id: str
    outcome: bool
    brier_score: float
    misses: list[str]
    p_yes_at_resolution: float


def resolve_market(
    *,
    job_id: str,
    outcome: bool,
    memory: MemoryStore,
    tools: AgentToolset,
) -> BrierUpdateResult:
    """Resolve a previously-forecasted market episode.

    Updates the episode with Brier score and identified misses,
    then returns a summary.
    """
    episode = memory.read_episode_by_id(job_id)
    if episode is None:
        raise EpisodeNotFoundError(job_id)
    if episode.brier_score is not None:
        raise AlreadyResolvedError(
            f"Episode {job_id} already resolved (Brier={episode.brier_score})"
        )

    p = episode.final_p_yes
    target = 1.0 if outcome else 0.0
    brier = (p - target) ** 2

    # Identify missed checks: any blind_spot_checks_skipped that would have
    # helped + structural diagnostics
    misses: list[str] = []
    if abs(p - target) > 0.3:
        misses.append("large_brier_deviation")
    for skipped in episode.blind_spot_checks_skipped:
        misses.append(f"missed_check:{skipped}")

    memory.update_episode_brier(
        job_id=job_id,
        brier_score=brier,
        misses=misses,
    )

    return BrierUpdateResult(
        job_id=job_id,
        market_id=episode.market_id,
        outcome=outcome,
        brier_score=brier,
        misses=misses,
        p_yes_at_resolution=p,
    )
