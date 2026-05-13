"""Episode resolution + Brier updates.

Miss inference is heuristic at this stage: for each skipped blind-spot check we
compute a synthetic counterfactual score via ``tools.gnn_score`` using a small,
check-conditioned perturbation of step/evidence/nodes, then flag a miss iff the
counterfactual shifts by >0.05 toward the realized outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.agent_loop import AgentToolset
from harness.memory_store import MemoryStore


class EpisodeNotFoundError(LookupError):
    """Raised when a resolution request references an unknown job_id."""


class AlreadyResolvedError(ValueError):
    """Raised when a Brier score is already written for this episode."""


@dataclass(frozen=True)
class BrierUpdateResult:
    job_id: str
    market_id: str
    outcome: bool
    brier_score: float
    misses: list[str]
    p_yes_at_resolution: float


def _infer_misses(job_id: str, outcome: bool, final_p_yes: float, skipped_checks: list[str], tools: AgentToolset) -> list[str]:
    misses: list[str] = []
    target = 1.0 if outcome else 0.0

    for idx, check in enumerate(skipped_checks):
        # Deterministic perturbation per check; lightweight until real
        # counterfactual subgraph filtering is wired.
        step = max(1, len(check) % 5 + 1)
        evidence = 1 + ((len(job_id) + idx) % 3)
        nodes = 6 + (sum(ord(c) for c in check) % 5)
        hypothetical_p = float(tools.gnn_score(step, evidence, nodes))

        delta = hypothetical_p - final_p_yes
        toward_outcome = (target - final_p_yes) * delta > 0
        if abs(delta) > 0.05 and toward_outcome:
            misses.append(check)

    return misses


def resolve_market(
    job_id: str,
    outcome: bool,
    memory: MemoryStore,
    tools: AgentToolset,
) -> BrierUpdateResult:
    episode = memory.read_episode_by_id(job_id)
    if episode is None:
        raise EpisodeNotFoundError(f"episode with job_id '{job_id}' not found")

    if episode.brier_score is not None:
        raise AlreadyResolvedError(f"episode with job_id '{job_id}' is already resolved")

    brier_score = (episode.final_p_yes - float(outcome)) ** 2
    misses = _infer_misses(job_id, outcome, episode.final_p_yes, episode.blind_spot_checks_skipped, tools)

    memory.update_episode_brier(job_id, brier_score, misses)

    return BrierUpdateResult(
        job_id=job_id,
        market_id=episode.market_id,
        outcome=outcome,
        brier_score=brier_score,
        misses=misses,
        p_yes_at_resolution=episode.final_p_yes,
    )


__all__ = [
    "AlreadyResolvedError",
    "BrierUpdateResult",
    "EpisodeNotFoundError",
    "resolve_market",
]
