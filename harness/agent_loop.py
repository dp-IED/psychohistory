"""Competition-ready agent loop core, with platform adapters kept out-of-band.

The loop stays pure: it returns structured results and writes episodic memory,
while API posting (e.g., Metaculus) lives in separate client modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable
from uuid import uuid4

from harness.calibration import apply_shrinkage
from harness.memory_schema import EpisodicRecord, ToolCallRecord
from harness.memory_store import MemoryStore
from harness.query_mapper import MarketFrame, blind_spot_to_query


@dataclass(frozen=True)
class MarketContext:
    market_id: str
    market_family: str
    key_actors: list[str]
    region: str | None
    cutoff_date: date
    resolution_date: date


@dataclass(frozen=True)
class GraphQueryResult:
    node_count: int
    notes: str = ""


@dataclass(frozen=True)
class ConstructionPolicy:
    blind_spot_checks: list[str]
    max_steps: int = 4
    convergence_epsilon: float = 0.01
    shrinkage: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.convergence_epsilon < 0:
            raise ValueError("convergence_epsilon must be >= 0")
        if self.shrinkage is not None and not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError("shrinkage must be in [0, 1] when set")


@dataclass(frozen=True)
class AgentToolset:
    web_search: Callable[[str, date], list[ToolCallRecord]]
    graph_query: Callable[[str, date], GraphQueryResult]
    gnn_score: Callable[[int, int, int], float]
    analogues: Callable[[str], list[str]]
    market_context: Callable[[str, date, date], MarketContext]


@dataclass
class AgentLoopState:
    job_id: str
    checks_fired: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    gnn_score_trajectory: list[float] = field(default_factory=list)
    step: int = 0


@dataclass(frozen=True)
class AgentLoopResult:
    job_id: str
    final_p_yes: float
    confidence_interval: tuple[float, float] | None
    reasoning_summary: str
    blind_spot_checks_fired: list[str]
    blind_spot_checks_skipped: list[str]
    gnn_score_trajectory: list[float]
    tool_call_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning_summary, str) or not self.reasoning_summary.strip():
            raise ValueError("reasoning_summary must be a non-empty string")
        _require_float01(self.final_p_yes, "final_p_yes")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            _require_float01(lo, "confidence_interval[0]")
            _require_float01(hi, "confidence_interval[1]")
            if lo > hi:
                raise ValueError("confidence_interval lower bound must be <= upper bound")


def _require_float01(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float in [0, 1]")
    fv = float(value)
    if fv < 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be a float in [0, 1]")


def _planner_checks(policy: ConstructionPolicy) -> tuple[list[str], list[str]]:
    if not policy.blind_spot_checks:
        return [], []
    # Minimal planner: execute all checks in v0.
    return list(policy.blind_spot_checks), []


def _converged(scores: list[float], eps: float) -> bool:
    if len(scores) < 3:
        return False
    d1 = abs(scores[-1] - scores[-2])
    d2 = abs(scores[-2] - scores[-3])
    return d1 < eps and d2 < eps


def run_agent_loop(
    question: str,
    cutoff_date: date,
    resolution_date: date,
    policy: ConstructionPolicy,
    memory: MemoryStore,
    tools: AgentToolset,
) -> AgentLoopResult:
    """Run the five-phase v0 loop: load memory -> plan -> research -> synthesize -> write episode."""
    if cutoff_date > resolution_date:
        raise ValueError("cutoff_date must be <= resolution_date")

    # 1) Load memory (read side kept lightweight for v0 wiring).
    context = tools.market_context(question, cutoff_date, resolution_date)
    _ = memory.read_recent_episodes(context.market_family, 5)
    _ = memory.read_patterns(context.market_family)

    # 2) Plan.
    fired, skipped = _planner_checks(policy)
    state = AgentLoopState(job_id=f"job-{uuid4().hex[:12]}", checks_fired=fired, checks_skipped=skipped)

    frame = MarketFrame(
        market_family=context.market_family,
        question=question,
        cutoff_date=cutoff_date,
        key_actors=context.key_actors,
        region=context.region,
    )

    node_count = 0
    # 3) Research loop.
    for step in range(1, policy.max_steps + 1):
        state.step = step

        for check in state.checks_fired:
            req = blind_spot_to_query(check, frame)
            state.tool_calls.extend(tools.web_search(req.query, req.as_of_date))

        graph_result = tools.graph_query(question, cutoff_date)
        node_count = max(node_count, graph_result.node_count)

        # Optional auxiliary signal (kept outside confidence math for now).
        _ = tools.analogues(question)

        score = tools.gnn_score(step, len(state.tool_calls), node_count)
        _require_float01(score, "gnn_score")
        state.gnn_score_trajectory.append(score)

        if _converged(state.gnn_score_trajectory, policy.convergence_epsilon):
            break

    # 4) Synthesize.
    if state.gnn_score_trajectory:
        final_p_yes = state.gnn_score_trajectory[-1]
    else:
        final_p_yes = 0.5

    final_p_yes = max(0.0, min(1.0, float(final_p_yes)))

    if policy.shrinkage is not None:
        final_p_yes = apply_shrinkage(final_p_yes, policy.shrinkage)
        final_p_yes = max(0.0, min(1.0, float(final_p_yes)))

    if state.gnn_score_trajectory:
        lo = max(0.0, final_p_yes - 0.08)
        hi = min(1.0, final_p_yes + 0.08)
        confidence_interval: tuple[float, float] | None = (lo, hi)
    else:
        confidence_interval = None

    reasoning_summary = (
        f"Assessed '{question}' as of {cutoff_date.isoformat()} using "
        f"{len(state.tool_calls)} evidence calls, {node_count} retrieved nodes, and "
        f"{len(state.gnn_score_trajectory)} scoring steps. Final p_yes={final_p_yes:.3f}."
    )

    result = AgentLoopResult(
        job_id=state.job_id,
        final_p_yes=final_p_yes,
        confidence_interval=confidence_interval,
        reasoning_summary=reasoning_summary,
        blind_spot_checks_fired=list(state.checks_fired),
        blind_spot_checks_skipped=list(state.checks_skipped),
        gnn_score_trajectory=list(state.gnn_score_trajectory),
        tool_call_count=len(state.tool_calls),
    )

    # 5) Write episode.
    episode = EpisodicRecord(
        job_id=state.job_id,
        market_id=context.market_id,
        market_family=context.market_family,
        question=question,
        resolution_date=resolution_date,
        cutoff_date=cutoff_date,
        blind_spot_checks_fired=list(state.checks_fired),
        blind_spot_checks_skipped=list(state.checks_skipped),
        tool_calls=list(state.tool_calls),
        subgraph_node_count=node_count,
        gnn_score_trajectory=list(state.gnn_score_trajectory),
        final_p_yes=result.final_p_yes,
        confidence_interval=result.confidence_interval,
        brier_score=None,
        misses=[],
        notes=result.reasoning_summary,
    )
    memory.write_episode(episode)

    return result


__all__ = [
    "AgentLoopResult",
    "AgentLoopState",
    "AgentToolset",
    "ConstructionPolicy",
    "GraphQueryResult",
    "MarketContext",
    "run_agent_loop",
]
