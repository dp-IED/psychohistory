"""Typed agent-loop contracts for Polymarket branch expansion.

This module is intentionally an interface stub: the deterministic branch builder
creates bounded ``SubgraphPortfolio`` scaffolds, while this harness will own
Steps 2-N of an agentic run: PIT-safe search, graph/warehouse retrieval, GNN
stress scoring, branch expansion, and stop-condition evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from schemas.polymarket_agentic import (
    Branch,
    EvidenceRef,
    MarketFrame,
    RequirementStressTest,
    SubgraphPortfolio,
)


@dataclass(frozen=True)
class WebSearchRequest:
    """PIT-safe web-search request emitted by the agent loop."""

    query: str
    as_of_time: str
    max_results: int = 5
    branch_id: str | None = None


@dataclass(frozen=True)
class GraphQueryRequest:
    """Warehouse/graph lookup request scoped to one market branch."""

    market_frame: MarketFrame
    branch: Branch
    as_of_time: str
    node_limit: int


@dataclass(frozen=True)
class GNNScoreRequest:
    """Portfolio scoring request passed to the graph/GNN layer."""

    portfolio: SubgraphPortfolio
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class StopConditionReport:
    """Why the current portfolio expansion should continue or stop."""

    should_stop: bool
    reasons: tuple[str, ...] = ()
    budget_exhausted_branch_ids: tuple[str, ...] = ()
    unresolved_risks: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentLoopState:
    """Mutable-run snapshot represented immutably for audit logging."""

    portfolio: SubgraphPortfolio
    iteration: int = 0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    stress_tests: tuple[RequirementStressTest, ...] = ()
    stop_report: StopConditionReport | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentLoopResult:
    """Final output of a bounded agentic expansion run."""

    final_state: AgentLoopState
    expanded_portfolio: SubgraphPortfolio
    audit_log: tuple[AgentLoopState, ...] = ()


class WebSearchTool(Protocol):
    """Tool contract for PIT-safe web evidence retrieval."""

    def __call__(self, request: WebSearchRequest) -> tuple[EvidenceRef, ...]:
        """Return evidence refs only; source bodies live outside the contract."""


class GraphQueryTool(Protocol):
    """Tool contract for graph/warehouse branch expansion."""

    def __call__(self, request: GraphQueryRequest) -> tuple[EvidenceRef, ...]:
        """Return PIT-compatible graph/warehouse evidence refs."""


class GNNScoreTool(Protocol):
    """Tool contract for scoring a portfolio after each expansion round."""

    def __call__(self, request: GNNScoreRequest) -> RequirementStressTest:
        """Return bounded stress-test diagnostics, not a terminal label."""


@dataclass(frozen=True)
class AgentLoopTools:
    """Concrete tools injected into ``run_agent_loop`` by the caller."""

    web_search: WebSearchTool
    graph_query: GraphQueryTool
    gnn_score: GNNScoreTool


def propose_tool_requests(state: AgentLoopState) -> tuple[WebSearchRequest | GraphQueryRequest, ...]:
    """Plan the next PIT-safe expansion requests for a portfolio state.

    TODO(agent-loop): Have the LLM inspect branch roles/directions, missingness,
    and prior stress-test diagnostics, then emit bounded web/graph requests under
    each branch's ``expansion_budget``.
    """

    raise NotImplementedError("agent-loop request planning is not implemented yet")


def apply_evidence_refs(state: AgentLoopState, evidence_refs: tuple[EvidenceRef, ...]) -> AgentLoopState:
    """Attach newly discovered evidence refs to the run state.

    TODO(agent-loop): materialize candidate portfolio elements/prerequisites from
    PIT-backed refs while preserving the resolution-label separation contract.
    """

    return AgentLoopState(
        portfolio=state.portfolio,
        iteration=state.iteration,
        evidence_refs=state.evidence_refs + evidence_refs,
        stress_tests=state.stress_tests,
        stop_report=state.stop_report,
        notes=state.notes,
    )


def evaluate_stop_conditions(state: AgentLoopState) -> StopConditionReport:
    """Evaluate branch budgets, missingness, disagreement, and policy stop rules.

    TODO(agent-loop): enforce max iterations, expansion budgets, policy-specific
    stop rules, and explicit logging for unresolved blind spots.
    """

    return StopConditionReport(
        should_stop=True,
        reasons=("stub: no expansion policy implemented",),
    )


def run_agent_loop(
    portfolio: SubgraphPortfolio,
    tools: AgentLoopTools,
    *,
    max_iterations: int = 4,
) -> AgentLoopResult:
    """Run bounded agentic expansion over a seed ``SubgraphPortfolio``.

    The implementation will eventually:
    1. plan PIT-safe ``web_search``/``graph_query`` calls from branch structure;
    2. attach evidence-backed elements/prerequisites;
    3. call ``gnn_score`` for stress diagnostics;
    4. stop when budgets/policy/missingness conditions are satisfied.

    This stub defines the integration seam without pretending the loop exists.
    """

    _ = tools
    _ = max_iterations
    initial = AgentLoopState(portfolio=portfolio)
    stop_report = evaluate_stop_conditions(initial)
    final = AgentLoopState(
        portfolio=portfolio,
        iteration=initial.iteration,
        evidence_refs=initial.evidence_refs,
        stress_tests=initial.stress_tests,
        stop_report=stop_report,
        notes=initial.notes,
    )
    return AgentLoopResult(final_state=final, expanded_portfolio=portfolio, audit_log=(initial, final))


__all__ = [
    "AgentLoopResult",
    "AgentLoopState",
    "AgentLoopTools",
    "GNNScoreRequest",
    "GNNScoreTool",
    "GraphQueryRequest",
    "GraphQueryTool",
    "StopConditionReport",
    "WebSearchRequest",
    "WebSearchTool",
    "apply_evidence_refs",
    "evaluate_stop_conditions",
    "propose_tool_requests",
    "run_agent_loop",
]
