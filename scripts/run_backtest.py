from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from harness.agent_loop import AgentLoopResult, AgentToolset, ConstructionPolicy
from harness.corpus.backtest_corpus import BacktestQuestion
from harness.memory_store import JsonlMemoryStore


@dataclass(frozen=True)
class BacktestResult:
    question_id: str
    p_yes: float
    brier_score: float | None


def _market_family_for_episode(question: BacktestQuestion) -> str:
    """Map a backtest question to a market family for memory storage."""
    return question.source if hasattr(question, "source") else "binary"


def _toolset_with_question_family(family: str) -> AgentToolset:
    """Build an AgentToolset tuned for a specific market family."""
    return AgentToolset(
        web_search=lambda req: [],
        graph_query=lambda q, c: type("GraphQueryResult", (), {"node_count": 0, "notes": ""})(),
        gnn_score=lambda s, e, n: 0.5,
        analogues=lambda q: [],
        market_context=lambda q, c, r: type("MarketContext", (), {
            "market_id": "stub",
            "market_family": family,
            "key_actors": [],
            "region": "global",
            "cutoff_date": c,
            "resolution_date": r,
        })(),
    )


def _rollup_summary(results: list[BacktestResult]) -> dict[str, Any]:
    """Compute aggregate stats over a backtest batch."""
    resolved = [r for r in results if r.brier_score is not None]
    return {
        "total": len(results),
        "resolved": len(resolved),
        "mean_brier": (
            sum(r.brier_score for r in resolved) / len(resolved)
            if resolved
            else None
        ),
    }


def run_single_backtest(
    *,
    question: BacktestQuestion,
    memory: JsonlMemoryStore,
    tools: AgentToolset,
    policy: ConstructionPolicy,
) -> BacktestResult:
    """Run a single backtest question and return the result."""
    result = _stub_run_loop(
        question=question.question_text,
        cutoff_date=question.cutoff_date or date.today(),
        resolution_date=question.resolution_date or date.today(),
        policy=policy,
        memory=memory,
        tools=tools,
    )
    brier = None
    if hasattr(question, "outcome") and question.outcome is not None:  # type: ignore[union-attr]
        target = 1.0 if question.outcome else 0.0  # type: ignore[union-attr]
        brier = (result.final_p_yes - target) ** 2

    return BacktestResult(
        question_id=getattr(question, "question_id", "unknown"),
        p_yes=result.final_p_yes,
        brier_score=brier,
    )


def run_backtest_batch(
    *,
    questions: list[BacktestQuestion],
    memory: JsonlMemoryStore,
    tools: AgentToolset,
    policy: ConstructionPolicy,
) -> list[BacktestResult]:
    """Run a batch of backtest questions."""
    return [
        run_single_backtest(
            question=q,
            memory=memory,
            tools=tools,
            policy=policy,
        )
        for q in questions
    ]


def _stub_run_loop(**kwargs: object) -> AgentLoopResult:
    return AgentLoopResult(
        job_id="backtest-stub",
        final_p_yes=0.5,
        confidence_interval=(0.4, 0.6),
        reasoning_summary="Stub backtest result.",
    )
