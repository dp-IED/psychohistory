from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from harness.agent_loop import (
    AgentLoopResult,
    AgentToolset,
    ConstructionPolicy,
    GraphQueryResult,
    MarketContext,
    run_agent_loop,
)
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore, NullMemoryStore


def _stub_tools() -> AgentToolset:
    def web_search(query: str, as_of_date: date) -> list[ToolCallRecord]:
        _ = as_of_date
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=query,
                as_of_time="2026-05-13T00:00:00Z",
                evidence_count=2,
                notes="stub web retrieval",
            )
        ]

    def graph_query(question: str, cutoff_date: date) -> GraphQueryResult:
        _ = (question, cutoff_date)
        return GraphQueryResult(node_count=12, notes="stub graph slice")

    def gnn_score(step: int, evidence_count: int, node_count: int) -> float:
        _ = (evidence_count, node_count)
        return 0.45 + (0.01 * step)

    def analogues(question: str) -> list[str]:
        return [f"analogue for {question}"]

    def market_context(question: str, cutoff_date: date, resolution_date: date) -> MarketContext:
        _ = question
        return MarketContext(
            market_id="metaculus-q-123",
            market_family="metaculus_binary",
            key_actors=["Actor A", "Actor B"],
            region="Global",
            cutoff_date=cutoff_date,
            resolution_date=resolution_date,
        )

    return AgentToolset(
        web_search=web_search,
        graph_query=graph_query,
        gnn_score=gnn_score,
        analogues=analogues,
        market_context=market_context,
    )


def test_agent_loop_requires_non_empty_reasoning_summary() -> None:
    with pytest.raises(ValueError, match="reasoning_summary"):
        AgentLoopResult(
            job_id="job-1",
            final_p_yes=0.5,
            confidence_interval=(0.4, 0.6),
            reasoning_summary="   ",
            blind_spot_checks_fired=[],
            blind_spot_checks_skipped=[],
            gnn_score_trajectory=[0.5],
            tool_call_count=1,
        )


def test_run_agent_loop_writes_episode_and_returns_reasoning(tmp_path) -> None:
    store = JsonlMemoryStore(tmp_path / "memory")
    policy = ConstructionPolicy(
        blind_spot_checks=["coalition_stability_check", "sanctions_escalation_check"],
        max_steps=3,
        convergence_epsilon=0.02,
    )

    result = run_agent_loop(
        question="Will country X sign a ceasefire by July 1, 2026?",
        cutoff_date=date(2026, 5, 20),
        resolution_date=date(2026, 7, 1),
        policy=policy,
        memory=store,
        tools=_stub_tools(),
    )

    assert result.final_p_yes >= 0.0 and result.final_p_yes <= 1.0
    assert result.reasoning_summary.strip() != ""
    assert result.tool_call_count > 0

    episodes = store.read_recent_episodes("metaculus_binary", 5)
    assert len(episodes) == 1
    assert episodes[0].job_id == result.job_id
    assert episodes[0].question.startswith("Will country X")


def test_run_agent_loop_with_null_memory_store_still_returns_result() -> None:
    result = run_agent_loop(
        question="Will inflation be below 3% by Q4 2026?",
        cutoff_date=date(2026, 5, 20),
        resolution_date=date(2026, 12, 31),
        policy=ConstructionPolicy(blind_spot_checks=["treaty_compliance_check"]),
        memory=NullMemoryStore(),
        tools=_stub_tools(),
    )

    assert result.reasoning_summary
    assert isinstance(result.gnn_score_trajectory, list)
    assert result.tool_call_count >= len(result.gnn_score_trajectory)
