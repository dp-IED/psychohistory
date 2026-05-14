from __future__ import annotations

from datetime import date

import pytest

from harness.agent_loop import AgentToolset, ConstructionPolicy, GraphQueryResult, MarketContext, run_agent_loop
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore
from harness.query_mapper import WebSearchRequest
from harness.resolution import AlreadyResolvedError, EpisodeNotFoundError, resolve_market


def _toolset() -> AgentToolset:
    def web_search(req: WebSearchRequest) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=req.query,
                as_of_time=f"{req.as_of_date.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="stub",
            )
        ]

    def graph_query(_question: str, _cutoff: date) -> GraphQueryResult:
        return GraphQueryResult(node_count=10, notes="stub")

    def gnn_score(step: int, evidence: int, nodes: int) -> float:
        if evidence >= 2 and nodes <= 9:
            return 0.8
        return 0.6 + min(0.04, step * 0.01)

    def analogues(_question: str) -> list[str]:
        return []

    def market_context(_question: str, cutoff: date, resolution: date) -> MarketContext:
        return MarketContext(
            market_id="market-1",
            market_family="metaculus_binary",
            key_actors=["actor"],
            region="global",
            cutoff_date=cutoff,
            resolution_date=resolution,
        )

    return AgentToolset(
        web_search=web_search,
        graph_query=graph_query,
        gnn_score=gnn_score,
        analogues=analogues,
        market_context=market_context,
    )


def test_resolve_market_integration_updates_episode_and_guards_double_resolution(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")
    tools = _toolset()

    policy = ConstructionPolicy(blind_spot_checks=["coalition_stability_check"], max_steps=2)
    loop_result = run_agent_loop(
        "Will a ceasefire be announced before June 1?",
        cutoff_date=date(2026, 5, 20),
        resolution_date=date(2026, 6, 1),
        policy=policy,
        memory=memory,
        tools=tools,
    )

    update = resolve_market(job_id=loop_result.job_id, outcome=True, memory=memory, tools=tools)

    assert update.job_id == loop_result.job_id
    assert update.market_id == "market-1"
    assert 0.0 <= update.brier_score <= 1.0
    assert isinstance(update.misses, list)

    updated_episode = memory.read_episode_by_id(loop_result.job_id)
    assert updated_episode is not None
    assert updated_episode.brier_score == update.brier_score
    assert updated_episode.misses == update.misses

    with pytest.raises(AlreadyResolvedError):
        resolve_market(job_id=loop_result.job_id, outcome=True, memory=memory, tools=tools)


def test_resolve_market_unknown_job_id_raises(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")

    with pytest.raises(EpisodeNotFoundError):
        resolve_market(job_id="job-missing", outcome=False, memory=memory, tools=_toolset())
