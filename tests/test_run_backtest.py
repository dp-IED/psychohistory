from __future__ import annotations

from datetime import date

import pytest

from harness.agent_loop import AgentToolset, ConstructionPolicy, GraphQueryResult, MarketContext
from harness.corpus.backtest_corpus import BacktestQuestion
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore
from harness.resolution import BrierUpdateResult
from scripts.run_backtest import _market_family_for_episode, _rollup_summary, _toolset_with_question_family, run_backtest_batch, run_single_backtest


def _constant_probability_toolset(constant: float) -> AgentToolset:
    def web_search(query: str, as_of_date: date) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=query,
                as_of_time=f"{as_of_date.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="test-backtest",
            )
        ]

    def graph_query(_question: str, _cutoff: date) -> GraphQueryResult:
        return GraphQueryResult(node_count=6, notes="test-backtest")

    def gnn_score(_step: int, _evidence: int, _nodes: int) -> float:
        return constant

    def analogues(_question: str) -> list[str]:
        return []

    def market_context(_question: str, cutoff: date, resolution: date) -> MarketContext:
        return MarketContext(
            market_id="unit-test-market",
            market_family="unit_test",
            key_actors=[],
            region=None,
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


def _sample_question(suffix: str, price: float | None, resolution: bool) -> BacktestQuestion:
    return BacktestQuestion(
        question_id=f"q-{suffix}",
        source="polymarket",
        question_text=f"Unit test market {suffix}",
        open_date=date(2024, 1, 1),
        close_date=date(2024, 6, 1),
        resolution=resolution,
        market_price_at_open=price,
        category=None,
    )


def test_single_backtest_writes_episode(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")
    tools = _constant_probability_toolset(0.62)
    policy = ConstructionPolicy(blind_spot_checks=[], max_steps=3)
    question = _sample_question("writes", price=0.5, resolution=True)

    resolved = run_single_backtest(question, memory, policy, tools)

    stored = memory.read_episode_by_id(resolved.job_id)
    assert stored is not None
    assert stored.question == question.question_text


def test_single_backtest_brier_set(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")
    tools = _constant_probability_toolset(0.71)
    policy = ConstructionPolicy(blind_spot_checks=[], max_steps=3)
    question = _sample_question("brier", price=None, resolution=False)

    resolved = run_single_backtest(question, memory, policy, tools)

    assert resolved.brier_score is not None
    replay = memory.read_episode_by_id(resolved.job_id)
    assert replay is not None
    assert replay.brier_score == pytest.approx(resolved.brier_score)


@pytest.mark.asyncio
async def test_batch_summary_agent_edge(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")
    constant_prediction = 0.62
    tools = _constant_probability_toolset(constant_prediction)
    policy = ConstructionPolicy(blind_spot_checks=[], max_steps=3)

    corpus = [
        _sample_question("alpha", price=0.55, resolution=True),
        _sample_question("beta", price=0.72, resolution=True),
        _sample_question("gamma", price=None, resolution=True),
    ]

    summary = await run_backtest_batch(corpus, memory, policy, tools, concurrency=2)

    assert summary.total == len(corpus)
    assert summary.completed == len(corpus)
    assert summary.failed == 0

    agent_scores_all = [(constant_prediction - float(item.resolution)) ** 2 for item in corpus]
    mean_agent_total = sum(agent_scores_all) / len(agent_scores_all)

    priced = [question for question in corpus if question.market_price_at_open is not None]

    baseline_priced = [
        (float(question.market_price_at_open) - float(question.resolution)) ** 2 for question in priced
    ]
    baseline_mean_priced = sum(baseline_priced) / len(baseline_priced)

    agent_scores_dual = [(constant_prediction - float(question.resolution)) ** 2 for question in priced]
    agent_mean_dual = sum(agent_scores_dual) / len(agent_scores_dual)
    expected_edge = baseline_mean_priced - agent_mean_dual

    assert summary.mean_brier == pytest.approx(mean_agent_total)
    assert summary.agent_edge == pytest.approx(expected_edge)
    assert summary.mean_market_baseline_brier == pytest.approx(baseline_mean_priced)


@pytest.mark.asyncio
async def test_concurrency_sem_completes(tmp_path) -> None:
    memory = JsonlMemoryStore(tmp_path / "memory")
    tools = _constant_probability_toolset(0.54)
    policy = ConstructionPolicy(blind_spot_checks=[], max_steps=3)

    corpus = [_sample_question(str(index), price=0.5, resolution=index % 2 == 0) for index in range(6)]

    summary = await run_backtest_batch(corpus, memory, policy, tools, concurrency=3)

    assert summary.completed == len(corpus)
    assert summary.failed == 0


def test_agent_edge_excludes_failed_questions() -> None:
    """Failed questions with prices must not leak into baseline or agent_edge."""

    good_priced = _sample_question("good-priced", price=0.55, resolution=True)
    good_unpriced = _sample_question("good-unpriced", price=None, resolution=False)
    bad_priced = _sample_question("bad-priced", price=0.72, resolution=True)

    good_priced_result = BrierUpdateResult(
        job_id="j1", market_id="m1", outcome=True,
        brier_score=0.1444, misses=[], p_yes_at_resolution=0.62,
    )
    good_unpriced_result = BrierUpdateResult(
        job_id="j2", market_id="m2", outcome=False,
        brier_score=0.3844, misses=[], p_yes_at_resolution=0.62,
    )

    summary = _rollup_summary(
        corpus=[good_priced, good_unpriced, bad_priced],
        observations=[good_priced_result, good_unpriced_result, RuntimeError("failed")],
    )

    assert summary.total == 3
    assert summary.completed == 2
    assert summary.failed == 1

    # Baseline must only include good_priced (priced+completed).
    # (0.55 - 1.0)² = 0.2025
    assert summary.mean_market_baseline_brier == pytest.approx(0.2025)

    # agent_edge = baseline(0.2025) - agent_brier(0.1444) = 0.0581
    assert summary.agent_edge == pytest.approx(0.0581)


def test_episode_market_family_flows_from_question_category(tmp_path) -> None:
    """market_family in the written episode must match question.category."""
    memory = JsonlMemoryStore(tmp_path / "memory")
    tools = _constant_probability_toolset(0.55)
    policy = ConstructionPolicy(blind_spot_checks=[], max_steps=2)

    question = _sample_question("crypto-q", price=None, resolution=True)
    question = BacktestQuestion(
        question_id=question.question_id,
        source=question.source,
        question_text=question.question_text,
        open_date=question.open_date,
        close_date=question.close_date,
        resolution=question.resolution,
        market_price_at_open=question.market_price_at_open,
        category="crypto",
    )

    wrapped = _toolset_with_question_family(tools, question)
    resolved = run_single_backtest(question, memory, policy, wrapped)

    episode = memory.read_episode_by_id(resolved.job_id)
    assert episode is not None
    assert episode.market_family == "crypto"


def test_market_family_for_episode_fallback() -> None:
    assert _market_family_for_episode(_sample_question("x", price=None, resolution=True)) == "general"
