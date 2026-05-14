"""CLI + async orchestration harness for corpus-scale backtests."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from harness.agent_loop import AgentToolset, ConstructionPolicy, GraphQueryResult, MarketContext, run_agent_loop
from harness.calibration import compute_calibration
from harness.corpus.backtest_corpus import BacktestQuestion, build_manifold_corpus, build_polymarket_corpus
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore, MemoryStore
from harness.resolution import BrierUpdateResult, resolve_market


def _market_family_for_episode(question: BacktestQuestion) -> str:
    cat_cell = question.category
    if isinstance(cat_cell, str):
        trimmed = cat_cell.strip().lower()
        if trimmed:
            return trimmed
    return "general"


def _toolset_with_question_family(base: AgentToolset, question: BacktestQuestion) -> AgentToolset:
    episode_family = _market_family_for_episode(question)
    inner_context = base.market_context

    def wrapped_market_context(question_text: str, cutoff: date, resolution_date: date) -> MarketContext:
        prior_cell = inner_context(question_text, cutoff, resolution_date)
        return MarketContext(
            market_id=prior_cell.market_id,
            market_family=episode_family,
            key_actors=prior_cell.key_actors,
            region=prior_cell.region,
            cutoff_date=prior_cell.cutoff_date,
            resolution_date=prior_cell.resolution_date,
        )

    return AgentToolset(
        web_search=base.web_search,
        graph_query=base.graph_query,
        gnn_score=base.gnn_score,
        analogues=base.analogues,
        market_context=wrapped_market_context,
    )


@dataclass
class BacktestSummary:
    total: int
    completed: int
    failed: int
    mean_brier: float
    mean_market_baseline_brier: float | None
    agent_edge: float | None
    by_source: dict[str, float]


def build_stub_toolset() -> AgentToolset:
    def web_search(query: str, as_of_date: date) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=query,
                as_of_time=f"{as_of_date.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="backtest-stub-search",
            )
        ]

    def graph_query(question: str, cutoff: date) -> GraphQueryResult:
        _ = (question, cutoff)
        return GraphQueryResult(node_count=8, notes="backtest-stub-graph")

    def gnn_score(step: int, evidence_count: int, nodes: int) -> float:
        _ = nodes
        return min(0.95, 0.52 + 0.015 * step + 0.0005 * evidence_count)

    def analogues(question_text: str) -> list[str]:
        _ = question_text
        return []

    def market_context(question_text: str, cutoff: date, resolution: date) -> MarketContext:
        return MarketContext(
            market_id="stub-market-id",
            market_family="stub_family",
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


def run_single_backtest(
    question: BacktestQuestion,
    memory: MemoryStore,
    policy: ConstructionPolicy,
    tools: AgentToolset,
) -> BrierUpdateResult:
    episode_outcome = run_agent_loop(
        question.question_text,
        cutoff_date=question.open_date,
        resolution_date=question.close_date,
        policy=policy,
        memory=memory,
        tools=tools,
    )

    resolved = resolve_market(
        job_id=episode_outcome.job_id,
        outcome=question.resolution,
        memory=memory,
        tools=tools,
    )
    return resolved



def _rollup_summary(*, corpus: list[BacktestQuestion], observations: list[object]) -> BacktestSummary:
    if len(observations) != len(corpus):
        raise ValueError("corpus and observation vectors must align")

    successes: list[tuple[BacktestQuestion, BrierUpdateResult]] = []
    failures = 0
    for question, observation in zip(corpus, observations, strict=True):
        if isinstance(observation, BrierUpdateResult):
            successes.append((question, observation))
        else:
            failures += 1

    mean_brier = (
        sum(result.brier_score for _, result in successes) / len(successes) if successes else 0.0
    )

    # Priced AND completed — the correct denominator for both baseline and edge.
    priced_completed = [
        (question, obs.brier_score)
        for question, obs in successes
        if question.market_price_at_open is not None
    ]

    if priced_completed:
        mean_market_baseline_brier: float | None = (
            sum((question.market_price_at_open - float(question.resolution)) ** 2
                for question, _ in priced_completed)
            / len(priced_completed)
        )
        agent_brier_on_priced = (
            sum(score for _, score in priced_completed) / len(priced_completed)
        )
        agent_edge: float | None = mean_market_baseline_brier - agent_brier_on_priced
    else:
        mean_market_baseline_brier = None
        agent_edge = None

    grouped_scores: defaultdict[str, list[float]] = defaultdict(list)
    for question, observation in successes:
        grouped_scores[question.source].append(observation.brier_score)

    by_source_snapshot = {
        bucket: sum(values) / len(values) if values else 0.0 for bucket, values in grouped_scores.items()
    }

    return BacktestSummary(
        total=len(corpus),
        completed=len(successes),
        failed=failures,
        mean_brier=mean_brier,
        mean_market_baseline_brier=mean_market_baseline_brier,
        agent_edge=agent_edge,
        by_source=dict(sorted(by_source_snapshot.items())),
    )


async def run_backtest_batch(
    corpus: list[BacktestQuestion],
    memory: MemoryStore,
    policy: ConstructionPolicy,
    tools: AgentToolset,
    *,
    concurrency: int = 3,
) -> BacktestSummary:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    semaphore = asyncio.Semaphore(concurrency)

    async def process_candidate(candidate: BacktestQuestion) -> BrierUpdateResult:
        async with semaphore:
            scoped_tools = _toolset_with_question_family(tools, candidate)
            return await asyncio.to_thread(run_single_backtest, candidate, memory, policy, scoped_tools)

    observations = await asyncio.gather(
        *[process_candidate(question_payload) for question_payload in corpus],
        return_exceptions=True,
    )

    return _rollup_summary(corpus=corpus, observations=list(observations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch backtest harness over Polymarket or Manifold corpora.")
    parser.add_argument("--source", choices=("polymarket", "manifold"), default="polymarket")
    parser.add_argument("--min-date", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--min-volume", type=float, default=5000.0)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--memory-dir", type=Path, default=Path(".harness_memory"))

    cli_args = parser.parse_args(argv)

    if cli_args.min_volume < 0:
        parser.error("--min-volume must be non-negative")

    memory_store = JsonlMemoryStore(cli_args.memory_dir.expanduser().resolve())

    stub_tools = build_stub_toolset()

    calibration = compute_calibration(memory_store)
    shrinkage = calibration.suggested_shrinkage if not calibration.insufficient_data else None

    planner = ConstructionPolicy(blind_spot_checks=[], max_steps=3, shrinkage=shrinkage)

    if cli_args.source == "polymarket":
        dataset = build_polymarket_corpus(cli_args.min_date, cli_args.min_volume, cli_args.max_questions)
    else:
        dataset = build_manifold_corpus(cli_args.min_date, cli_args.max_questions)

    aggregated = asyncio.run(
        run_backtest_batch(dataset, memory_store, planner, stub_tools, concurrency=cli_args.concurrency),
    )
    print(aggregated)

    if aggregated.agent_edge is not None:
        print(f"agent_edge={aggregated.agent_edge:+.4f}")
    else:
        print("agent_edge=N/A (no priced+completed questions)")

    return 0


__all__ = [
    "BacktestSummary",
    "build_stub_toolset",
    "main",
    "run_backtest_batch",
    "run_single_backtest",
]


if __name__ == "__main__":
    raise SystemExit(main())
