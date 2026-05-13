"""Competition runner for Spring 2026 AIB.

Provides deterministic one-shot helpers and a minimal CLI:
- python -m harness.competition_runner --question-id 12345
- python -m harness.competition_runner --batch 10
- python -m harness.competition_runner --resolve --question-id 12345
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from harness.agent_loop import (
    AgentLoopResult,
    AgentToolset,
    ConstructionPolicy,
    GraphQueryResult,
    MarketContext,
    run_agent_loop,
)
from harness.memory_schema import ToolCallRecord
from harness.memory_store import JsonlMemoryStore, MemoryStore
from harness.metaculus_client import MetaculusAPIError, MetaculusClient, MetaculusQuestion
from harness.resolution import AlreadyResolvedError, BrierUpdateResult, resolve_market

SPRING_2026_AIB_PROJECT_ID = 32916


@dataclass(frozen=True)
class RunnerResult:
    question_id: int
    posted_probability: float
    posted_comment: str
    resolution: BrierUpdateResult | None = None


def _default_tools(cutoff_date: date, resolution_date: date) -> AgentToolset:
    def web_search(query: str, as_of_date: date) -> list[ToolCallRecord]:
        _ = as_of_date
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=query,
                as_of_time=f"{cutoff_date.isoformat()}T00:00:00Z",
                evidence_count=1,
                notes="competition-runner stub",
            )
        ]

    def graph_query(_question: str, _cutoff: date) -> GraphQueryResult:
        return GraphQueryResult(node_count=8, notes="competition-runner stub")

    def gnn_score(step: int, _evidence: int, _nodes: int) -> float:
        return min(0.9, 0.52 + 0.01 * step)

    def analogues(_question: str) -> list[str]:
        return []

    def market_context(_question: str, _cutoff: date, _resolution: date) -> MarketContext:
        return MarketContext(
            market_id="metaculus-smoke",
            market_family="metaculus_binary",
            key_actors=[],
            region=None,
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


def _default_run_loop_factory(memory: MemoryStore, tools: AgentToolset) -> Callable[[str, date, date], AgentLoopResult]:
    def _run_loop(question: str, cutoff_date: date, resolution_date: date) -> AgentLoopResult:
        _ = (cutoff_date, resolution_date)
        policy = ConstructionPolicy(blind_spot_checks=[], max_steps=3, convergence_epsilon=0.01)
        return run_agent_loop(question, cutoff_date, resolution_date, policy, memory, tools)

    return _run_loop


def try_resolve_question(
    *,
    client: MetaculusClient,
    question_id: int,
    job_id: str,
    memory: MemoryStore,
    tools: AgentToolset,
    resolver: Callable[..., BrierUpdateResult] = resolve_market,
) -> BrierUpdateResult | None:
    outcome = client.get_resolution(question_id)
    if outcome is None:
        return None

    try:
        return resolver(job_id=job_id, outcome=outcome, memory=memory, tools=tools)
    except AlreadyResolvedError:
        return None


def run_one_question(
    *,
    client: MetaculusClient,
    run_loop: Callable[[str, date, date], AgentLoopResult],
    project_id: int = SPRING_2026_AIB_PROJECT_ID,
    question_id: int | None = None,
    resolve: bool = False,
    memory: MemoryStore | None = None,
    tools: AgentToolset | None = None,
    resolver: Callable[..., BrierUpdateResult] = resolve_market,
) -> RunnerResult:
    questions = client.get_open_questions(project_id=project_id)
    if not questions:
        raise RuntimeError(f"No open questions found for project {project_id}")

    if question_id is None:
        q = questions[0]
    else:
        q = next((item for item in questions if item.question_id == question_id), None)
        if q is None:
            raise ValueError(f"question_id {question_id} not found in open set for project {project_id}")

    result = run_loop(q.title, q.close_date, q.resolution_date)
    client.post_forecast(q.question_id, result.final_p_yes, result.reasoning_summary)

    resolution: BrierUpdateResult | None = None
    if resolve:
        if memory is None or tools is None:
            raise ValueError("memory and tools are required when resolve=True")
        resolution = try_resolve_question(
            client=client,
            question_id=q.question_id,
            job_id=result.job_id,
            memory=memory,
            tools=tools,
            resolver=resolver,
        )

    return RunnerResult(
        question_id=q.question_id,
        posted_probability=result.final_p_yes,
        posted_comment=result.reasoning_summary,
        resolution=resolution,
    )


def run_batch(
    *,
    client: MetaculusClient,
    run_loop: Callable[[str, date, date], AgentLoopResult],
    batch_size: int,
    project_id: int = SPRING_2026_AIB_PROJECT_ID,
) -> list[RunnerResult]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    questions = client.get_open_questions(project_id=project_id)
    if not questions:
        raise RuntimeError(f"No open questions found for project {project_id}")

    out: list[RunnerResult] = []
    for q in questions[:batch_size]:
        result = run_loop(q.title, q.close_date, q.resolution_date)
        client.post_forecast(q.question_id, result.final_p_yes, result.reasoning_summary)
        out.append(
            RunnerResult(
                question_id=q.question_id,
                posted_probability=result.final_p_yes,
                posted_comment=result.reasoning_summary,
            )
        )
    return out


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[str], MetaculusClient] = MetaculusClient,
    run_loop: Callable[[str, date, date], AgentLoopResult] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run one-shot Metaculus AIB forecasts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question-id", type=int, help="Target a specific question id")
    mode.add_argument("--batch", type=int, help="Forecast N open questions sequentially")
    parser.add_argument("--resolve", action="store_true", help="Try resolving posted forecast immediately")
    parser.add_argument("--project-id", type=int, default=SPRING_2026_AIB_PROJECT_ID)
    args = parser.parse_args(argv)

    token = os.environ.get("METACULUS_API_TOKEN", "").strip()
    if not token:
        print("Missing METACULUS_API_TOKEN", file=sys.stderr)
        return 2

    if args.question_id is not None and args.question_id <= 0:
        print("question-id must be > 0", file=sys.stderr)
        return 2
    if args.batch is not None and args.batch <= 0:
        print("batch must be > 0", file=sys.stderr)
        return 2
    if args.resolve and args.question_id is None:
        print("--resolve requires --question-id", file=sys.stderr)
        return 2

    client = client_factory(token)

    # shared runtime deps for default path (episode persistence required for resolve).
    memory = JsonlMemoryStore(Path(os.environ.get("HARNESS_MEMORY_DIR", ".harness_memory")))

    try:
        if args.question_id is not None:
            synthetic_tools = _default_tools(date.today(), date.today())
            active_run_loop = run_loop or _default_run_loop_factory(memory, synthetic_tools)
            result = run_one_question(
                client=client,
                run_loop=active_run_loop,
                project_id=args.project_id,
                question_id=args.question_id,
                resolve=args.resolve,
                memory=memory,
                tools=synthetic_tools,
            )
            print(f"posted question_id={result.question_id} p_yes={result.posted_probability:.4f}")
            if result.resolution is not None:
                print(
                    "resolved "
                    f"job_id={result.resolution.job_id} "
                    f"brier={result.resolution.brier_score:.4f} "
                    f"outcome={result.resolution.outcome}"
                )
        else:
            active_run_loop = run_loop or _default_run_loop_factory(memory, _default_tools(date.today(), date.today()))
            results = run_batch(
                client=client,
                run_loop=active_run_loop,
                batch_size=args.batch,
                project_id=args.project_id,
            )
            print(f"posted {len(results)} forecasts")
        return 0
    except MetaculusAPIError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SPRING_2026_AIB_PROJECT_ID",
    "RunnerResult",
    "run_one_question",
    "run_batch",
    "try_resolve_question",
    "main",
]
