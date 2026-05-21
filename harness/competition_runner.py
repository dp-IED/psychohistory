from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from harness.agent_loop import AgentLoopResult
from harness.metaculus_client import MetaculusAPIError, MetaculusClient, MetaculusQuestion
from harness.resolution import BrierUpdateResult

# ── Constants ────────────────────────────────────────────────────────

# Summer 2026 AIB tournament project ID.
# Note: distinct from the Metaculus Cup Summer 2026 (project 33021).
SUMMER_2026_AIB_PROJECT_ID = 33022


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuestionResult:
    question_id: int
    resolution: BrierUpdateResult | None = None


# ── Core runner ──────────────────────────────────────────────────────

RunLoopFn = Callable[..., AgentLoopResult]
ResolveFn = Callable[..., BrierUpdateResult]


def run_one_question(
    *,
    client: MetaculusClient,
    run_loop: RunLoopFn,
    question_id: int | None = None,
    resolve: bool = False,
    memory: object | None = None,
    tools: object | None = None,
    resolver: ResolveFn | None = None,
) -> QuestionResult:
    """Fetch open questions from the AIB project, pick one, forecast it.

    If question_id is given, target that specific question.
    Otherwise pick the first open question from the default project.
    """
    questions = client.get_open_questions(SUMMER_2026_AIB_PROJECT_ID)

    if not questions:
        raise RuntimeError("No open questions found in AIB project")

    if question_id is not None:
        matching = [q for q in questions if q.question_id == question_id]
        if not matching:
            raise ValueError(
                f"Question ID {question_id} not found in open questions"
            )
        target = matching[0]
    else:
        target = questions[0]

    result = run_loop(
        target.title,
        target.close_date,
        target.resolution_date,
        None,  # policy
        memory,
        tools,
    )

    client.post_forecast(
        question_id=target.question_id,
        p_yes=result.final_p_yes,
        comment=result.reasoning_summary,
    )

    out = QuestionResult(question_id=target.question_id)

    if resolve and resolver is not None:
        out = QuestionResult(
            question_id=target.question_id,
            resolution=try_resolve_question(
                client=client,
                question_id=target.question_id,
                job_id=result.job_id,
                memory=memory,
                tools=tools,
                resolver=resolver,
            ),
        )

    return out


def run_batch(
    *,
    client: MetaculusClient,
    run_loop: RunLoopFn,
    batch_size: int,
) -> list[QuestionResult]:
    """Forecast up to batch_size open questions from the AIB project."""
    questions = client.get_open_questions(SUMMER_2026_AIB_PROJECT_ID)
    results: list[QuestionResult] = []

    for q in questions[:batch_size]:
        result = run_one_question(
            client=client,
            run_loop=run_loop,
            question_id=q.question_id,
        )
        results.append(result)

    return results


def try_resolve_question(
    *,
    client: MetaculusClient,
    question_id: int,
    job_id: str,
    memory: object,
    tools: object | None,
    resolver: ResolveFn,
) -> BrierUpdateResult | None:
    """Check if a question has resolved and, if so, run the resolver."""
    outcome = client.get_resolution(question_id)
    if outcome is None:
        return None
    return resolver(
        job_id=job_id,
        outcome=outcome,
        memory=memory,
        tools=tools,
    )


# ── CLI ──────────────────────────────────────────────────────────────


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[str], MetaculusClient] | None = None,
    run_loop: RunLoopFn | None = None,
) -> int:
    """CLI entry point for competition runner.

    Usage:
        python -m harness.competition_runner --question-id 12345
        python -m harness.competition_runner --batch 5
        python -m harness.competition_runner --resolve --question-id 12345

    Returns exit code: 0=success, 1=API error, 2=usage error
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]

    # Parse flags
    question_id: int | None = None
    batch_size: int | None = None
    resolve_flag = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--question-id":
            i += 1
            if i >= len(argv):
                print("Missing value for --question-id", file=sys.stderr)
                return 2
            try:
                question_id = int(argv[i])
            except ValueError:
                print(f"Invalid question-id: {argv[i]}", file=sys.stderr)
                return 2
            if question_id <= 0:
                print("question-id must be positive", file=sys.stderr)
                return 2
        elif arg == "--batch":
            i += 1
            if i >= len(argv):
                print("Missing value for --batch", file=sys.stderr)
                return 2
            try:
                batch_size = int(argv[i])
            except ValueError:
                print(f"Invalid batch size: {argv[i]}", file=sys.stderr)
                return 2
        elif arg == "--resolve":
            resolve_flag = True
        i += 1

    # Usage validation
    if resolve_flag and question_id is None:
        print("--resolve requires --question-id", file=sys.stderr)
        return 2

    token = os.environ.get("METACULUS_API_TOKEN")
    if not token:
        print("METACULUS_API_TOKEN environment variable not set", file=sys.stderr)
        return 2

    factory = client_factory or MetaculusClient
    client = factory(token)

    try:
        if question_id is not None and not resolve_flag:
            # Single question forecast
            run_one_question(
                client=client,
                run_loop=run_loop or _stub_run_loop,
                question_id=question_id,
            )
        elif batch_size is not None:
            run_batch(
                client=client,
                run_loop=run_loop or _stub_run_loop,
                batch_size=batch_size,
            )
        elif resolve_flag and question_id is not None:
            # Resolution-only mode
            outcome = client.get_resolution(question_id)
            if outcome is not None:
                print(f"Question {question_id} resolved: {outcome}")
            else:
                print(f"Question {question_id} not yet resolved")
        return 0
    except MetaculusAPIError as e:
        print(f"API error: {e}", file=sys.stderr)
        return 1


def _stub_run_loop(**kwargs: object) -> AgentLoopResult:
    """Stub run_loop for CLI testing — returns a plausible result."""
    return AgentLoopResult(
        job_id="stub-job",
        final_p_yes=0.55,
        confidence_interval=(0.45, 0.65),
        reasoning_summary="Stub forecast from competition runner CLI.",
    )
