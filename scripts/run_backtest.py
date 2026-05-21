#!/usr/bin/env python3
"""Backtest runner — loads questions, runs forecasts concurrently, writes to vault.

Supports --skip-existing to avoid re-forecasting questions already in the vault.
Use --concurrency N to run N questions in parallel (default 1).
Shows live Rich progress bars per question slot.

Usage:
    python -m scripts.run_backtest --source polymarket --max-questions 10
    python -m scripts.run_backtest --source polymarket --max-questions 10 --orchestrate
    python -m scripts.run_backtest --source polymarket --max-questions 10 --allow-categories politics,economics
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR, DEFAULT_POLICY_PATH
from harness.corpus.backtest_corpus import BacktestQuestion, build_polymarket_corpus
from harness.orchestrator import run_structured, run_orchestrated
from harness.runs import runs_count, mean_brier, read_all_runs

# ── Library functions (also importable) ────────────────────────────────────

from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
GRAPH_VAULT = ROOT / "graph-vault"


@dataclass(frozen=True)
class BacktestResult:
    question_id: str
    p_yes: float
    brier_score: float | None


def run_single_backtest(
    question: BacktestQuestion,
) -> BacktestResult:
    """Run a single backtest question through the cognitive pipeline.

    Uses the same pipeline as Metaculus: outside-view anchor →
    3-path reasoning → Delphi → premortem → aggregate.
    """
    from harness.orchestrator_v2 import run_cognitive_pipeline
    from harness.outside_view import OutputType

    question_text = getattr(question, "question_text", "")
    cutoff_dt = getattr(question, "open_date", None) or date.today()
    qid = getattr(question, "question_id", "unknown")

    try:
        result = run_cognitive_pipeline(
            question_text=question_text,
            cutoff=cutoff_dt,
            vault_dir=GRAPH_VAULT,
            output_type=OutputType.BINARY,
            question_id=qid,
            source="polymarket-backtest",
            enforce_pit=True,
            query_polymarket=False,
        )
        p_yes = result.p_yes or 0.5
    except Exception:
        p_yes = 0.5

    brier = None
    outcome = getattr(question, "resolution", None)
    if outcome is not None:
        target = 1.0 if outcome else 0.0
        brier = (p_yes - target) ** 2

    return BacktestResult(
        question_id=qid,
        p_yes=p_yes,
        brier_score=brier,
    )


def run_backtest_batch(
    questions: list[BacktestQuestion],
) -> list[BacktestResult]:
    """Run a batch of backtest questions through the cognitive pipeline."""
    return [run_single_backtest(q) for q in questions]


def _rollup_summary(results: list[BacktestResult]) -> dict[str, Any]:
    resolved = [r for r in results if r.brier_score is not None]
    return {
        "total": len(results),
        "resolved": len(resolved),
        "mean_brier": (
            sum(r.brier_score for r in resolved) / len(resolved)
            if resolved else None
        ),
    }


# ── CLI helpers ────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Normalise question text for dedup comparison."""
    return (
        text.lower().strip().rstrip("?.")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _load_existing_ids_and_texts(vault_dir: str) -> tuple[set[str], set[str]]:
    """Load question IDs and texts from existing vault runs.

    Handles both old v1 runs (field: ``question``) and v2 runs
    (field: ``question_id`` + ``_body``).
    Returns (set of IDs, set of full normalised texts).
    """
    ids: set[str] = set()
    texts: set[str] = set()
    for r in read_all_runs(vault_dir):
        qid = r.get("question_id", "")
        if qid and str(qid).strip():
            ids.add(str(qid).strip())
        q_text = r.get("question", "")
        if q_text and isinstance(q_text, str) and q_text.strip():
            texts.add(_normalise(q_text.strip()))
        body = r.get("_body", "") or ""
        body = body.strip()
        if body and not body.startswith("#"):
            texts.add(_normalise(body))
    return ids, texts


def _is_duplicate(
    question_text: str,
    existing_ids: set[str],
    existing_texts: set[str],
    question_id: str = "",
) -> bool:
    """Check if a question has already been forecasted."""
    if question_id and question_id in existing_ids:
        return True
    qn = _normalise(question_text)
    if qn in existing_texts:
        return True
    for et in existing_texts:
        if qn.startswith(et) or et.startswith(qn):
            return True
        if qn in et or et in qn:
            return True
    return False


def _truncate(text: str, max_len: int = 55) -> str:
    return text[: max_len - 1] + "\u2026" if len(text) > max_len else text


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run forecasting backtest (parallel)."
    )
    parser.add_argument(
        "--source", choices=["polymarket"], default="polymarket"
    )
    parser.add_argument(
        "--max-questions", type=int, default=10
    )
    parser.add_argument(
        "--vault",
        default=str(VAULT_DIR),
        help="Vault directory (writes to graph-vault/runs/)",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="Forecast rules file in graph-vault",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions already forecasted in the vault",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of questions to forecast in parallel (default 1)",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Use multi-agent orchestration (sub-agents via delegate_task)",
    )
    parser.add_argument(
        "--allow-categories",
        default="",
        help="Comma-separated categories to include (e.g. politics,economics,crypto). "
             "Omit to include all categories.",
    )
    args = parser.parse_args()

    # Parse category filter
    allowed_categories: frozenset[str] | None = None
    if args.allow_categories:
        allowed_categories = frozenset(
            c.strip().lower() for c in args.allow_categories.split(",") if c.strip()
        )

    # Load existing vault questions if skipping
    existing_ids: set[str] = set()
    existing_texts: set[str] = set()
    if args.skip_existing:
        existing_ids, existing_texts = _load_existing_ids_and_texts(args.vault)
        print(
            f"Vault has {runs_count(args.vault)} runs "
            f"({len(existing_ids)} unique IDs, {len(existing_texts)} unique texts)."
        )

    # Load corpus
    corpus = build_polymarket_corpus(
        min_date=date.today() - timedelta(days=180),
        max_questions=(args.max_questions + 10) * 3,  # over-fetch for dedup
        allowed_categories=allowed_categories,
    )

    # Filter out existing questions
    if args.skip_existing:
        before = len(corpus)
        filtered = [
            q
            for q in corpus
            if not _is_duplicate(
                q.question_text, existing_ids, existing_texts, str(q.question_id)
            )
        ]
        corpus = filtered[: args.max_questions]
        print(
            f"Corpus: {before} total, {len(corpus)} new after dedup "
            f"(need {args.max_questions})."
        )
    else:
        corpus = corpus[: args.max_questions]

    if not corpus:
        print("No questions to forecast (all filtered out as existing).")
        return 0

    print(
        f"Forecasting {len(corpus)} {args.source} questions "
        f"(concurrency={args.concurrency})."
    )

    before = runs_count(args.vault)
    total = len(corpus)

    # Build live display layout
    results: dict[int, dict] = {}
    successes = 0
    failures = 0

    def make_table():
        from rich.table import Table
        from rich.text import Text
        from rich import box

        t = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            title=f"Forecasting {total} questions (concurrency={args.concurrency})",
            title_style="bold",
        )
        t.add_column("#", justify="right", width=2)
        t.add_column("Question", width=56)
        t.add_column("Status", width=10)
        t.add_column("p_yes", justify="right", width=7)
        t.add_column("Brier", justify="right", width=7)
        t.add_column("Elapsed", width=8)

        for i in range(total):
            if i in results:
                r = results[i]
                if r.get("err"):
                    t.add_row(
                        str(i + 1),
                        _truncate(r.get("text", ""), 55),
                        Text("FAIL", style="red bold"),
                        "---",
                        "---",
                        r.get("elapsed", ""),
                    )
                elif r.get("done"):
                    brier = r.get("brier", "")
                    brier_style = (
                        "green"
                        if brier != "---"
                        and (isinstance(brier, float) and brier < 0.05)
                        else "yellow"
                        if brier != "---"
                        and (isinstance(brier, float) and brier < 0.20)
                        else "red"
                    )
                    t.add_row(
                        str(i + 1),
                        _truncate(r.get("text", ""), 55),
                        Text("DONE", style="green bold"),
                        (
                            f'{r.get("p_yes", "---"):.3f}'
                            if isinstance(r.get("p_yes"), float)
                            else "---"
                        ),
                        (
                            Text(f"{brier:.4f}", style=brier_style)
                            if isinstance(brier, float)
                            else "---"
                        ),
                        r.get("elapsed", ""),
                    )
                else:
                    t.add_row(
                        str(i + 1),
                        _truncate(r.get("text", ""), 55),
                        Text("RUNNING", style="yellow"),
                        "---",
                        "---",
                        r.get("elapsed", ""),
                    )
            else:
                t.add_row(
                    str(i + 1),
                    "[dim](queued)",
                    Text("QUEUED", style="dim"),
                    "---",
                    "---",
                    "",
                )
        return t

    # Detect if we have a real TTY for Rich live display
    _HAS_TTY = sys.stdout.isatty()

    # Parallel execution with live display
    live_ctx = None
    if _HAS_TTY:
        try:
            from rich.live import Live

            live_ctx = Live(auto_refresh=True, refresh_per_second=4, screen=True)
            live_ctx.__enter__()
        except Exception:
            live_ctx = None

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            fut_to_idx = {
                pool.submit(
                    _forecast_one,
                    q,
                    args.vault,
                    args.source,
                    args.orchestrate,
                ): i
                for i, q in enumerate(corpus)
            }

            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                text, p_yes, err, brier_val, elapsed = fut.result()
                results[idx] = {
                    "text": text,
                    "p_yes": p_yes,
                    "brier": brier_val,
                    "err": err,
                    "elapsed": elapsed,
                    "done": err is None,
                }
                if err:
                    failures += 1
                    msg = f"  {_truncate(text, 55)} -- {err}"
                else:
                    successes += 1
                    brier_str = (
                        f" brier={brier_val:.4f}"
                        if isinstance(brier_val, float)
                        else ""
                    )
                    msg = f"  {_truncate(text, 55)} p_yes={p_yes:.3f}{brier_str}"
                if not _HAS_TTY:
                    print(msg, flush=True)

                if _HAS_TTY and live_ctx is not None:
                    live_ctx.update(make_table())
    finally:
        if _HAS_TTY and live_ctx is not None:
            live_ctx.__exit__(None, None, None)

    # Final summary
    after = runs_count(args.vault)
    mb = mean_brier(args.vault)
    print(f"\nDone: {successes}/{total} completed, {failures} failed.")
    print(f"Runs in vault: {before} -> {after}")
    if mb is not None:
        print(f"Mean Brier: {mb:.4f}")

    return 0 if failures == 0 else 1


def _forecast_one(
    q: BacktestQuestion,
    vault_dir: str,
    source: str,
    orchestrate: bool = False,
) -> tuple[str, float | None, str | None, float | str, str]:
    """Run a single forecast. Returns (text, p_yes, error, brier, elapsed_str)."""
    t0 = time.time()
    try:
        if orchestrate:
            p_yes, reasoning, _metadata = run_orchestrated(
                q.question_text,
                cutoff=q.open_date,
                vault_dir=vault_dir,
                question_id=str(q.question_id),
                source=source,
                category=getattr(q, "category", "general"),
                resolution=q.resolution,
                volume=getattr(q, "volume", None),
            )
        else:
            p_yes, reasoning, _metadata = run_structured(
                q.question_text,
                cutoff=q.open_date,
                vault_dir=vault_dir,
                question_id=str(q.question_id),
                source=source,
                category=getattr(q, "category", "general"),
                resolution=q.resolution,
                volume=getattr(q, "volume", None),
            )
        elapsed = time.time() - t0
        brier = (
            (p_yes - (1.0 if q.resolution else 0.0)) ** 2
            if q.resolution is not None
            else None
        )
        elapsed_str = f"{elapsed:.0f}s"
        return (q.question_text[:60], p_yes, None, brier, elapsed_str)
    except Exception as e:
        elapsed = time.time() - t0
        return (q.question_text[:60], None, str(e), "---", f"{elapsed:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
