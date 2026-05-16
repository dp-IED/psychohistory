#!/usr/bin/env python3
"""Backtest runner -- loads questions, runs forecasts, writes runs to vault."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.orchestrator import run_structured
from harness.runs import runs_count, mean_brier
from harness.config import load_policy
from harness.corpus import build_polymarket_corpus, build_metaculus_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasting backtest.")
    parser.add_argument("--source", choices=["polymarket", "metaculus"], default="polymarket")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--vault", default="vault", help="Vault directory (writes to vault/runs/)")
    parser.add_argument("--policy", default="vault/policy.md", help="Policy file path")
    args = parser.parse_args()

    # Load policy
    cfg = load_policy(args.policy)
    policy_body = cfg.body

    # Load corpus
    if args.source == "polymarket":
        corpus = build_polymarket_corpus(
            min_date=date.today() - timedelta(days=180),
            min_volume=100.0,
            max_questions=args.max_questions,
        )
    else:
        corpus = build_metaculus_corpus()

    print(f"Loaded {len(corpus)} {args.source} questions.")

    # Before count
    before = runs_count(args.vault)
    successes = 0
    failures = 0

    for q in corpus:
        try:
            p_yes, reasoning = run_structured(
                q.question_text,
                cutoff=q.open_date,
                vault_dir=args.vault,
                policy_body=policy_body,
                question_id=str(q.question_id),
                source=args.source,
                category=getattr(q, "category", "general"),
                resolution=q.resolution,
            )
            successes += 1
            brier_str = f" brier={(p_yes - (1.0 if q.resolution else 0.0)) ** 2:.4f}" if q.resolution is not None else ""
            print(f"  \u2713 {q.question_text[:60]} p_yes={p_yes:.3f}{brier_str}")
        except Exception as e:
            failures += 1
            print(f"  \u2717 {q.question_text[:60]} -- {e}")

    after = runs_count(args.vault)
    total = successes + failures
    mb = mean_brier(args.vault)
    print(f"\nDone: {successes}/{total} completed, {failures} failed.")
    print(f"Runs in vault: {before} -> {after}")
    if mb is not None:
        print(f"Mean Brier: {mb:.4f}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
