#!/usr/bin/env python3
"""CLI for the PIT research librarian sub-agent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR
from harness.pit_research import pit_search, run_pit_research


def main() -> int:
    parser = argparse.ArgumentParser(description="PIT research librarian.")
    parser.add_argument("question", help="Forecast question (for relevance ranking)")
    parser.add_argument("--cutoff", required=True, help="ISO date YYYY-MM-DD")
    parser.add_argument("--vault", default=str(VAULT_DIR))
    parser.add_argument(
        "--mode",
        choices=["search", "librarian"],
        default="librarian",
        help="search=deterministic excerpts; librarian=Hermes sub-agent brief",
    )
    parser.add_argument("--market-yes", type=float, default=None)
    args = parser.parse_args()

    cutoff = date.fromisoformat(args.cutoff[:10])
    vault = Path(args.vault)

    if args.mode == "search":
        resp = pit_search(args.question, cutoff, vault_dir=vault)
        print(json.dumps(
            {"cutoff": resp.cutoff.isoformat(), "manifest_count": resp.manifest_count,
             "error": resp.error, "results": [{"path": r.path, "excerpt": r.excerpt[:500]} for r in resp.results]},
            indent=2,
        ))
        return 0

    brief, _tmp = run_pit_research(
        args.question, cutoff, vault_dir=vault, market_yes_at_cutoff=args.market_yes,
    )
    print(brief.to_prompt_block())
    print("\n--- JSON ---\n", json.dumps(brief.raw_json or {}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
