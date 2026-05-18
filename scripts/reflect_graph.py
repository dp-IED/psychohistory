#!/usr/bin/env python3
"""Reflection: review recent forecast runs, then curate graph-vault.

The agent reads recent runs from graph-vault/runs/, inspects the vault,
and decides what to create, enrich, merge, or prune.

Use --dry-run to preview without making changes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from harness.config import VAULT_DIR

GRAPH_VAULT = VAULT_DIR
_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200


def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes (profile={_PROFILE}) failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def _count_md(dir_path: Path) -> int:
    return len(list(dir_path.glob("*.md"))) if dir_path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph vault reflection.")
    parser.add_argument("--dry-run", action="store_true", help="Print findings without modifying")
    parser.add_argument("--recent-runs", type=int, default=10, help="Number of recent runs to review")
    args = parser.parse_args()

    # Scan graph vault
    entities = _count_md(GRAPH_VAULT / "entities")
    events = 0  # events/ dir was removed — this count kept at 0 for backward compat
    concepts = 0  # concepts/ dir was removed — kept at 0 for backward compat
    quarters = _count_md(GRAPH_VAULT / "timeline")

    # Read recent runs
    runs_dir = GRAPH_VAULT / "runs"
    recent_runs = sorted(runs_dir.glob("*.md"), reverse=True)[:args.recent_runs]
    non_stub = [r for r in recent_runs if not r.name.startswith("_")]  # skip index files
    run_summaries = []
    for r in non_stub[:args.recent_runs]:
        text = r.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        # Extract frontmatter tags
        frontmatter_end = 0
        p_yes = "?"
        source = "?"
        category = "?"
        brier = "?"
        reasoning_preview = ""
        reading_fm = False
        for i, line in enumerate(lines):
            if line.strip() == "---" and not reading_fm:
                reading_fm = True
                continue
            if line.strip() == "---" and reading_fm:
                reading_fm = False
                frontmatter_end = i
                break
            if reading_fm:
                if line.startswith("p_yes:"):
                    p_yes = line.split(":", 1)[1].strip()
                elif line.startswith("source:"):
                    source = line.split(":", 1)[1].strip().strip("'\"")
                elif line.startswith("category:"):
                    category = line.split(":", 1)[1].strip().strip("'\"")
                elif line.startswith("brier:"):
                    brier = line.split(":", 1)[1].strip()
        # Get first non-empty body line as preview
        body_lines = lines[frontmatter_end + 1:]
        question = body_lines[0].strip() if body_lines else r.stem
        run_summaries.append({
            "file": r.name,
            "question": question,
            "p_yes": p_yes,
            "source": source,
            "category": category,
            "brier": brier,
        })

    print(f"Graph vault: {entities} entities, {events} events, {concepts} concepts, {quarters} quarters")
    print(f"Recent runs: {len(non_stub)} non-index runs (reviewing newest {args.recent_runs})")
    print()

    # Build prompt
    prompt = [
        "=== GRAPH VAULT REFLECTION ===",
        "",
        "Review recent forecast runs and curate the graph vault.",
        "The graph vault is your second brain for forecasting — entities, events,",
        "concepts, and anything else that helps you forecast better.",
        "",
        "=== CURRENT GRAPH VAULT ===",
        f"entities/: {entities} nodes",
        f"events/: {events} nodes",
        f"concepts/: {concepts} nodes",
        f"timeline/: {quarters} quarter nodes",
        "",
        "=== RECENT FORECAST RUNS ===",
    ]
    for r in run_summaries:
        prompt.append(f"  [{r['category']}] {r['question']}")
        prompt.append(f"    p_yes={r['p_yes']}  brier={r['brier']}  source={r['source']}")
    prompt += [
        "",
        "=== YOUR TASK ===",
        "Read recent runs and the current graph vault, then decide what to do.",
        "You own this vault — create, enrich, merge, prune as you see fit.",
        "",
        "Some things you might do:",
        "- Create entity nodes for people, places, orgs that appeared in multiple forecasts",
        "- Create event nodes for signal-rich turning points",
        "- Promote recurring patterns into concept nodes",
        "- Enrich existing nodes with new cross-references",
        "- Prune nodes that aren't earning their keep",
        "- Invent new structures if they'd help",
        "",
        "Use read_file, write_file, patch, and search_files to make changes.",
        "Read the graph-vault/ directory structure first, then the recent runs, then decide.",
        "Be creative. A lean, well-linked vault beats a bloated one.",
        "Report what you changed and why.",
    ]
    prompt_str = "\n".join(prompt)

    if args.dry_run:
        print("[DRY-RUN] Would prompt Hermes:\n")
        print(prompt_str)
    else:
        print("Running Hermes reflection agent...")
        result = _call_hermes(prompt_str)
        print(result)


if __name__ == "__main__":
    _PROFILE = _HERMES_PROFILE
    main()
