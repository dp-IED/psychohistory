#!/usr/bin/env python3
"""PIT summary training harness — parallel batches with reflection.

Processes N quarters at a time in parallel, then runs pit_reflect.py,
then the next N quarters. Each quarter agent writes only its
quarters/ file (no shared entity/spec conflicts during parallel work).
Reflection handles cross-quarter thread/concept/entity creation.
"""

from __future__ import annotations

import argparse
import calendar
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from harness.config import VAULT_DIR

GRAPH_VAULT = VAULT_DIR
_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200


def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _HERMES_PROFILE]
    # Run from graph-vault directory so relative paths (timeline/, etc.) resolve correctly
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(GRAPH_VAULT))
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes (profile={_HERMES_PROFILE}) failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def quarter_date_range(y: int, q: int) -> tuple[date, date]:
    start_month = (q - 1) * 3 + 1
    start = date(y, start_month, 1)
    if q == 4:
        return (start, date(y, 12, 31))
    end_month = start_month + 2
    last_day = calendar.monthrange(y, end_month)[1]
    return (start, date(y, end_month, last_day))


def quarter_label(y: int, q: int) -> str:
    return f"{y}-Q{q}"


def build_quarter_prompt(y: int, q: int) -> str:
    start, end = quarter_date_range(y, q)
    label = quarter_label(y, q)
    vault = str(GRAPH_VAULT)
    return f"""=== PIT SUMMARY: {label} ({start} to {end}) ===

You are at the end of {label}. Research what happened during this period
using ONLY information available up to {end.isoformat()}. No future knowledge.

=== YOUR TOOLS ===

- Read_file/search_files to read prior context from {vault}/quarters/, {vault}/threads/, {vault}/concepts/
- Web search at {end.isoformat()} (PIT-constrained)
- Write_file to create/update files (you can write as many as you like)

=== VAULT STRUCTURE ===

The vault uses a Thread-as-Primary-Node model:

  quarters/   — one file per quarter (e.g. 1922-Q3.md). Write your quarter summary here.
  threads/    — ongoing narratives (e.g. threads/cold-war-nuclear-arms-race.md). Threads span
                multiple quarters and represent causal chains, movements, or long-running dynamics.
  concepts/   — recurring ideas and frameworks (e.g. concepts/balance-of-power.md).

Write at minimum: {vault}/quarters/{label}.md
You may also create or update thread stubs in threads/ and concept stubs in concepts/ if
obvious candidates emerge. Use wikilinks ([[EntityName]]) to link across all three.

=== YOUR TASK ===

Write a PIT-constrained summary for {label}. Make it rich and detailed.
Focus on events, decisions, turning points, and emerging patterns.
When you identify a causal chain or recurring dynamic, note it — reflection will
promote it to a thread file if warranted.

Write your files using absolute paths under {vault}/.
For example: write_file(path="{vault}/quarters/{label}.md", content="...").
Do NOT write to _spec.md, _procedure.md, or any file outside quarters/, threads/, concepts/.

You can read _spec.md and prior quarters/ files for context (relative paths
resolve to {vault}). Schema and cross-quarter entity management is handled by reflection.

=== PIT CONSTRAINT ===

No information from after {end.isoformat()}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT training with parallel batches.")
    parser.add_argument("--start", default="1900-Q1")
    parser.add_argument("--end", default="2025-Q4")
    parser.add_argument("--batch-size", type=int, default=4, help="Quarters per parallel batch (default 4)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sy, sq = args.start.split("-Q")
    ey, eq = args.end.split("-Q")
    start = (int(sy), int(sq))
    end = (int(ey), int(eq))

    all_qs: list[tuple[int, int]] = []
    for y in range(start[0], end[0] + 1):
        q_min = start[1] if y == start[0] else 1
        q_max = end[1] if y == end[0] else 4
        for q in range(q_min, q_max + 1):
            all_qs.append((y, q))

    total = len(all_qs)
    batch_size = min(args.batch_size, total)
    print(f"PIT training: {total} quarters ({args.start} to {args.end}), "
          f"batch size {batch_size}, {(total + batch_size - 1) // batch_size} batches")
    print()

    for batch_idx, batch_start in enumerate(range(0, total, batch_size)):
        batch = all_qs[batch_start:batch_start + batch_size]
        labels = [quarter_label(y, q) for y, q in batch]
        batch_num = batch_idx + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"[Batch {batch_num}/{total_batches}] {labels[0]} to {labels[-1]} "
              f"({len(batch)} quarters)")
        sys.stdout.flush()

        if args.dry_run:
            for y, q in batch:
                print(f"  Would run: {quarter_label(y, q)}")
                print(build_quarter_prompt(y, q))
                print()
            continue

        # --- Run quarters in parallel ---
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            fut_map = {pool.submit(
                _call_hermes, build_quarter_prompt(y, q)
            ): i for i, (y, q) in enumerate(batch)}

            for fut in as_completed(fut_map):
                i = fut_map[fut]
                y, q = batch[i]
                label = quarter_label(y, q)
                try:
                    result = fut.result()
                    results[i] = result
                    print(f"  [{label}] Done.")
                except Exception as e:
                    results[i] = f"ERROR: {e}"
                    print(f"  [{label}] FAILED: {e}")
                sys.stdout.flush()

        for i, (y, q) in enumerate(batch):
            label = quarter_label(y, q)
            r = results.get(i, "")
            if r.startswith("ERROR"):
                print(f"  {label}: {r}")
            else:
                lines = r.strip().split("\n")
                preview = lines[0][:120] if lines else "(empty)"
                print(f"  {label}: {preview}")

        # --- Commit quarter summaries before reflection ---
        print(f"  Committing {', '.join(labels)} summaries...")
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"summaries: {', '.join(labels)}"],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        print()

        # --- Reflection ---
        if not args.dry_run:
            print()
            print(f"  Running reflection for {', '.join(labels)}...")
            refl_cmd = [
                sys.executable, str(Path(__file__).resolve().parent / "pit_reflect.py"),
                "--quarters", *labels,
            ]
            try:
                refl_result = subprocess.run(refl_cmd, capture_output=True, text=True, timeout=1200)
                out = (refl_result.stdout or "").strip()
                lines = out.split("\n")[-5:] if out else ["(no output)"]
                for line in lines:
                    print(f"  [reflection] {line}")
            except Exception as e:
                print(f"  [reflection] FAILED: {e}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()