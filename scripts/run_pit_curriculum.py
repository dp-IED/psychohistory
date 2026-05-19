#!/usr/bin/env python3
"""Run the PIT training curriculum: build timeline → calibrate on resolved markets.

Phases (run any subset):
  build     — chronological quarter summaries (pit_train.py)
  calibrate — resolved Polymarket backtest with PIT vault snapshots (run_backtest.py)
  audit     — print admissible vault paths for a sample cutoff (no hermes)

Examples:
  python scripts/run_pit_curriculum.py audit --cutoff 2024-06-01
  python scripts/run_pit_curriculum.py build --start 2022-Q1 --end 2025-Q4 --dry-run
  python scripts/run_pit_curriculum.py calibrate --max-questions 10 --skip-existing
  python scripts/run_pit_curriculum.py build calibrate --start 2022-Q1 --end 2024-Q4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR
from harness.vault_pit import list_admissible_paths, quarter_end_date

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def phase_audit(vault: Path, cutoff: date) -> int:
    paths = list_admissible_paths(vault, cutoff)
    timeline = [p for p in paths if p.startswith("timeline/")]
    entities = [p for p in paths if p.startswith("entities/")]
    print(f"Cutoff {cutoff.isoformat()} — {len(paths)} admissible files")
    print(f"  timeline: {len(timeline)}")
    for p in sorted(timeline)[-8:]:
        print(f"    {p}")
    print(f"  entities: {len(entities)}")
    for p in sorted(entities)[:12]:
        print(f"    {p}")
    if len(entities) > 12:
        print(f"    ... ({len(entities) - 12} more)")
    print(f"  policy: {[p for p in paths if p.startswith('_')]}")
    latest_q = None
    for p in timeline:
        label = Path(p).stem
        end = quarter_end_date(label)
        if end and (latest_q is None or end > latest_q[0]):
            latest_q = (end, label)
    if latest_q:
        print(f"  Latest quarter in view: {latest_q[1]} (ends {latest_q[0]})")
    else:
        print("  WARNING: no timeline quarters admissible — run `build` first.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PIT curriculum driver.")
    parser.add_argument(
        "phases",
        nargs="+",
        choices=["build", "calibrate", "audit"],
        help="Phases to run (in order)",
    )
    parser.add_argument("--vault", default=str(VAULT_DIR))
    parser.add_argument("--start", default="2022-Q1", help="Build phase: first quarter")
    parser.add_argument("--end", default="2025-Q4", help="Build phase: last quarter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-questions", type=int, default=15)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=1800)
    parser.add_argument(
        "--allow-categories",
        nargs="+",
        default=["all"],
        help="Market categories for calibrate (default: all)",
    )
    parser.add_argument("--cutoff", type=str, default="2024-06-01", help="Audit phase cutoff (ISO date)")
    args = parser.parse_args()

    vault = Path(args.vault).resolve()
    rc = 0

    for phase in args.phases:
        if phase == "audit":
            rc = phase_audit(vault, date.fromisoformat(args.cutoff))
        elif phase == "build":
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "pit_train.py"),
                "--start", args.start,
                "--end", args.end,
                "--batch-size", str(args.batch_size),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            rc = _run(cmd, "PIT timeline build")
        elif phase == "calibrate":
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_backtest.py"),
                "--vault", str(vault),
                "--max-questions", str(args.max_questions),
                "--lookback-days", str(args.lookback_days),
                "--enforce-pit",
                "--allow-categories", *args.allow_categories,
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            if args.concurrency > 1:
                cmd.extend(["--concurrency", str(args.concurrency)])
            if args.orchestrate:
                cmd.append("--orchestrate")
            if args.dry_run:
                print("(dry-run: would run calibrate with command above)")
                continue
            rc = _run(cmd, "PIT calibration backtest")

        if rc != 0:
            print(f"Phase {phase} failed (exit {rc}).")
            return rc

    print("\nCurriculum phase(s) complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
