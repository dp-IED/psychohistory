#!/usr/bin/env python3
"""Chain: 3 batches of 5 Polymarket questions + reflection each.
Writes log to LOG FILE so you can tail it.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from harness.config import VAULT_DIR, DEFAULT_POLICY_PATH

LOG = Path(__file__).resolve().parent.parent / ".hermes/batch_chain.log"
VAULT = VAULT_DIR
ROOT = Path(__file__).resolve().parent.parent

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run(cmd: list[str], label: str, timeout: int = 3600) -> bool:
    log(f"{label}: starting")
    log(f"  {' '.join(str(c) for c in cmd)}")
    try:
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONPATH"] = str(ROOT)
        subprocess_env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=subprocess_env,
            cwd=str(ROOT),
        )

        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n\r")
            if line:
                log(f"  {line}")

        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            log(f"{label}: FAILED (exit {proc.returncode})")
            return False
        log(f"{label}: OK")
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        log(f"{label}: TIMED OUT ({timeout}s)")
        return False
    except Exception as e:
        log(f"{label}: ERROR: {e}")
        return False

def vault_stats() -> str:
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from harness.runs import runs_count, mean_brier, brier_by_category
        rc = runs_count(str(VAULT))
        mb = mean_brier(str(VAULT))
        bc = brier_by_category(str(VAULT))
        parts = [f"runs={rc}"]
        if mb is not None:
            parts.append(f"brier={mb:.4f}")
        if bc:
            parts.append(f"by_cat={dict(sorted(bc.items()))}")
        return " | ".join(parts)
    except Exception:
        return "(stats unavailable)"

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch chain: batch → reflect loop.")
    parser.add_argument("--orchestrate", action="store_true",
                        help="Use multi-agent orchestration for forecasts")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show orchestrator dashboard after completion")
    args = parser.parse_args()

    log("=" * 60)
    log("BATCH CHAIN STARTED — 3 cycles × (5 questions + reflection)")
    log(f"Vault: {VAULT}")
    log(f"Mode: {'ORCHESTRATED' if args.orchestrate else 'SINGLE-AGENT'}")
    log(f"Initial: {vault_stats()}")
    log("=" * 60)

    for cycle in range(1, 4):
        log("")
        log(f"{'='*50}")
        log(f"CYCLE {cycle}/3 — BATCH")
        log(f"{'='*50}")

        batch_cmd = [
            sys.executable, "-m", "scripts.run_backtest",
            "--source", "polymarket",
            "--max-questions", "10",
            "--vault", str(VAULT),
            "--policy", str(DEFAULT_POLICY_PATH),
            "--concurrency", "5",
            "--skip-existing",
            "--allow-categories", "politics", "economics", "crypto",
        ]
        if args.orchestrate:
            batch_cmd.append("--orchestrate")

        ok = run(
            batch_cmd,
            label=f"Batch {cycle}",
            timeout=3600,
        )

        log(f"After batch: {vault_stats()}")

        if ok:
            log("")
            log(f"{'='*50}")
            log(f"CYCLE {cycle}/3 — REFLECTION")
            log(f"{'='*50}")
            run(
                [sys.executable, "-m", "scripts.reflect_graph"],
                label=f"Reflect {cycle}",
                timeout=900,
            )
            log(f"After reflect: {vault_stats()}")
        else:
            log("Skipping reflection — batch failed")

        log(f"End of cycle {cycle}/3")
        log("")

    log("=" * 60)
    log("ALL CYCLES COMPLETE")
    log(f"Final: {vault_stats()}")
    log("=" * 60)

    if args.dashboard:
        log("Showing orchestration dashboard...")
        subprocess.run(
            [sys.executable, "-m", "scripts.orchestrator_dashboard"],
            cwd=str(ROOT),
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
