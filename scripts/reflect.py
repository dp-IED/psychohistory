#!/usr/bin/env python3
"""Reflection — points the agent at the vault and tells it to reorganize.

No JSON parsing. The agent uses the Obsidian skill directly to edit files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runs import read_all_runs, runs_count, mean_brier, brier_by_category, worst_runs, best_runs


def _call_hermes(prompt: str, *, timeout: int = 600) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found")
    cmd = ["hermes", "-z", prompt, "--profile", "forecasting"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def build_prompt(vault_dir: str | Path) -> str:
    """Build a prompt that tells the agent to reorganize the vault."""
    vault_p = Path(vault_dir).resolve()
    policy_text = ""
    pp = vault_p / "policy.md"
    if pp.exists():
        policy_text = pp.read_text(encoding="utf-8")

    runs = read_all_runs(vault_dir)
    total = len(runs)
    mb = mean_brier(vault_dir)
    by_cat = brier_by_category(vault_dir)
    worst = worst_runs(vault_dir, 3)
    best = best_runs(vault_dir, 3)

    # Summary of each run
    run_summaries = []
    for r in runs[:10]:
        body = r.get("_body", "")[:80] if r.get("_body") else "(no text)"
        run_summaries.append(
            f"- {r.get('timestamp','?')} | {body} | "
            f"p_yes={r.get('p_yes','?')} | brier={r.get('brier','N/A')} | "
            f"src={r.get('source','?')} | cat={r.get('category','?')}"
        )

    mb_str = f"{mb:.4f}" if mb is not None else "N/A"
    per_cat_lines = "\n".join(f"  - {c}: {b:.4f}" for c, b in sorted(by_cat.items())) or "  (none scored)"
    worst_lines = "\n".join(
        f'  - brier={r.get("brier",0):.4f} p_yes={r.get("p_yes",0):.3f} | {r.get("_body","")[:80]}'
        for r in worst
    )
    best_lines = "\n".join(
        f'  - brier={r.get("brier",0):.4f} p_yes={r.get("p_yes",0):.3f} | {r.get("_body","")[:80]}'
        for r in best
    )
    recent_lines = "\n".join(run_summaries[:10]) if run_summaries else "(none)"

    return f"""You are a forecasting vault curator.

Your Obsidian vault is at {vault_p}. You have full read/write access using the Obsidian skill (`read_file`, `write_file`, `patch`, `search_files`).

Your task: reorganise this vault to maximise your future forecasting accuracy.

## Current vault structure

```
vault/
  policy.md    — YAML config + synthesis heuristics
  runs/        — {total} timestamped forecast run notes
  approaches/  — domain methodology notes
```

## Current policy.md

{policy_text[:3000] if policy_text.strip() else "(empty)"}

## Current runs ({total} total)

Mean Brier: {mb_str}

Per-category Brier:
{per_cat_lines}

Worst runs:
{worst_lines if worst else "  (none)"}

Best runs:
{best_lines if best else "  (none)"}

## Recent runs

{recent_lines}

## Your mandate

1. **Analyse performance** — what worked, what didn't, why
2. **Update policy.md** — adjust YAML frontmatter (shrinkage, max_steps, blind_spots) and write synthesis heuristics
3. **Write approach notes** — create `approaches/*.md` files for each domain you forecast in, with methodology and calibration notes
4. **Add wikilinks** — connect runs to approaches, approaches to policy.md, everything discoverable
5. **Reorganise** — create new folders, merge or split files, delete stale content. The vault is yours.

Be creative and analytic. Find structure that helps you make better predictions. You have the Obsidian skill — use `write_file`, `patch`, `search_files` directly.

Do not output a plan. Just do the work."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vault reflection.")
    parser.add_argument("--vault", default="vault", help="Vault directory")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt and exit")
    args = parser.parse_args()

    prompt = build_prompt(args.vault)

    if args.dry_run:
        print(prompt)
        return

    total = runs_count(args.vault)
    print(f"Reflection: {total} runs in {args.vault}/runs/. Prompting agent...")
    result = _call_hermes(prompt)
    print("Reflection complete.")
    if result:
        print(result[:500])


if __name__ == "__main__":
    raise SystemExit(main())
