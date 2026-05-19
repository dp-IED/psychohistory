#!/usr/bin/env python3
"""One-off targeted reflection for persistent market-calibration misses.

Does not modify pit_reflect.py or pit_market_calibration.py. Run:
  python scripts/pit_reflect_calibration_targets.py
  python scripts/pit_reflect_calibration_targets.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.config import VAULT_DIR
from harness.pit_market_probe import (
    DEFAULT_MARKET_CALIBRATION_BAND,
    _classify_calibration_miss,
    load_results,
)
import importlib.util

_reflect_spec = importlib.util.spec_from_file_location(
    "pit_reflect", ROOT / "scripts" / "pit_reflect.py"
)
_pit_reflect = importlib.util.module_from_spec(_reflect_spec)
assert _reflect_spec.loader is not None
_reflect_spec.loader.exec_module(_pit_reflect)
_REFLECT_DIFF_PATHS = _pit_reflect._REFLECT_DIFF_PATHS
_call_hermes = _pit_reflect._call_hermes
get_git_diff = _pit_reflect.get_git_diff
tree = _pit_reflect.tree

GRAPH_VAULT = VAULT_DIR
RESULTS = ROOT / "data" / "pit_market_probes" / "results.jsonl"

DEFAULT_PROBE_IDS = (
    "gold-gold_18_will-biden-drop-out-before-the-democratic-nation",
    "gold-gold_03_israel-x-hamas-ceasefire-by-july-15",
)


def _load_target_rows(probe_ids: tuple[str, ...]) -> list[dict]:
    rows = load_results(RESULTS)
    by_id = {r["probe_id"]: r for r in rows}
    missing = [pid for pid in probe_ids if pid not in by_id]
    if missing:
        raise SystemExit(f"probe(s) not in {RESULTS}: {missing}")
    return [by_id[pid] for pid in probe_ids]


def format_target_feedback(rows: list[dict], *, band: float) -> str:
    lines = [
        "=== TARGETED CALIBRATION REFLECT (2 probes only) ===",
        "",
        "Ground truth at cutoff: Polymarket YES price. Success = next forecaster within ±5pt of PM.",
        "",
    ]
    for r in rows:
        q = (r.get("question") or "").split("\n")[0][:100]
        lines.append(
            f"PROBE {r['probe_id']}\n"
            f"  cutoff={r['cutoff']}  forecaster_p={float(r['p_yes']):.3f}  "
            f"market={r.get('market_yes_at_cutoff')}  mae={r.get('market_abs_error')}\n"
            f"  question: {q}\n"
            f"  diagnosis: {_classify_calibration_miss(r, band=band)}"
        )
        for label, key, n in (
            ("librarian_sources", "librarian_sources", 8),
            ("librarian_excluded", "librarian_excluded", 4),
        ):
            vals = r.get(key) or []
            if vals:
                lines.append(f"  {label}: {', '.join(str(v) for v in vals[:n])}")
        lib = (r.get("librarian_conjuncture") or "")[:500]
        if lib:
            lines.append(f"  librarian_conjuncture (truncated): {lib}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(rows: list[dict], *, structure: str, diff: str, feedback: str) -> str:
    biden = rows[0]
    hamas = rows[1] if len(rows) > 1 else rows[0]
    return "\n".join([
        "=== TARGETED MARKET-CALIBRATION REFLECTION AGENT ===",
        "",
        "You are a specialized reflector for TWO persistent ±5pt misses after a full vault batch.",
        "Edit ONLY what is needed so the NEXT forecaster outputs p_yes within ±0.05 of Polymarket at cutoff.",
        "",
        feedback,
        "",
        "=== FAILURE MODE A (Biden DNC — forecaster TOO HIGH) ===",
        f"Probe: {biden['probe_id']}  cutoff={biden['cutoff']}  "
        f"forecaster={float(biden['p_yes']):.3f}  market={biden.get('market_yes_at_cutoff')}",
        "",
        "Root cause to fix in the VAULT (not forecaster prose):",
        "- Librarian/forecaster treated 'Biden withdrew July 21' as 'question effectively resolved → p≈1'.",
        "- Rule 9 already says PM is ground truth; PM was 0.715 — residual uncertainty was still priced.",
        "- threads/2024-us-presidential-election.md has NO pit_body_cutoff; overview/timeline read as post-hoc certainty.",
        "",
        "Required edits (pick minimal set):",
        "1. Add pit_body_cutoff: 2024-07-28 to threads/2024-us-presidential-election.md.",
        "2. Add a ## PIT conjuncture at cutoff (2024-07-28) section listing ONLY pre-cutoff facts + "
        "   **market-priced residual uncertainties** (resolution lag, delegate release mechanics, "
        "   scope of 'drop out before convention' vs successor consolidation) — NOT 'withdrawal = p≈1'.",
        "3. Add Rule 11 to _forecast_instructions.md: when a public event occurred before cutoff but "
        "   PM YES < 0.90, align output to PM; document residual uncertainties the market prices — "
        "   never override PM with 'event already happened'.",
        "4. Optional: short concept stub concepts/polymarket-residual-uncertainty-after-public-event.md "
        "   if mechanism is reusable.",
        "",
        "FORBIDDEN: adding more post-July-21 narrative to fix the miss; citing runs/forecasts/; "
        "writing 'the market was wrong'.",
        "",
        "=== FAILURE MODE B (Hamas Jul 15 — forecaster TOO LOW) ===",
        f"Probe: {hamas['probe_id']}  cutoff={hamas['cutoff']}  "
        f"forecaster={float(hamas['p_yes']):.3f}  market={hamas.get('market_yes_at_cutoff')}",
        "",
        "Root cause to fix in the VAULT:",
        "- Prior reflection added Key Dynamic #8 (7-10 week diplomatic bandwidth lag) and forecaster "
        "  applied it to a **7-day deadline** question, crushing p_yes to 0.15 while PM was 0.49.",
        "- Lag model is for long-horizon announcement timing (e.g. Oct breakthrough), NOT for "
        "  short-fuse markets with active shuttle diplomacy and Iran-ceasefire spillover.",
        "",
        "Required edits:",
        "1. Add pit_body_cutoff: 2025-07-08 to threads/gaza-ceasefire-negotiations-2025.md.",
        "2. Split Key Dynamic #8 into: (A) long-horizon refocusing lag 7-10 weeks for major breakthrough; "
        "   (B) short-horizon (≤14 day deadline) — PM can stay ~0.45-0.55 when diplomacy is active, "
        "   Iran war just ended, and cumulative pressure vectors are building — do NOT apply lag as veto.",
        "3. Add ## PIT conjuncture at cutoff (2025-07-08): 7 days to Jul 15 deadline; list active "
        "   pre-cutoff diplomatic signals (spring rebuild, June 24 Iran ceasefire freeing bandwidth, "
        "   no Oct breakthrough facts).",
        "4. Add Rule 12 to _forecast_instructions.md OR extend concepts/diplomatic-pressure-tipping-point.md: "
        "   horizon-matched base rates — long lag concepts must not zero out short-deadline market prices.",
        "",
        "FORBIDDEN: citing October 2025 ceasefire; strengthening 'won't happen in 7 days' without "
        "explaining why PM was 0.49 anyway.",
        "",
        "=== ANTI-LEAKAGE (same as Rule 10) ===",
        "Do NOT document terminal outcomes. Do NOT read graph-vault/runs/ or forecasts/.",
        "Allowed: pit_body_cutoff, PIT sections, mechanism splits, Rule 11/12, trim post-cutoff bullets.",
        "",
        "=== FILES TO READ FIRST ===",
        "  threads/2024-us-presidential-election.md",
        "  threads/gaza-ceasefire-negotiations-2025.md",
        "  concepts/incumbent-withdrawal-cascade.md",
        "  concepts/diplomatic-pressure-tipping-point.md",
        "  graph-vault/_forecast_instructions.md (Rules 9-10)",
        "",
        "=== CURRENT STRUCTURE (abbrev) ===",
        structure,
        "",
        "=== GIT DIFF (graph paths only) ===",
        diff,
        "",
        "=== DELIVERABLE ===",
        "Make the edits. End with a short report: files changed, which probe each targets, "
        "and the mechanism that should move p_yes toward market on re-run.",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted calibration-miss reflection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-id", action="append", dest="probe_ids")
    parser.add_argument("--band", type=float, default=DEFAULT_MARKET_CALIBRATION_BAND)
    args = parser.parse_args()

    probe_ids = tuple(args.probe_ids) if args.probe_ids else DEFAULT_PROBE_IDS
    rows = _load_target_rows(probe_ids)
    feedback = format_target_feedback(rows, band=args.band)
    structure = "\n".join(tree(GRAPH_VAULT)[:35])
    diff = get_git_diff()
    prompt = build_prompt(rows, structure=structure, diff=diff, feedback=feedback)

    print(feedback)
    print(f"\nGit diff: {len(diff)} chars\n")

    if args.dry_run:
        print("[DRY-RUN] Prompt:\n")
        print(prompt)
        return 0

    print("--- Targeted reflection (Hermes) ---")
    result = _call_hermes(prompt)
    print(result)

    diff_stat = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True,
        text=True,
        cwd=str(GRAPH_VAULT),
    )
    subprocess.run(
        ["git", "add", "--", *_REFLECT_DIFF_PATHS],
        cwd=str(GRAPH_VAULT),
        check=False,
    )
    diff_stat = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True,
        text=True,
        cwd=str(GRAPH_VAULT),
    )
    if diff_stat.stdout.strip():
        msg = "reflection: targeted calibration misses (Biden DNC + Hamas Jul15)"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(GRAPH_VAULT),
            check=False,
        )
        print(f"[git] {diff_stat.stdout.strip()}")
    else:
        unstaged = subprocess.run(
            ["git", "diff", "--stat", "--", *_REFLECT_DIFF_PATHS],
            capture_output=True,
            text=True,
            cwd=str(GRAPH_VAULT),
        )
        if unstaged.stdout.strip():
            print("[git] Hermes reported edits but nothing staged — check vault manually.")
            print(unstaged.stdout)
        else:
            print("[git] No vault file changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
