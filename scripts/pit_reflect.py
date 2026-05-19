#!/usr/bin/env python3
"""Reflection: review PIT summaries and improve the graph vault system.

The agent reads the git diff of what changed in the latest batch,
evaluates quality, and improves the system structure (_spec.md, schemas, etc.).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import VAULT_DIR

GRAPH_VAULT = VAULT_DIR
TESTBED = Path(__file__).resolve().parent.parent
BLIND_RESULTS_PATH = TESTBED / "pit_blind_test" / "results.json"
MARKET_CALIBRATION_RESULTS = TESTBED / "data" / "pit_market_probes" / "results.jsonl"
MARKET_CALIBRATION_FEEDBACK = TESTBED / "data" / "pit_market_probes" / "last_calibration_feedback.txt"
_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200


def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _HERMES_PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes (profile={_HERMES_PROFILE}) failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def tree(dir_path: Path, prefix: str = "", max_files: int = 40) -> list[str]:
    """Build a compact ASCII tree view of a directory. Returns list of lines."""
    lines: list[str] = []
    entries = sorted(dir_path.iterdir())
    count = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if count >= max_files:
            remaining = len([e for e in entries if not e.name.startswith(".")]) - max_files
            lines.append(f"{prefix}+-- ... ({remaining} more)")
            break
        is_last = (count == len([e for e in entries if not e.name.startswith(".")]) - 1) or count >= max_files - 1
        connector = "+-- " if is_last else "|-- "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            if count < max_files:
                indent = "    " if is_last else "|   "
                sub = tree(entry, prefix + indent, max_files - count - 1)
                lines.extend(sub)
        else:
            size = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name} ({size}b)")
        count += 1
    return lines


def format_predictive_feedback(
    vault_dir: Path,
    *,
    max_blind_misses: int = 8,
    max_bad_runs: int = 5,
) -> str:
    """Summarize forecast misses for reflection to extend conjunctures."""
    lines: list[str] = []

    if BLIND_RESULTS_PATH.is_file():
        try:
            data = json.loads(BLIND_RESULTS_PATH.read_text())
            misses = [r for r in data.get("results", []) if not r.get("correct")]
            lines.append(
                f"Gold blind harness: {data.get('correct', '?')}/{data.get('total', '?')} correct "
                f"({data.get('accuracy_pct', '?')}%)"
            )
            for r in misses[:max_blind_misses]:
                exp = "YES" if r.get("expected") == 1.0 else "NO"
                got = r.get("prediction", "?")
                qs = ", ".join(r.get("quarters_used") or [])
                q = (r.get("question") or r.get("case_id", ""))[:120]
                lines.append(f"  MISS [{r.get('case_id', '')[:40]}] expected={exp} got={got}")
                lines.append(f"    quarters in view: {qs or '(none)'}")
                lines.append(f"    question: {q}")
        except (json.JSONDecodeError, OSError) as e:
            lines.append(f"(could not read blind results: {e})")

    from harness.runs import mean_brier, worst_runs

    mb = mean_brier(vault_dir)
    if mb is not None:
        lines.append(f"Polymarket backtest mean Brier: {mb:.4f}")
    for r in worst_runs(vault_dir, n=max_bad_runs):
        b = r.get("brier")
        q = (r.get("_body") or r.get("question") or r.get("question_text") or "")[:100]
        cat = r.get("category", "general")
        lines.append(f"  HIGH BRIER {b:.4f} [{cat}] {q}")

    if MARKET_CALIBRATION_FEEDBACK.is_file():
        try:
            cal = MARKET_CALIBRATION_FEEDBACK.read_text(encoding="utf-8").strip()
            if cal:
                lines.append("")
                lines.append("=== MARKET CALIBRATION (p_yes vs Polymarket at cutoff) ===")
                lines.append(cal)
        except OSError:
            pass
    elif MARKET_CALIBRATION_RESULTS.is_file():
        try:
            from harness.pit_market_probe import format_market_calibration_feedback

            cal = format_market_calibration_feedback(
                MARKET_CALIBRATION_RESULTS, for_reflect=True,
            )
            lines.append("")
            lines.append("=== MARKET CALIBRATION (p_yes vs Polymarket at cutoff) ===")
            lines.append(cal)
        except Exception as e:
            lines.append(f"(market calibration feedback unavailable: {e})")

    if not lines:
        return "(no predictive feedback yet — run calibrate or pit_blind_forecast_test.py score)"
    return "\n".join(lines)


# Graph paths safe for reflection — excludes runs/forecasts that embed outcomes and reasoning.
_REFLECT_DIFF_PATHS = (
    "threads",
    "concepts",
    "timeline",
    "entities",
    "history",
    "agent-roles",
    "_forecast_instructions.md",
    "_procedure.md",
    "_spec.md",
    "_index.md",
)


def get_git_diff() -> str:
    """Git diff for graph nodes only — not runs/forecasts (calibration leakage)."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", *_REFLECT_DIFF_PATHS],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--", *_REFLECT_DIFF_PATHS],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        diff = result.stdout.strip() + "\n" + staged.stdout.strip()
        diff = diff.strip()
        if not diff:
            return "(no changes since last commit)"
        # Truncate to avoid blowing the prompt
        lines = diff.split("\n")
        if len(lines) > 400:
            lines = lines[:400] + [f"... ({len(lines) - 400} more lines truncated)"]
        return "\n".join(lines)
    except Exception as e:
        return f"(could not get git diff: {e})"


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT summary reflection.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quarters", nargs="+", help="Specific quarter labels to focus on, e.g. 1900-Q1 1900-Q2")
    parser.add_argument("--forecast-threshold", type=float, default=0.25,
                        help="Max acceptable Brier after vault fixes (default 0.25). Set 0 to skip forecast retry.")
    parser.add_argument(
        "--no-predictive-feedback",
        action="store_true",
        help="Do not inject blind-harness / backtest miss summary into the reflection prompt.",
    )
    args = parser.parse_args()

    # Scan vault structure
    structure = "\n".join(tree(GRAPH_VAULT))

    # Count files
    all_files = list(GRAPH_VAULT.rglob("*.md"))
    non_timeline = [f for f in all_files if "timeline" not in str(f)]

    print(f"Graph vault: {len(all_files)} total files, {len(non_timeline)} non-timeline")
    print()

    # Get git diff
    diff = get_git_diff()
    print(f"Git diff size: {len(diff)} chars")
    print()

    predictive = ""
    if not args.no_predictive_feedback:
        predictive = format_predictive_feedback(GRAPH_VAULT)
        print("Predictive feedback loaded for reflection.")
        print()

    prompt = [
        "=== GRAPH VAULT REFLECTION (CONJUNCTURE + PREDICTIVE POWER) ===",
        "",
        "Review the vault and improve **relational / conjunctural** knowledge — not entity catalogs.",
        "Quarters should read as: forces interacting in time → threads/concepts carry causality.",
        "",
    ]
    if args.quarters:
        prompt.append(f"Focus on these recently written quarters: {', '.join(args.quarters)}")
        prompt.append("")
    prompt += [
        "=== PREDICTIVE FEEDBACK (priority signal) ===",
        predictive,
        "",
        "Use misses above to decide what to extend. A failed forecast means the conjuncture",
        "was wrong, thin, or missing an interaction — not that you need more [[proper nouns]].",
        "",
        "When MARKET CALIBRATION feedback is present: diagnose librarian retrieval",
        "(leaked post-cutoff facts? missing uncertainties?) AND forecaster calibration",
        "(ignored PM price? over-trusted vault narrative?). Patch threads with pit_body_cutoff,",
        "trim post-hoc bullets, extend conjuncture — update pit-research-librarian.md only if",
        "retrieval rules need changing.",
        "",
        "=== ANTI-LEAKAGE FOR DOC EDITS (mandatory) ===",
        "Calibration misses mean 'p_yes was wrong vs Polymarket at cutoff' — NOT 'write the",
        "outcome into the vault so the next forecaster says p_yes≈1.'",
        "NEVER add post-cutoff resolution facts to threads/concepts/timeline to 'fix' a miss.",
        "NEVER read or cite graph-vault/runs/ or forecasts/ during reflection — they contain hindsight.",
        "Allowed: pit_body_cutoff, PIT-only conjuncture files, trimming bullets after cutoff,",
        "mechanism/concept fixes, Rule 9/10 in _forecast_instructions.md, librarian retrieval rules.",
        "",
        "=== CURRENT STRUCTURE ===",
        structure,
        "",
        "=== GIT DIFF (what changed this batch) ===",
        diff,
        "",
        "=== PRIMARY NODES ===",
        "  timeline/  — quarter files: Conjuncture → Threads → Chronicle → Cross-domain",
        "  threads/   — ongoing dynamics; extend with quarter deltas and cross-links",
        "  concepts/  — patterns that transfer across domains; tie to Validated By / Failed By",
        "",
        "=== YOUR TASK ===",
        "",
        "1. Read timeline/ files from this batch. If they are entity-heavy chronicles, rewrite",
        "   the Conjuncture and Cross-domain sections; demote proper-noun wikilinks to plain text.",
        "2. For each predictive MISS or high-Brier run: diagnose which **interaction** was missing",
        "   (e.g. political deadline × ceasefire, Fed guidance × fiscal shock).",
        "3. Extend conjunctures where predictive power demands it:",
        "   - Update threads/ with new causal links and [[quarter]] backlinks",
        "   - Update concepts/ with sharpened mechanisms; note which forecasts validated/failed them",
        "   - Patch timeline/ KEY THEMES or Conjuncture sections if the quarter file exists but",
        "     under-specified the interaction that caused the miss",
        "4. Entity stubs: only for named actors in a **failed or high-Brier forecast question**",
        "   who lack any vault coverage — not for every wikilink in a training quarter.",
        "5. Light-touch _procedure.md edits only if they reinforce conjuncture-first training.",
        "6. Report: what conjunctures you extended and which forecast errors they target.",
        "",
        "Success = next forecast on similar questions uses thread/concept interaction, not trivia.",
    ]
    prompt_str = "\n".join(prompt)
    if args.dry_run:
        print("[DRY-RUN] Would prompt Hermes:\n")
        print(prompt_str)
        return

    # Structural checks that gate the loop (agent can fix these in one session)
    STRUCTURAL_CHECKS = {
        "Dual directory", "Frontmatter drift", "Zero-byte files",
        "Missing annual summaries", "Entity backlinks",
        "Quarter cutoff", "Related Periods",
    }

    RE_FORECAST_BRIER_CUTOFF = args.forecast_threshold

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Reflection attempt {attempt}/{max_attempts} ---")
        result = _call_hermes(prompt_str)
        print(result)

        # ── Phase 1: Run vault validation ──
        validate_script = Path(__file__).resolve().parent / "validate_vault.py"
        val = subprocess.run(
            [sys.executable, str(validate_script), "--json"],
            capture_output=True, text=True, timeout=120,
        )
        try:
            vdata = json.loads(val.stdout) if val.stdout.strip() else {"passed": False, "error": "empty validation output"}
        except json.JSONDecodeError:
            print(f"[VALIDATION] Could not parse validation JSON:\n{val.stdout[:500]}")
            vdata = {"passed": False, "error": "parse failure"}

        # Separate structural failures (gating) from aspirational (info only)
        structural_failures: list[dict] = []
        aspirational_issues = 0
        for check in vdata.get("checks", []):
            if not check.get("passed"):
                if check["name"] in STRUCTURAL_CHECKS:
                    structural_failures.append(check)
                else:
                    aspirational_issues += len(check.get("issues", []))

        if aspirational_issues:
            print(f"\n  [info] {aspirational_issues} aspirational issues (non-gating)")
        if structural_failures:
            total_structural = sum(len(c.get("issues", [])) for c in structural_failures)
            print(f"\n✗ {total_structural} structural issues remain (attempt {attempt}/{max_attempts})")
            for check in structural_failures:
                issues = check.get("issues", [])
                print(f"  [{check['name']}] {len(issues)} issues")
                for iss in issues[:5]:
                    print(f"    {iss}")
                if len(issues) > 5:
                    print(f"    ... ({len(issues) - 5} more)")

            if attempt >= max_attempts:
                print(f"\nMax attempts ({max_attempts}) reached. Exiting with structural failures.")
                sys.exit(1)

            # Append structural failures to the prompt for the retry
            prompt_str += f"\n\n=== STRUCTURAL FAILURES TO FIX (attempt {attempt}) ===\n"
            for check in structural_failures:
                prompt_str += f"\n## {check['name']}\n"
                for iss in check.get("issues", [])[:30]:
                    prompt_str += f"- {iss}\n"
                if len(check.get("issues", [])) > 30:
                    prompt_str += f"- ... and {len(check['issues']) - 30} more\n"
            prompt_str += "\nFix ALL of the above structural issues before the next validation pass."
            continue  # Go back to Phase 1 — vault still broken

        # ── Phase 2 (vault is structurally clean): Forecast retry loop ──
        print(f"\n✓ Vault structurally clean. Checking forecast quality...")

        if RE_FORECAST_BRIER_CUTOFF <= 0:
            print("  Forecast retry disabled (--forecast-threshold <= 0).")
            break

        # Import only when needed (avoids circular deps at module level)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from harness.orchestrator import run_structured
        from harness.runs import read_all_runs

        recent = read_all_runs(str(GRAPH_VAULT))
        resolved = [r for r in recent if r.get("resolution") is not None][:5]

        if not resolved:
            print("  No resolved markets to validate forecasts against.")
            break

        # Re-forecast each resolved market using the improved vault
        bad_predictions: list[tuple[str, float, str]] = []
        for r in resolved:
            qtext = r.get("_body", r.get("question", r.get("question_text", "")))
            if not qtext:
                continue
            try:
                t0 = time.time()
                p_yes, reasoning = run_structured(
                    qtext,
                    cutoff=r.get("cutoff"),
                    vault_dir=str(GRAPH_VAULT),
                    question_id=r.get("question_id", ""),
                    source=r.get("source", "polymarket"),
                    category=r.get("category", "general"),
                    resolution=r["resolution"],
                )
                new_brier = (p_yes - (1.0 if r["resolution"] else 0.0)) ** 2
                elapsed = time.time() - t0
                old_brier = r.get("brier")
                delta = f" (was {old_brier:.4f})" if old_brier is not None else ""
                print(f"  re-forecast [{elapsed:.0f}s] Brier={new_brier:.4f}{delta} — {qtext[:55]}")
                if new_brier > RE_FORECAST_BRIER_CUTOFF:
                    bad_predictions.append((qtext, new_brier, reasoning[:400]))
            except Exception as e:
                print(f"  re-forecast FAILED — {qtext[:55]}: {e}")

        if not bad_predictions:
            print(f"\n✓ All {len(resolved)} re-forecasts passed Brier ≤ {RE_FORECAST_BRIER_CUTOFF}.")
            break

        # Some forecasts still bad — do another reflection pass
        print(f"\n✗ {len(bad_predictions)}/{len(resolved)} predictions exceed Brier cutoff.")
        print("  Running targeted reflection to fix vault gaps that caused poor forecasts...")

        prompt_str += f"\n\n=== FORECAST QUALITY FAILURES (attempt {attempt}) ===\n"
        prompt_str += f"The following predictions had Brier > {RE_FORECAST_BRIER_CUTOFF} after vault fixes.\n"
        prompt_str += (
            "Diagnose which **conjuncture / thread interaction / concept** was missing or wrong.\n"
            "Extend threads and concepts; rewrite quarter Conjuncture or Cross-domain sections if needed.\n"
            "Entity stubs only when a named actor in the question had zero vault role in the failure.\n"
        )
        for qtext, brier, _reason in bad_predictions:
            prompt_str += f"\n## Brier={brier:.4f}: {qtext}\n"
            prompt_str += (
                "  (Re-forecast reasoning omitted — do not infer terminal outcome from it. "
                "Fix conjuncture/PIT structure only.)\n"
            )
        prompt_str += (
            "\n---\n"
            "Do not add entity-phone-book coverage. Fix the interaction structure so the next "
            "forecast reasons through forces in time, not proper-noun lookup.\n"
            "Do not document what eventually happened — only what should have been knowable at cutoff.\n"
        )

        # If at max attempts, fail open — commit what we have and exit
        if attempt >= max_attempts:
            print(f"\nMax attempts ({max_attempts}) reached. Some predictions still exceed Brier cutoff.")
            print("Committing vault fixes and exiting (forecast quality failures remain).")
            break

    # Commit reflection changes so the next batch starts clean
    commit_result = subprocess.run(
        ["git", "add", "-A"],
        capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
    )
    diff_stat = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
    )
    if diff_stat.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", f"reflection: post-{', '.join(args.quarters) if args.quarters else 'batch'} review"],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        print(f"  [git] Committed reflection changes: {diff_stat.stdout.strip()}")
    else:
        print("  [git] No changes from reflection.")


if __name__ == "__main__":
    main()