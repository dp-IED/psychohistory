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
from pathlib import Path

from harness.config import VAULT_DIR

GRAPH_VAULT = VAULT_DIR
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


def get_git_diff() -> str:
    """Get the git diff (uncommitted changes + staged) in graph-vault."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=str(GRAPH_VAULT), timeout=30,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached"],
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

    prompt = [
        "=== GRAPH VAULT REFLECTION ===",
        "",
        "Review the graph vault and improve the system.",
        "You own this vault entirely — structure, schemas, procedures.",
        "",
    ]
    if args.quarters:
        prompt.append(f"Focus on these recently written quarters: {', '.join(args.quarters)}")
        prompt.append("")
    prompt += [
        "=== CURRENT STRUCTURE ===",
        structure,
        "",
        "=== GIT DIFF (what changed this batch) ===",
        diff,
        "",
        "=== VAULT STRUCTURE ===",
        "",
        "The vault uses a Thread-as-Primary-Node model:",
        "  quarters/   — one file per quarter (written by the quarter agents)",
        "  threads/    — ongoing narratives spanning multiple quarters (causal chains,",
        "                movements, long-running dynamics)",
        "  concepts/   — recurring ideas and frameworks referenced across quarters",
        "",
        "=== YOUR TASK ===",
        "",
        "1. Read the files in graph-vault/. Understand what's there.",
        "2. Evaluate quality: are the summaries useful? Is the structure",
        "   well-organized? Does the diff suggest improvements?",
        "3. Improve the system:",
        "   - Edit _spec.md or create one if missing (defines the schema)",
        "   - Edit _procedure.md or create one (defines how summaries work)",
        "   - Create or update thread files in threads/ for causal chains and",
        "     recurring dynamics the quarters referenced (e.g. threads/cold-war.md)",
        "   - Create or update concept files in concepts/ for recurring ideas and",
        "     frameworks the quarters referenced (e.g. concepts/balance-of-power.md)",
        "   - Create entity stubs for any person, place, or org referenced via wikilinks",
        "   - Reorganize files if needed; prune what's not working",
        "   - Write notes for your future self",
        "4. Report what you changed and why.",
        "",
        "The goal: an evolving system that gets better at PIT summaries",
        "over time. Threads and concepts are the primary nodes — build them out.",
        "You are both the author and the student.",
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

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Reflection attempt {attempt}/{max_attempts} ---")
        result = _call_hermes(prompt_str)
        print(result)

        # Run vault validation
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
        if not structural_failures:
            print(f"\n✓ All {len(vdata.get('checks', []))} validation checks passed (structural clean).")
            break

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