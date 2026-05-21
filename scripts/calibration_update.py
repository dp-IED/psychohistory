#!/usr/bin/env python3
"""Update concept calibration tables from resolved forecast runs."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAULT = HERE.parent / "graph-vault"
sys.path.insert(0, str(HERE.parent))

# ── mechanism → indicator combination mapping ──────────────────────────

# Maps resolved runs to mechanism concepts and their indicator combinations.
# Each entry: (concept_file, indicator_combination, mechanism_label)
MECHANISM_MAP = {
    # FPTP fragmentation
    "gold_26": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Front-runner 35-45% + 2+ opposition candidates + FPTP system",
                "Divided-opposition-win"),
    "gold_07": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Front-runner 35-45% + 2+ opposition candidates + FPTP system",
                "Divided-opposition-win"),
    "gold_27": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Third-party/non-front-runner + FPTP system",
                "Third-party-ceiling"),
    "gold_05": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Third-party/non-front-runner + FPTP system",
                "Third-party-ceiling"),
    "gold_06": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Third-party/non-front-runner + FPTP system",
                "Third-party-ceiling"),
    "gold_04": ("domains/east-asia/concepts/divided-opposition-plurality-win/_concept.md",
                "Third-party/non-front-runner + FPTP system",
                "Third-party-ceiling"),
    # Procedural certainty
    "gold_21": ("domains/global/concepts/short-horizon-procedural-certainty/_concept.md",
                "All remaining steps ministerial + <2 week window + no disruption path",
                "Ministerial lock-in"),
    "gold_28": ("domains/global/concepts/short-horizon-procedural-certainty/_concept.md",
                "All remaining steps ministerial + 2-4 week window + no disruption path",
                "Ministerial lock-in (longer)"),
    "gold_29": ("domains/global/concepts/short-horizon-procedural-certainty/_concept.md",
                "All remaining steps ministerial + <2 week window + no disruption path",
                "Ministerial lock-in"),
    "gold_22": ("domains/global/concepts/short-horizon-procedural-certainty/_concept.md",
                "Court cert before judgment + compressed schedule + zero stays",
                "Procedural inverse (NO lock-in)"),
    # Authoritarian electoral facade
    "gold_19": ("domains/usa/concepts/authoritarian-electoral-facade.md",
                "Opposition PVT infrastructure + credible tally collection + unified opposition",
                "Opposition wins vote (documented)"),
    "gold_20": ("domains/usa/concepts/authoritarian-electoral-facade.md",
                "No PVT infrastructure + regime controls electoral commission",
                "Regime candidate loses vote"),
}


def load_runs() -> list[dict]:
    """Load all forecast runs and merge in gold set resolutions when available."""
    runs_dir = VAULT / "runs"
    if not runs_dir.exists():
        return []
    from harness.runs import read_all_runs
    all_runs = read_all_runs(str(VAULT))

    # Load gold set results for resolution data
    gold_resolutions = {}
    gold_results_path = HERE.parent / "pit_blind_test" / "results.json"
    if gold_results_path.exists():
        import json
        gold_data = json.loads(gold_results_path.read_text())
        for r in gold_data.get("results", []):
            gold_resolutions[r["case_id"]] = r["expected"]

    runs = []
    for run in all_runs:
        # Extract gold_XX from question_id
        qid = run.get("question_id", "")
        source = run.get("source", "")
        short_gold_id = None
        for text in [qid, source]:
            m = re.search(r"gold_(\d+)", text)
            if m:
                short_gold_id = f"gold_{int(m.group(1)):02d}"
                # Find full case ID in gold_resolutions
                for full_id in gold_resolutions:
                    if full_id.startswith(short_gold_id):
                        run["resolution"] = bool(gold_resolutions[full_id])
                        break
                break

        if run.get("resolution") is not None:
            runs.append(run)

    return runs


def extract_gold_id(run: dict) -> str | None:
    """Extract gold_XX from a run's question_id or source."""
    qid = run.get("question_id", "")
    source = run.get("source", "")
    for text in [qid, source]:
        m = re.search(r"gold_(\d+)", text)
        if m:
            gold_num = int(m.group(1))
            for key in MECHANISM_MAP:
                if key.endswith(f"_{gold_num:02d}"):  # match e.g. gold_07
                    return key
            # Fallback: just return the full gold_XX prefix
            for key in MECHANISM_MAP:
                if key.startswith(f"gold_{gold_num:02d}"):
                    return key
    return None


def update_calibration_table(concept_path: Path, indicator: str, resolution: bool) -> bool:
    """Update one row in a concept file's calibration table. Returns True if changed."""
    if not concept_path.exists():
        print(f"  SKIP: concept file not found: {concept_path}")
        return False

    content = concept_path.read_text(encoding="utf-8")
    today = date.today().isoformat()

    # Find the row for this indicator combination
    # Pattern: | indicator_text | mechanism | YES | NO | ... |
    # We need to find the row and increment YES or NO
    rows = content.split("\n")
    updated = False
    new_rows = []

    in_calibration = False
    for line in rows:
        if "## Empirical Calibration" in line or "## Calibration" in line:
            in_calibration = True

        if in_calibration and indicator in line and "| " in line:
            # Skip header and separator rows
            if "---" in line or "Indicator Combination" in line or "YES" in line.split("|")[3] if len(line.split("|")) > 3 else False:
                new_rows.append(line)
                continue

            # Parse existing row: | indicator | mechanism | YES | NO | Hit Rate | N | Updated |
            parts = [p.strip() for p in line.split("|")]
            yes_count = 0
            no_count = 0
            if len(parts) >= 7:
                try:
                    yes_count = int(parts[3]) if parts[3] and parts[3] != "YES" else 0
                    no_count = int(parts[4]) if parts[4] and parts[4] != "NO" else 0
                except (ValueError, IndexError):
                    # If parsing fails, treat as header/unknown — skip update but keep row
                    new_rows.append(line)
                    continue

            if resolution:
                yes_count += 1
            else:
                no_count += 1
            total = yes_count + no_count
            hit_rate = f"{yes_count / total * 100:.0f}%" if total > 0 else "—"

            # Rebuild the row
            new_line = f"| {indicator} | {parts[2] if len(parts) > 2 else ''} | {yes_count} | {no_count} | {hit_rate} | {total} | {today} |"
            new_rows.append(new_line)
            updated = True
        else:
            new_rows.append(line)

    if updated:
        concept_path.write_text("\n".join(new_rows), encoding="utf-8")
        print(f"  UPDATED: {concept_path.relative_to(VAULT)} — {indicator[:50]}... (now {yes_count}Y/{no_count}N)")
    else:
        print(f"  NO MATCH: {concept_path.relative_to(VAULT)} — indicator not found in calibration table")

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Update concept calibration tables from resolved runs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't write")
    args = parser.parse_args()

    runs = load_runs()
    resolved = [r for r in runs if r.get("resolution") is not None]
    print(f"Resolved runs: {len(resolved)}")

    if not resolved:
        print("No resolved runs found.")
        return 0

    updated = 0
    skipped = 0

    for run in resolved:
        gold_id = extract_gold_id(run)
        if not gold_id:
            q = (run.get("_body") or run.get("question", ""))[:60]
            print(f"  SKIP: no gold ID match — {q}")
            skipped += 1
            continue

        if gold_id not in MECHANISM_MAP:
            print(f"  SKIP: {gold_id} not in mechanism map")
            skipped += 1
            continue

        concept_rel, indicator, mechanism = MECHANISM_MAP[gold_id]
        concept_path = VAULT / concept_rel
        resolution = bool(run["resolution"])
        p_yes = run.get("p_yes", "?")

        print(f"\n{'+' if args.dry_run else '→'} {gold_id}: p={p_yes} res={resolution} → {mechanism}")

        if not args.dry_run:
            changed = update_calibration_table(concept_path, indicator, resolution)
            if changed:
                updated += 1
        else:
            q = (run.get("_body") or run.get("question", ""))[:60]
            print(f"  [DRY RUN] Would update: {concept_rel}")
            print(f"    indicator: {indicator}")
            print(f"    outcome: {'YES' if resolution else 'NO'}")

    print(f"\n── Summary ──")
    if args.dry_run:
        print(f"  Would update: {len(resolved)} runs")
    else:
        print(f"  Updated: {updated} calibration rows")
        print(f"  Skipped: {skipped} runs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
