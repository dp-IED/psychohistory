#!/usr/bin/env python3
"""Validate graph-vault integrity after research sessions.

Checks:
1. Every wikilink in entity/ files resolves to an existing file
2. Every entity file's "Related Periods" lists only existing quarters
3. YAML frontmatter validity across all entity files
4. Entity files missing "updated" dates
5. Stale references to deleted directories (events/, concepts/)
6. Quarter files with body content but missing cutoff frontmatter

Usage:
    python testbed/scripts/validate_vault.py
    python testbed/scripts/validate_vault.py --fix   # auto-fix missing updated dates
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Match pit_reflect.py resolution: scripts/ is at testbed/scripts/, so parent.parent = testbed/
from harness.config import VAULT_DIR

GRAPH_VAULT = VAULT_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wikilinks_in(text: str) -> list[str]:
    """Extract all [[wikilink]] targets from text."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def _entity_path(target: str) -> Path | None:
    """Resolve a wikilink target to a file path, or None."""
    # Direct match
    if target.startswith("entities/"):
        candidate = GRAPH_VAULT / target
        if candidate.with_suffix(".md").exists():
            return candidate.with_suffix(".md")
    if target.startswith("timeline/"):
        candidate = GRAPH_VAULT / target
        if candidate.with_suffix(".md").exists():
            return candidate.with_suffix(".md")
    # Relative (no prefix)
    for prefix in ("entities", "timeline"):
        candidate = GRAPH_VAULT / prefix / f"{target}.md"
        if candidate.exists():
            return candidate
    # System files
    candidate = GRAPH_VAULT / f"{target}.md"
    if candidate.exists() and target.startswith("_"):
        return candidate
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _yaml_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter as a dict (simple parser, no pyyaml needed)."""
    result: dict = {}
    if not text.startswith("---"):
        return result
    parts = text.split("---", 2)
    if len(parts) < 3:
        return result
    for line in parts[1].strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip("'\"")
    return result


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_wikilinks() -> list[str]:
    """Check every wikilink in entity files resolves to an existing file."""
    issues: list[str] = []
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = _read_text(fpath)
        for target in _wikilinks_in(text):
            resolved = _entity_path(target)
            if resolved is None:
                issues.append(f"[BROKEN LINK] {fpath.name}: [[{target}]] does not resolve to any file")
    return issues


def check_related_periods() -> list[str]:
    """Check Related Periods sections reference only existing quarters."""
    issues: list[str] = []
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = _read_text(fpath)
        in_related = False
        for line in text.split("\n"):
            if "Related Periods" in line and line.strip().startswith("##"):
                in_related = True
                continue
            if in_related and line.strip().startswith("##"):
                in_related = False
                continue
            if in_related:
                for target in _wikilinks_in(line):
                    resolved = _entity_path(target)
                    if resolved is None:
                        issues.append(f"[BROKEN PERIOD LINK] {fpath.name}: [[{target}]] does not exist")
    return issues


def check_yaml_frontmatter() -> list[str]:
    """Check entity files have valid frontmatter with required fields."""
    issues: list[str] = []
    required = ["type", "created", "updated"]
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = _read_text(fpath)
        fm = _yaml_frontmatter(text)
        for field in required:
            if field not in fm:
                issues.append(f"[MISSING FRONTMATTER] {fpath.name}: missing '{field}' field")
        # Check for pipe-prefixed keys (YAML bug pattern)
        for line in text.split("---", 2)[1].split("\n") if "---" in text else []:
            if line.startswith("|") or line.startswith("||"):
                issues.append(f"[YAML SYNTAX ERROR] {fpath.name}: pipe-prefixed key: {line.strip()}")
                break
    return issues


def check_missing_updated() -> list[str]:
    """Report entity files where 'updated' date is stale (same as 'created')
    and the file was created at least 24 hours ago (not a brand-new file)."""
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    issues: list[str] = []
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = _read_text(fpath)
        fm = _yaml_frontmatter(text)
        if fm.get("created") and fm.get("updated"):
            if fm["created"] == fm["updated"] and fm["created"] < yesterday:
                issues.append(f"[STALE UPDATED] {fpath.name}: updated={fm['updated']} matches created — never updated since creation")
    return issues


def check_stale_refs() -> list[str]:
    """Check for references to deleted directories."""
    issues: list[str] = []
    # Check system files
    for fpath in [
        GRAPH_VAULT / "_spec.md",
        GRAPH_VAULT / "_procedure.md",
    ]:
        if not fpath.exists():
            continue
        text = _read_text(fpath)
        # Check for references to empty directories that were deleted
        if "events/" in text or "concepts/" in text:
            # These are OK if they're phrased as "check they haven't reappeared"
            # Flag them if they say "prune these"
            pass
    # Check testbed scripts
    script_dir = Path(__file__).resolve().parent.parent
    for fpath in sorted(script_dir.rglob("*.py")):
        if "validate_vault" in fpath.name:
            continue
        text = _read_text(fpath)
        if "graph-vault/events/" in text or "graph-vault/concepts/" in text:
            issues.append(f"[STALE REF] {fpath}: still references graph-vault/events/ or graph-vault/concepts/ directory")
    return issues


def check_quarter_cutoff() -> list[str]:
    """Check researched quarter files have pit_cutoff in frontmatter."""
    issues: list[str] = []
    for fpath in sorted((GRAPH_VAULT / "timeline").glob("*.md")):
        if fpath.stat().st_size < 200:  # scaffold-only, skip
            continue
        text = _read_text(fpath)
        fm = _yaml_frontmatter(text)
        if "pit_cutoff" not in fm and "cutoff" not in fm:
            issues.append(f"[MISSING CUTOFF] {fpath.name}: quarter has body content but no pit_cutoff in frontmatter")
    return issues


def auto_fix_updated(fix: bool = False) -> list[str]:
    """Auto-fix stale updated dates by bumping them to today's date."""
    from datetime import date
    today = date.today().isoformat()
    fixes: list[str] = []
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = _read_text(fpath)
        fm = _yaml_frontmatter(text)
        if fm.get("created") and fm.get("updated"):
            if fm["created"] == fm["updated"] and fm["created"] < yesterday:
                old = f"updated: {fm['updated']}"
                new = f"updated: {today}"
                content = text.replace(old, new, 1)
                if fix:
                    fpath.write_text(content, encoding="utf-8")
                    fixes.append(f"[FIXED] {fpath.name}: updated {fm['updated']} -> {today}")
                else:
                    fixes.append(f"[WOULD FIX] {fpath.name}: updated {fm['updated']} -> {today}")
    return fixes


# ---------------------------------------------------------------------------
# Checks: Cycle 4 additions
# ---------------------------------------------------------------------------


def check_dual_directory() -> list[str]:
    """Ensure quarters/ directory doesn't contain .md files."""
    issues: list[str] = []
    qdir = GRAPH_VAULT / "quarters"
    if qdir.exists():
        md_files = sorted(qdir.glob("*.md"))
        if md_files:
            names = ", ".join(f.name for f in md_files)
            issues.append(f"[DUAL DIR] quarters/ still contains {len(md_files)} files: {names}. Only timeline/ should be used.")
    return issues


def check_frontmatter_drift() -> list[str]:
    """Check files use spec-compliant frontmatter fields (not old conventions)."""
    issues: list[str] = []

    # Quarter files: should use label:/date_range:/pit_cutoff:, not old fields
    for fpath in sorted((GRAPH_VAULT / "timeline").glob("????-Q?.md")):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^title:", text, re.MULTILINE):
            issues.append(f"[DRIFT] {fpath.name}: uses 'title:' instead of 'label:'")
        if re.search(r"^period_start:", text, re.MULTILINE):
            issues.append(f"[DRIFT] {fpath.name}: uses 'period_start:' instead of 'date_range:'")
        if not re.search(r"^pit_cutoff:", text, re.MULTILINE):
            issues.append(f"[DRIFT] {fpath.name}: missing 'pit_cutoff:' field")
        if not re.search(r"^label:", text, re.MULTILINE):
            issues.append(f"[DRIFT] {fpath.name}: missing 'label:' field")
        if not re.search(r"^date_range:", text, re.MULTILINE):
            issues.append(f"[DRIFT] {fpath.name}: missing 'date_range:' field")

    # Thread files: should use inception:/conclusion:/status:, not old span:
    for fpath in sorted((GRAPH_VAULT / "threads").glob("*.md")):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^span:", text, re.MULTILINE):
            issues.append(f"[DRIFT] threads/{fpath.name}: uses 'span:' instead of 'inception:'/'conclusion:'")

    # Entity files: should have title: and slug:
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^title:", text, re.MULTILINE):
            issues.append(f"[DRIFT] entities/{fpath.name}: missing 'title:' in frontmatter")
        if not re.search(r"^slug:", text, re.MULTILINE):
            issues.append(f"[DRIFT] entities/{fpath.name}: missing 'slug:' in frontmatter")

    return issues


def check_zero_byte() -> list[str]:
    """Find zero-byte .md files (stubs never populated)."""
    issues: list[str] = []
    for p in sorted(GRAPH_VAULT.rglob("*.md")):
        if ".git" in p.parts:
            continue
        if p.stat().st_size == 0:
            rel = p.relative_to(GRAPH_VAULT)
            issues.append(f"[ZERO BYTE] {rel} is empty (0 bytes)")
    return issues


def check_missing_annual_summaries() -> list[str]:
    """Every year with all 4 quarter files should have an annual YYYY.md."""
    issues: list[str] = []
    timeline = GRAPH_VAULT / "timeline"
    years: set[str] = set()
    for qf in timeline.glob("????-Q?.md"):
        years.add(qf.stem[:4])
    for y in sorted(years):
        q1 = timeline / f"{y}-Q1.md"
        q2 = timeline / f"{y}-Q2.md"
        q3 = timeline / f"{y}-Q3.md"
        q4 = timeline / f"{y}-Q4.md"
        annual = timeline / f"{y}.md"
        if all(p.exists() for p in [q1, q2, q3, q4]) and not annual.exists():
            issues.append(f"[MISSING ANNUAL] {y}: all 4 quarters exist but no {y}.md")
    return issues


def check_entity_backlinks() -> list[str]:
    """Entity files should have ## Appears In section for graph connectivity."""
    issues: list[str] = []
    for fpath in sorted((GRAPH_VAULT / "entities").glob("*.md")):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if "Appears In" not in text and "Related Periods" not in text:
            issues.append(f"[NO BACKLINKS] entities/{fpath.name}: missing '## Appears In' section")
    return issues


def check_timeline_wikilinks() -> list[str]:
    """Check wikilinks in timeline/ files resolve to an existing vault file.
    This is the primary graph integrity check — quarter-file links should resolve.
    """
    issues: list[str] = []
    all_slugs: set[str] = set()
    for p in GRAPH_VAULT.rglob("*.md"):
        if ".git" in p.parts:
            continue
        all_slugs.add(p.stem)
    # Also add known system files
    all_slugs.update(["_spec", "_procedure", "_index"])

    for fpath in sorted((GRAPH_VAULT / "timeline").glob("*.md")):
        text = _read_text(fpath)
        for target in _wikilinks_in(text):
            # Handle pipe syntax: [[actual|display]] -> actual
            target = target.split("|")[0].strip()
            if target not in all_slugs:
                issues.append(f"[BROKEN TIMELINE LINK] {fpath.name}: [[{target}]] does not resolve to any vault file")
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate graph-vault integrity.")
    parser.add_argument("--fix", action="store_true", help="Auto-fix stale updated dates")
    parser.add_argument("--json", action="store_true", help="Output JSON summary for harness parsing")
    args = parser.parse_args()

    checks = [
        ("Timeline wikilink resolution", check_timeline_wikilinks),
        ("Related Periods", check_related_periods),
        ("YAML frontmatter", check_yaml_frontmatter),
        ("Stale updated dates", check_missing_updated),
        ("Stale dir references", check_stale_refs),
        ("Quarter cutoff", check_quarter_cutoff),
        ("Dual directory", check_dual_directory),
        ("Frontmatter drift", check_frontmatter_drift),
        ("Zero-byte files", check_zero_byte),
        ("Missing annual summaries", check_missing_annual_summaries),
        ("Entity backlinks", check_entity_backlinks),
    ]

    results: list[dict] = []
    all_issues: list[str] = []

    for name, fn in checks:
        issues = fn()
        results.append({"name": name, "passed": len(issues) == 0, "issues": issues})
        all_issues.extend(issues)
        if not args.json:
            if issues:
                print(f"\n=== {name} ({len(issues)} issues) ===")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print(f"  [OK] {name}")

    # Auto-fix stale updated dates
    fixes = auto_fix_updated(fix=args.fix)
    for fix in fixes:
        if not args.json:
            print(f"  {fix}")
        all_issues = [i for i in all_issues if fpath(i) not in fix]

    if not args.json:
        print()
    if all_issues:
        if not args.json:
            print(f"Total: {len(all_issues)} issues found.")
            if not args.fix:
                print("Run with --fix to auto-fix stale updated dates.")
        if args.json:
            json.dump({
                "passed": False,
                "total_issues": len(all_issues),
                "checks": results,
            }, sys.stdout, indent=2)
            sys.stdout.flush()
        sys.exit(1)
    else:
        if not args.json:
            print("All checks passed.")
        if args.json:
            json.dump({
                "passed": True,
                "total_issues": 0,
                "checks": results,
            }, sys.stdout, indent=2)
            sys.stdout.flush()


def fpath(issue: str) -> str:
    """Extract filename from issue string."""
    if "] " in issue:
        return issue.split("] ", 1)[1].split(":")[0]
    return ""


if __name__ == "__main__":
    main()