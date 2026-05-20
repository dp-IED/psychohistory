#!/usr/bin/env python3
"""
Thread Continuity Auditor (Gap 5)
----------------------------------
Scans vault threads, checks which timeline quarters they're linked in,
and flags stale threads.

Usage:
    python scripts/thread_continuity_audit.py
    python scripts/thread_continuity_audit.py --issues-only
    python scripts/thread_continuity_audit.py --json
    python scripts/thread_continuity_audit.py --auto-fix-stale
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VAULT_DIR = PROJECT_ROOT / "graph-vault"
THREADS_DIR = VAULT_DIR / "threads"
TIMELINE_DIR = VAULT_DIR / "timeline"

# ── Quarter helpers ────────────────────────────────────────────────────────

QUARTER_RE = re.compile(r"(\d{4}-Q[1-4])")
QUARTER_FILENAME_RE = re.compile(r"(\d{4}-Q[1-4])\.md$")


def quarter_to_end_date(q: str) -> date:
    """Return the last calendar day of a quarter-string like '2024-Q1'."""
    year = int(q[:4])
    qnum = int(q[6])
    month = qnum * 3  # 3, 6, 9, 12
    # last day of that month
    if month == 12:
        return date(year, 12, 31)
    # last day of month = first day of next month minus 1 day
    import calendar
    return date(year, month, calendar.monthrange(year, month)[1])


def quarter_to_start_date(q: str) -> date:
    """Return the first calendar day of a quarter-string like '2024-Q1'."""
    year = int(q[:4])
    qnum = int(q[6])
    month = qnum * 3 - 2  # 1, 4, 7, 10
    return date(year, month, 1)


def parse_frontmatter(text: str) -> dict:
    """Parse YAML-like frontmatter from a markdown file. Returns dict."""
    fm = {}
    # Find frontmatter between --- delimiters
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return fm
    body = m.group(1)
    # Simple line-based YAML parser for flat fields
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Remove quotes
            val = val.strip('"').strip("'")
            if val == "null" or val == "" or val == "~":
                val = None
            fm[key] = val
    return fm


def get_quarter_files() -> list[tuple[str, str]]:
    """Return list of (quarter_label, filepath) for all timeline quarter files."""
    results = []
    for fpath in TIMELINE_DIR.glob("*.md"):
        m = QUARTER_FILENAME_RE.search(str(fpath.name))
        if m:
            results.append((m.group(1), str(fpath)))
    # Sort by year then quarter
    results.sort(key=lambda x: (int(x[0][:4]), int(x[0][6])))
    return results


def extract_quarter_wikilinks(text: str) -> list[str]:
    """Find all [[YYYY-QN]] wikilinks in text (both [[Q]] and [[prefix/Q]] variants)."""
    links = set()
    # Match [[2024-Q1]] or [[timeline/2024-Q1]] or [[threads/...]] (we filter below)
    for m in re.finditer(r"\[\[([^\]]+)\]\]", text):
        link = m.group(1)
        # Extract quarter code from link
        qm = QUARTER_RE.search(link)
        if qm:
            links.add(qm.group(1))
    return sorted(links)


def compute_quarters_between(q1: str, q2: str) -> int:
    """Number of quarters between q1 and q2 inclusive as end-date comparison."""
    d1 = quarter_to_end_date(q1)
    d2 = quarter_to_end_date(q2)
    # rough quarter count based on month difference
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    return max(0, months // 3)


def main():
    parser = argparse.ArgumentParser(
        description="Thread Continuity Auditor — check thread linkage against timeline quarters"
    )
    parser.add_argument(
        "--issues-only",
        action="store_true",
        help="Only show threads with issues (omit OK section)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--auto-fix-stale",
        action="store_true",
        help="Set status=fading for stale active threads (gap >= 2 quarters)",
    )
    args = parser.parse_args()

    # ── 1. Gather all quarter files ──────────────────────────────────────
    quarter_files = get_quarter_files()
    if not quarter_files:
        print("ERROR: No timeline quarter files found in", TIMELINE_DIR, file=sys.stderr)
        sys.exit(1)

    quarter_labels = [q[0] for q in quarter_files]
    most_recent_quarter = quarter_labels[-1]
    most_recent_quarter_end = quarter_to_end_date(most_recent_quarter)

    # ── 2. Gather all thread files ───────────────────────────────────────
    thread_files = sorted(THREADS_DIR.glob("*.md"))
    if not thread_files:
        print("ERROR: No thread files found in", THREADS_DIR, file=sys.stderr)
        sys.exit(1)

    # ── 3. Pre-index: which quarters link to which threads? ──────────────
    # For each quarter file, find all thread wikilinks
    quarter_thread_links: dict[str, set[str]] = {q: set() for q in quarter_labels}
    for qlabel, qfpath in quarter_files:
        try:
            text = Path(qfpath).read_text(encoding="utf-8")
        except Exception as e:
            print(f"WARNING: Could not read {qfpath}: {e}", file=sys.stderr)
            continue
        # Find [[slug]] and [[threads/slug]] wikilinks
        for m in re.finditer(r"\[\[([^\]]+)\]\]", text):
            link = m.group(1)
            # Could be just slug, or threads/slug, or threads/slug|title
            link_clean = link.split("|")[0].strip()
            # Check if it references a thread
            # Matches: just a slug, or threads/slug
            for tf in thread_files:
                slug = tf.stem  # filename without .md
                if link_clean == slug or link_clean == f"threads/{slug}":
                    quarter_thread_links[qlabel].add(slug)
                    break

    # ── 4. Analyze each thread ───────────────────────────────────────────
    results = []

    for tfpath in thread_files:
        slug = tfpath.stem
        try:
            text = tfpath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"WARNING: Could not read {tfpath}: {e}", file=sys.stderr)
            continue

        fm = parse_frontmatter(text)

        status = fm.get("status", "active")  # absent → treat as active
        if status is None:
            status = "active"

        title = fm.get("title", slug)
        inception_str = fm.get("inception")
        conclusion_str = fm.get("conclusion")

        # ── 4a. Find linked quarters from thread body ────────────────────
        body = text  # includes frontmatter; that's fine, quarter links may appear in fm too
        body_quarter_links = extract_quarter_wikilinks(body)

        # ── 4b. Merge: also check which quarters link TO this thread ─────
        reverse_links = []
        for qlabel, linked_slugs in quarter_thread_links.items():
            if slug in linked_slugs:
                reverse_links.append(qlabel)

        all_linked = sorted(set(body_quarter_links + reverse_links))

        # ── 4c. Determine last linked quarter ────────────────────────────
        if all_linked:
            last_linked = all_linked[-1]  # sorted, so last = most recent
            last_linked_end = quarter_to_end_date(last_linked)
        else:
            last_linked = None
            last_linked_end = None

        # ── 4d. Compute gap ──────────────────────────────────────────────
        if last_linked_end and last_linked != most_recent_quarter:
            gap = compute_quarters_between(last_linked, most_recent_quarter)
        elif last_linked == most_recent_quarter:
            gap = 0
        else:
            gap = None  # no linkages at all

        # ── 4e. Audit logic ──────────────────────────────────────────────
        flags = []
        fix_needed = False

        # "superseded" is not in spec — treat like resolved
        effective_status = status
        if status == "superseded":
            effective_status = "resolved"

        if status == "active" or status is None:
            if gap is not None and gap >= 2:
                flags.append(("STALE", "severe", f"Gap of {gap} quarters — UPDATE REQUIRED"))
                fix_needed = True
        elif status == "fading":
            if gap is not None and gap >= 3:
                flags.append(("STALE", "moderate", f"Gap of {gap} quarters — check if still alive"))
        elif status == "resolved" or status == "superseded":
            # Check that the concluding quarter is (or includes) the last linked quarter
            if conclusion_str and conclusion_str != "null":
                try:
                    conclusion_date = datetime.strptime(conclusion_str, "%Y-%m-%d").date()
                    # Determine which quarter the conclusion falls in
                    for ql in reversed(quarter_labels):
                        q_start = quarter_to_start_date(ql)
                        q_end = quarter_to_end_date(ql)
                        if q_start <= conclusion_date <= q_end:
                            concluding_quarter = ql
                            break
                    else:
                        concluding_quarter = None

                    if concluding_quarter and last_linked and concluding_quarter != last_linked:
                        # Check if last_linked is before concluding_quarter
                        last_q_end = quarter_to_end_date(last_linked)
                        if last_q_end < conclusion_date:
                            flags.append(("INCONSISTENT", "moderate",
                                f"Conclusion={conclusion_str} (quarter={concluding_quarter}) "
                                f"but last linked quarter={last_linked} — thread not linked "
                                f"from any quarter file around its conclusion"))
                except ValueError:
                    pass
        elif status == "absent":
            flags.append(("NO_STATUS", "minor", "Thread has no status field — suggest setting one"))

        # Additional: if status is active and gap is 1, that's a warning but not a flag
        # Check if thread has no status at all (absent)
        if fm.get("status") is None:
            flags.append(("NO_STATUS", "minor", "Thread has no status field — suggest setting one"))
            fix_needed = True

        results.append({
            "slug": slug,
            "title": title,
            "status": status,
            "linked_quarters": all_linked,
            "last_linked": last_linked,
            "most_recent_quarter": most_recent_quarter,
            "gap": gap,
            "flags": flags,
            "fix_needed": fix_needed,
            "filepath": str(tfpath),
        })

    # ── 5. Auto-fix stale ───────────────────────────────────────────────
    if args.auto_fix_stale:
        fixed = 0
        for r in results:
            if r["fix_needed"] and r["status"] == "active":
                # Set status=fading
                fpath = Path(r["filepath"])
                text = fpath.read_text(encoding="utf-8")
                # Replace status: active with status: fading
                new_text = re.sub(
                    r"^status:\s*active\b",
                    "status: fading",
                    text,
                    count=1,
                    flags=re.MULTILINE,
                )
                if new_text != text:
                    fpath.write_text(new_text, encoding="utf-8")
                    print(f"  → Fixed: {r['slug']}: status=active → status=fading")
                    fixed += 1
                else:
                    print(f"  ✗ Failed to fix: {r['slug']} (status line not found)")
        print(f"\nAuto-fix complete: {fixed} thread(s) updated.")
        return

    # ── 6. Output ────────────────────────────────────────────────────────
    if args.json:
        output = {
            "audit_date": str(date.today()),
            "scanned": len(results),
            "most_recent_quarter": most_recent_quarter,
            "threads": results,
        }
        print(json.dumps(output, indent=2))
        return

    # ── 6b. Human-readable output ────────────────────────────────────────
    print(f"Thread Continuity Audit — {date.today()}")
    print(f"Scanned {len(results)} threads")
    print(f"Most recent timeline quarter: {most_recent_quarter} (ends {most_recent_quarter_end})")
    print()

    # Categorize
    stale_active = [r for r in results if any(f[0] == "STALE" and f[1] == "severe" for f in r["flags"])]
    stale_fading = [r for r in results if any(f[0] == "STALE" and f[1] == "moderate" for f in r["flags"])]
    inconsistent = [r for r in results if any(f[0] == "INCONSISTENT" for f in r["flags"])]
    no_status = [r for r in results if any(f[0] == "NO_STATUS" for f in r["flags"])]

    ok_threads = [r for r in results if not r["flags"]]

    if stale_active:
        print("=== STALE ACTIVE THREADS (must fix) ===")
        for r in stale_active:
            flag_text = r["flags"][0][2] if r["flags"] else ""
            print(f"  {r['slug']} (status={r['status']})")
            print(f"    Last updated: {r['last_linked']} ({quarter_to_end_date(r['last_linked']) if r['last_linked'] else 'N/A'})")
            print(f"    Most recent quarter: {r['most_recent_quarter']} ({quarter_to_end_date(r['most_recent_quarter'])})")
            print(f"    Gap: {r['gap']} quarters — {flag_text}")
            print(f"    Linked quarters: {', '.join(r['linked_quarters']) if r['linked_quarters'] else 'none'}")
            print()

    if stale_fading:
        print("=== STALE FADING THREADS (check if still alive) ===")
        for r in stale_fading:
            flag_text = r["flags"][0][2] if r["flags"] else ""
            print(f"  {r['slug']} (status={r['status']})")
            print(f"    Last updated: {r['last_linked']} ({quarter_to_end_date(r['last_linked']) if r['last_linked'] else 'N/A'})")
            print(f"    Most recent quarter: {r['most_recent_quarter']} ({quarter_to_end_date(r['most_recent_quarter'])})")
            print(f"    Gap: {r['gap']} quarters — {flag_text}")
            print()

    if inconsistent:
        print("=== INCONSISTENT RESOLVED THREADS ===")
        for r in inconsistent:
            flag_text = r["flags"][0][2] if r["flags"] else ""
            print(f"  {r['slug']} (status={r['status']})")
            print(f"    {flag_text}")
            print()

    if no_status:
        print("=== THREADS MISSING STATUS ===")
        for r in no_status:
            flag_text = r["flags"][0][2] if r["flags"] else ""
            print(f"  {r['slug']} — {flag_text}")
            print()

    if not args.issues_only:
        if ok_threads:
            print("=== THREADS WITH OK COVERAGE ===")
            for r in ok_threads:
                coverage = f"Linked in: {', '.join(r['linked_quarters']) if r['linked_quarters'] else 'none'}"
                ok_flag = " ✓" if r['linked_quarters'] else " (no quarter links)"
                print(f"  {r['slug']} (status={r['status']})")
                print(f"    {coverage}{ok_flag}")
                print()

    # Summary
    print(f"--- Summary ---")
    print(f"  Total threads: {len(results)}")
    print(f"  Stale active: {len(stale_active)}")
    print(f"  Stale fading: {len(stale_fading)}")
    print(f"  Inconsistent resolved: {len(inconsistent)}")
    print(f"  Missing status: {len(no_status)}")
    print(f"  OK coverage: {len(ok_threads)}")


if __name__ == "__main__":
    main()
