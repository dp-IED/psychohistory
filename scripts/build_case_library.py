#!/usr/bin/env python3
"""Build structured case library from resolved Polymarket gold dataset.

Creates individual case files in graph-vault/cases/ with YAML frontmatter
and a Dataview-compatible _case_index.md in the vault root.

Each case file encodes: event_type, domain, resolution outcome, time horizon,
base rate context, and wikilinks to relevant vault threads/concepts/entities.

Usage:
  python scripts/build_case_library.py
  python scripts/build_case_library.py --gold data/polymarket/gold_branch_dataset.json
  python scripts/build_case_library.py --dry-run   # preview only, don't write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VAULT = ROOT / "graph-vault"
CASES_DIR = VAULT / "cases"
CASE_INDEX = VAULT / "_case_index.md"
DEFAULT_GOLD = ROOT / "data" / "polymarket" / "gold_branch_dataset.json"

# ── Domain classification ──────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "geopolitics": [
        "war", "conflict", "invasion", "military", "nuclear", "missile",
        "nato", "ukraine", "russia", "china", "iran", "israel", "gaza",
        "hamas", "ceasefire", "truce", "peace", "sanction", "taiwan",
        "hezbollah", "houthi", "syria", "iraq", "afghanistan",
    ],
    "politics": [
        "election", "president", "vote", "parliament", "congress",
        "senate", "minister", "party", "government", "impeach",
        "biden", "trump", "nominee", "candidate", "ballot",
        "democratic", "republican", "supreme court", "governor",
        "mayor", "referendum", "duterte", "pope",
    ],
    "economics": [
        "gdp", "inflation", "cpi", "rate", "market", "price",
        "tariff", "trade", "recession", "fed", "stock", "bond",
        "currency", "debt", "interest rate", "fomc", "federal reserve",
        "etf", "bitcoin", "ethereum", "sec", "crypto",
    ],
    "technology": [
        "tiktok", "ban", "ai", "openai", "tesla", "musk",
        "regulation", "antitrust", "privacy", "data",
    ],
    "health": [
        "outbreak", "case", "disease", "virus", "infection",
        "epidemic", "hospital", "who", "cdc", "vaccine", "pandemic",
        "hantavirus", "covid", "ebola",
    ],
}

EVENT_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("ceasefire", r"ceasefire|truce|peace\s+(deal|agreement|treaty)|halt\s+in\s+military"),
    ("election", r"election|presidential\s+race|win\s+the\s+most\s+seats|vote|ballot|referendum"),
    ("resignation", r"drop\s+out|resign|step\s+down|withdraw\s+from\s+(presidential\s+)?race"),
    ("rate_decision", r"fed(?:eral\s+reserve)?\s+(?:cuts?|raises?|decreases?|increases?)|interest\s+rate|fomc|bps"),
    ("regulatory_approval", r"sec\s+(?:approves?|rejects?|delays?)|etf\s+(?:begins?|trading|approved|rejected)|ban(?:ned)?\s+in\s+the\s+us"),
    ("court_ruling", r"supreme\s+court|court\s+(?:ruling|decision|case)|trial|sentenced?|convict"),
    ("government_action", r"government\s+shutdown|debt\s+ceiling|budget|spending\s+bill"),
    ("military_strike", r"strike|attack|invasion|bomb(?:ing)?|military\s+(?:action|operation)"),
    ("legislative_action", r"bill\s+(?:pass|sign|veto)|legislat(?:ion|ive)|congress\s+(?:pass|approve)"),
    ("appointment", r"nominat(?:ion|ed|ee)|appoint|confirm(?:ation|ed)|vp\s+(?:nominee|pick|selection)"),
    ("macro_release", r"cpi|inflation|unemployment|gdp|jobs\s+report|economic\s+data"),
    ("other", r"."),  # catch-all
]


def classify_domain(question: str, case_id: str) -> str:
    """Map question to domain using keyword matching."""
    q = question.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        # Fallback: use expected_family or slug
        if "macro" in case_id.lower() or "fed" in case_id.lower():
            return "economics"
        if "ceasefire" in case_id.lower() or "war" in case_id.lower():
            return "geopolitics"
        return "politics"
    return best


def classify_event_type(question: str) -> str:
    """Map question to event type using regex patterns."""
    q = question.lower()
    for event_type, pattern in EVENT_TYPE_PATTERNS:
        if re.search(pattern, q):
            return event_type
    return "other"


def classify_output_type(record: dict[str, Any]) -> str:
    """Determine output type: binary, categorical, or numeric."""
    outcomes = record.get("outcomes", [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, TypeError):
            outcomes = []
    if not outcomes:
        return "binary"  # default
    # If outcomes are ["Yes", "No"] → binary
    if set(str(o).lower() for o in outcomes) == {"yes", "no"}:
        return "binary"
    # If outcomes are multiple named labels → categorical
    if len(outcomes) > 2:
        return "categorical"
    return "binary"


def compute_time_horizon(record: dict[str, Any]) -> int | None:
    """Compute time horizon in days from start/cutoff to end/resolution."""
    end_str = record.get("end_date") or record.get("closed_time") or ""
    start_str = record.get("start_date") or record.get("created_at") or ""

    def parse_dt(s: str) -> datetime | None:
        if not s:
            return None
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S+00",
            "%Y-%m-%d",
        ]:
            try:
                s_clean = re.sub(r"\.\d+Z$", "Z", s)
                return datetime.strptime(s_clean, fmt)
            except ValueError:
                continue
        # Try ISO format with timezone
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    start_dt = parse_dt(start_str)
    end_dt = parse_dt(end_str)
    if start_dt and end_dt:
        return (end_dt - start_dt).days
    return None


def extract_keywords(question: str, max_kw: int = 10) -> list[str]:
    """Extract meaningful keywords from question text."""
    stop = {
        "will", "there", "before", "after", "during", "this", "that",
        "with", "from", "into", "have", "been", "being", "market",
        "resolves", "resolution", "criteria", "shall", "question",
        "whether", "price", "above", "below", "yes", "no", "the",
        "and", "for", "of", "to", "in", "on", "by", "at", "a", "an",
        "is", "are", "was", "were", "be", "or", "not", "it", "its",
        "what", "who", "when", "where", "how", "which", "their",
        "they", "has", "had", "does", "did", "can", "could", "would",
        "should", "may", "might", "must", "first", "also", "other",
    }
    # Extract capitalized phrases and alphanumeric tokens
    tokens = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[a-z0-9]{3,}", question)
    keywords = []
    for t in tokens:
        t_lower = t.lower().strip()
        if t_lower not in stop and len(t_lower) > 2:
            keywords.append(t_lower)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:max_kw]


def link_vault_files(question: str, case_id: str, vault: Path) -> list[str]:
    """Find vault threads/concepts/entities relevant to this case via keyword match."""
    keywords = extract_keywords(question, max_kw=8)
    if not keywords:
        return []

    linked: list[str] = []
    search_dirs = [
        vault / "domains",
        vault / "timeline",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for md_file in search_dir.rglob("*.md"):
            if md_file.name.startswith("."):
                continue
            if md_file.name.startswith("_"):
                continue
            try:
                text = md_file.read_text(encoding="utf-8")[:3000].lower()
                hits = sum(1 for kw in keywords if kw.lower() in text)
                if hits >= 2:
                    rel = md_file.relative_to(vault)
                    linked.append(str(rel))
            except Exception:
                pass

    # Deduplicate and limit
    seen = set()
    result = []
    for path in sorted(linked, key=lambda p: len(p)):
        if path not in seen:
            seen.add(path)
            result.append(path)
        if len(result) >= 8:
            break
    return result


def build_case_entry(case_data: dict[str, Any], vault: Path) -> dict[str, Any]:
    """Convert one gold dataset case into a case library entry."""
    case_id = case_data["case_id"]
    record = case_data["record"]
    question = record.get("question", "").strip()
    description = record.get("description", "") or ""
    full_text = case_data.get("full_text", "")

    resolution = record.get("resolved_outcome", "")
    resolution_bool = resolution.lower() == "yes" if resolution else None

    domain = classify_domain(question, case_id)
    event_type = classify_event_type(question)
    output_type = classify_output_type(record)
    time_horizon = compute_time_horizon(record)

    end_date = record.get("end_date") or record.get("closed_time") or ""
    start_date = record.get("start_date") or record.get("created_at") or ""
    volume = record.get("volume", 0) or 0
    slug = record.get("slug", "")
    url = record.get("url", "")
    gamma_url = record.get("gamma_url", "")

    keywords = extract_keywords(full_text or question, max_kw=10)
    linked_vault = link_vault_files(question, case_id, vault)

    # Determine base_rate_context: all cases with same event_type + domain
    # This gets populated after all cases are built
    return {
        "case_id": case_id,
        "question": question,
        "description": description[:500],
        "full_text": full_text[:2000],
        "event_type": event_type,
        "domain": domain,
        "output_type": output_type,
        "resolution": resolution,
        "resolution_bool": resolution_bool,
        "time_horizon_days": time_horizon,
        "start_date": start_date,
        "end_date": end_date,
        "volume": volume,
        "slug": slug,
        "url": url,
        "gamma_url": gamma_url,
        "keywords": keywords,
        "linked_vault_files": linked_vault,
        "expected_family": case_data.get("expected_family", ""),
        "hardness_score": case_data.get("hardness_score", 0),
    }


def compute_reference_class_stats(
    cases: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute base rate stats for each (event_type, domain) reference class.

    Returns: {(event_type, domain): {base_rate, total, resolved_yes, resolved_no}}
    """
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for c in cases:
        key = (c["event_type"], c["domain"])
        if key not in stats:
            stats[key] = {"total": 0, "resolved_yes": 0, "resolved_no": 0, "cases": []}
        stats[key]["total"] += 1
        stats[key]["cases"].append(c["case_id"])
        if c["resolution_bool"] is True:
            stats[key]["resolved_yes"] += 1
        elif c["resolution_bool"] is False:
            stats[key]["resolved_no"] += 1

    for key, s in stats.items():
        if s["total"] > 0:
            s["base_rate"] = s["resolved_yes"] / s["total"]
        else:
            s["base_rate"] = 0.5

    return stats


# ── File writers ────────────────────────────────────────────────────


def write_case_file(entry: dict[str, Any], ref_stats: dict[str, Any], target_dir: Path) -> Path:
    """Write a single case file with YAML frontmatter to cases/<case_id>.md."""
    case_id = entry["case_id"]
    key = (entry["event_type"], entry["domain"])
    ref = ref_stats.get(key, {"base_rate": 0.5, "total": 0})

    # Build frontmatter
    fm_lines = [
        "---",
        f"case_id: \"{case_id}\"",
        f"event_type: \"{entry['event_type']}\"",
        f"domain: \"{entry['domain']}\"",
        f"output_type: \"{entry['output_type']}\"",
        f"resolution: \"{entry['resolution']}\"",
        f"resolution_bool: {str(entry['resolution_bool']).lower()}",
        f"time_horizon_days: {entry['time_horizon_days'] or 'null'}",
        f"volume: {entry['volume']:.0f}",
        f"slug: \"{entry['slug']}\"",
        f"reference_class_base_rate: {ref.get('base_rate', 0.5):.3f}",
        f"reference_class_size: {ref.get('total', 0)}",
    ]
    if entry["start_date"]:
        fm_lines.append(f"start_date: \"{entry['start_date']}\"")
    if entry["end_date"]:
        fm_lines.append(f"end_date: \"{entry['end_date']}\"")
    if entry["hardness_score"]:
        fm_lines.append(f"hardness_score: {entry['hardness_score']}")
    if entry["keywords"]:
        kw_str = ", ".join(entry["keywords"])
        fm_lines.append(f"keywords: [{kw_str}]")
    fm_lines.append(f"tags: [case, resolved, {entry['event_type']}, {entry['domain']}]")
    fm_lines.append("---")

    # Build body
    body_lines = [
        "",
        f"# {entry['question'][:80]}",
        "",
        f"**Resolution:** {entry['resolution']}",
        f"**Event type:** {entry['event_type']} | **Domain:** {entry['domain']}",
        "",
    ]
    if entry["time_horizon_days"]:
        body_lines.append(f"**Time horizon:** {entry['time_horizon_days']} days")
    body_lines.append(f"**Volume:** ${entry['volume']:,.0f}")
    body_lines.append("")

    if entry["description"]:
        body_lines.append("## Resolution Criteria")
        body_lines.append("")
        body_lines.append(entry["description"])
        body_lines.append("")

    if entry["linked_vault_files"]:
        body_lines.append("## Linked Vault Files")
        body_lines.append("")
        for vf in entry["linked_vault_files"]:
            body_lines.append(f"- [[{vf}]]")
        body_lines.append("")

    body_lines += [
        "## Reference Class Context",
        "",
        f"- **Base rate for {entry['event_type']}/{entry['domain']}:** "
        f"{ref.get('base_rate', 0.5):.1%} ({ref.get('resolved_yes', 0)} YES / "
        f"{ref.get('total', 0)} total)",
        f"- **Sibling cases:** {', '.join(ref.get('cases', [])[:10])}",
        "",
        "## Source",
        "",
        f"- [Polymarket]({entry['url']})",
        f"- [Gamma API]({entry['gamma_url']})",
    ]

    content = "\n".join(fm_lines) + "\n".join(body_lines)
    filepath = target_dir / f"{case_id}.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return filepath


def write_case_index(
    cases: list[dict[str, Any]],
    ref_stats: dict[tuple[str, str], dict[str, Any]],
    target: Path,
) -> None:
    """Write _case_index.md — Dataview-compatible index of all resolved cases.

    Uses Dataview TABLE syntax for queryability in Obsidian.
    """
    lines = [
        "---",
        "type: index",
        "tags: [meta, case-library, dataview]",
        "date: " + datetime.now().strftime("%Y-%m-%d"),
        "purpose: \"Structured case library for outside-view base rate anchoring\"",
        "---",
        "",
        "# Case Library",
        "",
        "Resolved Polymarket cases with structured metadata. Queryable via Dataview.",
        f"**{len(cases)} cases** across {len(ref_stats)} reference classes.",
        "",
        "## Reference Class Summary",
        "",
        "| Event Type | Domain | N | Base Rate | YES | NO |",
        "|------------|--------|---|-----------|-----|----|",
    ]

    for (event_type, domain), stats in sorted(ref_stats.items()):
        lines.append(
            f"| {event_type} | {domain} | {stats['total']} | "
            f"{stats['base_rate']:.1%} | {stats['resolved_yes']} | "
            f"{stats['resolved_no']} |"
        )

    lines += [
        "",
        "## All Cases",
        "",
    ]

    # Dataview table
    lines.append("```dataview")
    lines.append("TABLE resolution, event_type, domain, time_horizon_days, reference_class_base_rate")
    lines.append('FROM "cases"')
    lines.append("SORT domain ASC, event_type ASC")
    lines.append("```")
    lines.append("")

    # Per-domain breakdowns
    domains = sorted(set(c["domain"] for c in cases))
    for domain in domains:
        domain_cases = [c for c in cases if c["domain"] == domain]
        lines.append(f"## {domain.title()} ({len(domain_cases)} cases)")
        lines.append("")
        for c in sorted(domain_cases, key=lambda x: x["event_type"]):
            res_icon = "✅" if c["resolution_bool"] else "❌" if c["resolution_bool"] is not None else "⬜"
            lines.append(
                f"- {res_icon} [[cases/{c['case_id']}|{c['question'][:80]}]] "
                f"`{c['event_type']}` {c.get('time_horizon_days', '?')}d"
            )
        lines.append("")

    target.write_text("\n".join(lines), encoding="utf-8")


def build_library(
    gold_path: Path,
    vault: Path,
    *,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Main: read gold dataset, build case files + index.

    Returns (cases, ref_stats) for downstream use by outside_view.py.
    """
    if not gold_path.exists():
        print(f"ERROR: Gold dataset not found at {gold_path}", file=sys.stderr)
        sys.exit(1)

    with open(gold_path) as f:
        data = json.load(f)

    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        print(f"ERROR: Expected list of cases, got {type(raw_cases)}", file=sys.stderr)
        sys.exit(1)

    print(f"Building case library from {len(raw_cases)} gold cases...")
    cases: list[dict[str, Any]] = []
    for raw in raw_cases:
        try:
            entry = build_case_entry(raw, vault)
            cases.append(entry)
        except Exception as e:
            print(f"  WARNING: Failed to build case {raw.get('case_id', '?')}: {e}")

    # Compute reference class statistics
    ref_stats = compute_reference_class_stats(cases)

    # Print summary
    print(f"\nBuilt {len(cases)} case entries across {len(ref_stats)} reference classes:")
    for (event_type, domain), stats in sorted(ref_stats.items()):
        print(
            f"  {event_type}/{domain}: {stats['total']} cases, "
            f"base_rate={stats['base_rate']:.1%} "
            f"({stats['resolved_yes']}Y/{stats['resolved_no']}N)"
        )

    if dry_run:
        print("\n[Dry run — no files written]")
        return cases, ref_stats

    # Write individual case files
    cases_dir = vault / "cases"
    if cases_dir.exists():
        import shutil
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)

    for entry in cases:
        filepath = write_case_file(entry, ref_stats, cases_dir)
        print(f"  Wrote {filepath.relative_to(vault)}")

    # Write index
    write_case_index(cases, ref_stats, CASE_INDEX)
    print(f"  Wrote {CASE_INDEX.relative_to(vault)}")
    print(f"\nDone. {len(cases)} case files + index written to {vault}/")

    return cases, ref_stats


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured case library from gold dataset")
    parser.add_argument(
        "--gold", type=Path, default=DEFAULT_GOLD,
        help="Path to gold dataset JSON",
    )
    parser.add_argument(
        "--vault", type=Path, default=VAULT,
        help="Path to graph-vault directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview classification without writing files",
    )
    args = parser.parse_args()

    build_library(args.gold, args.vault, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
