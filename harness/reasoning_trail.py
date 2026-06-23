"""Parse agent reasoning trails from vault run notes.

Extracts structured features from the markdown reasoning trail that
the agent already writes: threads consulted, mechanisms invoked,
concepts referenced, sources used.

This enables per-feature calibration diagnostics: does invoking
procedural-certainty actually improve forecasts?
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


@dataclass
class ReasoningTrail:
    """Structured extract of an agent's reasoning process."""
    run_id: str
    question: str = ""
    p_yes: float = 0.0
    resolution: bool | None = None
    brier: float | None = None
    cutoff: date | None = None
    category: str = ""

    # Vault assets referenced in reasoning
    threads: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    # Polymarket context for comparison
    pm_tags: list[str] = field(default_factory=list)

    # Linking to calibration
    polymarket_slug: str = ""


def parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter from markdown."""
    match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def extract_list_items(text: str, header: str) -> list[str]:
    """Extract bullet list items from a markdown section."""
    # Find the section
    pattern = rf'{header}[:\s]*\n(.*?)(?=\n\S|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    items = []
    for line in match.group(1).strip().split('\n'):
        # Match `- item` or `  - item`
        m = re.match(r'\s*[-*]\s+(.+)', line)
        if m:
            item = m.group(1).strip()
            # Clean up trailing punctuation
            item = re.sub(r'[.,;:]$', '', item)
            items.append(item)
    return items


def parse_run(filepath: Path, resolutions: dict[str, bool] | None = None) -> ReasoningTrail | None:
    """Parse a single run note into a structured reasoning trail.
    
    Args:
        filepath: Path to the run markdown file
        resolutions: dict mapping question_id → resolution (from gold dataset)
    """
    text = filepath.read_text(encoding='utf-8')
    fm = parse_frontmatter(text)

    qid = fm.get('question_id', '')
    question = fm.get('question_text', '')
    if not question:
        # Extract question from body (first non-empty line after frontmatter)
        body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
        for line in body.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('>'):
                question = line
                break

    trail = ReasoningTrail(
        run_id=filepath.stem,
        question=question,
        p_yes=float(fm.get('p_yes', 0)),
        cutoff=_parse_date(fm.get('cutoff')),
        category=fm.get('category', ''),
    )

    # Resolve outcome from gold dataset
    if resolutions and qid:
        # Try exact match and fuzzy match
        if qid in resolutions:
            trail.resolution = resolutions[qid]
        else:
            # Try matching gold_XX pattern
            import re as _re
            m = _re.search(r'gold[_-](\d+)', qid)
            if m:
                gold_num = int(m.group(1))
                for rid, res in resolutions.items():
                    if f'gold_{gold_num:02d}' in rid or f'gold_{gold_num}' in rid:
                        trail.resolution = res
                        break

    # Extract from PIT Context section
    trail.threads = extract_list_items(text, r'Active threads')
    trail.mechanisms = extract_list_items(text, r'Mechanisms\s*/\s*concepts')
    trail.events = extract_list_items(text, r'Key events')
    trail.sources = extract_list_items(text, r'Sources')

    # Extract concepts from cross-references
    trail.concepts = _extract_wikilinks(text, 'concepts')

    return trail


def _extract_wikilinks(text: str, category: str = 'concepts') -> list[str]:
    """Extract [[wikilinks]] matching a category path."""
    items = []
    pattern = rf'\[\[(?:.*?/)?{category}/([^\]]+)\]\]'
    for match in re.finditer(pattern, text):
        items.append(match.group(1).strip())
    return items


def _parse_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except (ValueError, TypeError):
        return None


def parse_all_runs(vault_dir: str | Path,
                   resolutions: dict[str, bool] | None = None) -> list[ReasoningTrail]:
    """Parse all run notes in graph-vault/runs/."""
    runs_dir = Path(vault_dir) / "runs"
    if not runs_dir.exists():
        return []

    trails = []
    for f in sorted(runs_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        trail = parse_run(f, resolutions=resolutions)
        if trail and trail.question:
            trails.append(trail)

    return trails


# ── Diagnostics ────────────────────────────────────────────────────────

def mechanism_diagnostics(trails: list[ReasoningTrail]) -> dict:
    """Per-mechanism: does invoking this mechanism improve accuracy?"""
    mech_counts: dict[str, dict] = defaultdict(lambda: {
        "invoked": 0, "yes": 0, "no": 0,
        "total_brier": 0.0, "total_p_yes": 0.0,
    })

    for t in trails:
        for mech in t.mechanisms:
            name = mech.split(':')[0].strip().lower()
            mech_counts[name]["invoked"] += 1
            if t.resolution is True:
                mech_counts[name]["yes"] += 1
            elif t.resolution is False:
                mech_counts[name]["no"] += 1
            if t.brier is not None:
                mech_counts[name]["total_brier"] += t.brier
            mech_counts[name]["total_p_yes"] += t.p_yes

    results = {}
    for name, c in mech_counts.items():
        n = c["yes"] + c["no"]
        if n == 0:
            continue
        hit_rate = c["yes"] / n if n > 0 else 0
        avg_brier = c["total_brier"] / n if n > 0 else 0
        avg_p = c["total_p_yes"] / n if n > 0 else 0
        results[name] = {
            "invoked": c["invoked"],
            "resolved": n,
            "yes": c["yes"],
            "no": c["no"],
            "hit_rate": round(hit_rate, 3),
            "avg_brier": round(avg_brier, 4),
            "avg_p_yes": round(avg_p, 3),
        }

    return dict(sorted(results.items(), key=lambda x: -x[1]["resolved"]))


def thread_diagnostics(trails: list[ReasoningTrail]) -> dict:
    """Per-thread: which vault threads correlate with accuracy?"""
    thread_counts: dict[str, dict] = defaultdict(lambda: {
        "consulted": 0, "yes": 0, "no": 0, "total_brier": 0.0,
    })

    for t in trails:
        for thread in t.threads:
            # Normalize: strip _thread suffix, use short name
            name = thread.replace('_thread', '').strip().rstrip('/')
            name = name.split('/')[-1] if '/' in name else name
            thread_counts[name]["consulted"] += 1
            if t.resolution is True:
                thread_counts[name]["yes"] += 1
            elif t.resolution is False:
                thread_counts[name]["no"] += 1

    results = {}
    for name, c in thread_counts.items():
        n = c["yes"] + c["no"]
        if n == 0:
            continue
        results[name] = {
            "consulted": c["consulted"],
            "resolved": n,
            "hit_rate": round(c["yes"] / n, 3) if n > 0 else 0,
        }

    return dict(sorted(results.items(), key=lambda x: -x[1]["resolved"]))


def summary(trails: list[ReasoningTrail]) -> str:
    """Human-readable diagnostic summary."""
    resolved = [t for t in trails if t.resolution is not None]
    unresolved = [t for t in trails if t.resolution is None]

    lines = [
        f"=== Reasoning Trail Diagnostics ===",
        f"Total runs:     {len(trails)}",
        f"Resolved:       {len(resolved)}",
        f"Unresolved:     {len(unresolved)}",
        f"",
    ]

    if resolved:
        correct = sum(1 for t in resolved
                      if (t.p_yes >= 0.5) == t.resolution)
        lines.append(f"Accuracy (>0.5): {correct}/{len(resolved)} "
                     f"({correct/len(resolved)*100:.1f}%)")

        avg_brier = sum(t.brier for t in resolved if t.brier is not None)
        n_brier = sum(1 for t in resolved if t.brier is not None)
        if n_brier > 0:
            lines.append(f"Avg Brier:       {avg_brier/n_brier:.4f}")

    lines.append("")
    lines.append("--- Mechanism Diagnostics ---")
    for name, d in mechanism_diagnostics(trails).items():
        lines.append(
            f"  {name:<40} n={d['resolved']:>2}  "
            f"hit={d['hit_rate']:.2f}  brier={d['avg_brier']:.4f}"
        )

    lines.append("")
    lines.append("--- Thread Diagnostics ---")
    for name, d in thread_diagnostics(trails).items():
        lines.append(
            f"  {name:<40} n={d['resolved']:>2}  "
            f"hit={d['hit_rate']:.2f}"
        )

    return "\n".join(lines)


def load_gold_resolutions(testbed_dir: str | Path) -> dict[str, bool]:
    """Load resolution data from gold datasets."""
    resolutions: dict[str, bool] = {}
    base = Path(testbed_dir)

    # Try pit_blind_test/results.json
    pit_path = base / "pit_blind_test" / "results.json"
    if pit_path.exists():
        data = json.loads(pit_path.read_text())
        for r in data.get("results", []):
            cid = r.get("case_id", "")
            expected = r.get("expected")
            if cid and expected is not None:
                resolutions[cid] = bool(expected)
                # Also store short gold_XX variants
                import re as _re
                m = _re.search(r'(gold_\d+)', cid)
                if m:
                    resolutions[m.group(1)] = bool(expected)

    # Try resolved_markets.jsonl for live Polymarket resolutions
    pm_path = base / "data" / "polymarket" / "resolved_markets.jsonl"
    if pm_path.exists():
        with open(pm_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                slug = rec.get("slug", "")
                if slug and rec.get("resolution") is not None:
                    resolutions[slug] = bool(rec["resolution"])

    return resolutions


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    vault = sys.argv[1] if len(sys.argv) > 1 else "graph-vault"
    testbed = sys.argv[2] if len(sys.argv) > 2 else "."
    resolutions = load_gold_resolutions(testbed)
    trails = parse_all_runs(vault, resolutions=resolutions)
    print(summary(trails))
