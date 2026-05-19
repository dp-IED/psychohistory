"""PIT research librarian: fetch only knowable-as-of-cutoff vault context for a question."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from harness.vault_pit import (
    format_admissible_block,
    list_admissible_paths,
    materialize_pit_snapshot,
    truncate_thread_for_pit,
)

_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 600
_MAX_CONTEXT_CHARS = 18_000
_MAX_PATHS = 14
_STOP = frozenset(
    "a an the and or for of to in on at by with from as is was were will be has have had "
    "that this which what when where who how not no yes".split()
)


@dataclass
class PitSearchResult:
    path: str
    excerpt: str
    score: float = 0.0


@dataclass
class PitSearchResponse:
    cutoff: date
    results: list[PitSearchResult]
    manifest_count: int = 0
    error: str | None = None


@dataclass
class PitResearchBrief:
    cutoff: str
    conjuncture: str = ""
    key_events: list[str] = field(default_factory=list)
    active_threads: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    excluded_as_post_cutoff: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        lines = [
            "=== PIT RESEARCH BRIEF (librarian sub-agent) ===",
            f"Cutoff: {self.cutoff}",
            f"Conjuncture: {self.conjuncture}",
        ]
        if self.key_events:
            lines.append("Key events (≤ cutoff):")
            lines.extend(f"  - {e}" for e in self.key_events[:12])
        if self.active_threads:
            lines.append("Active threads:")
            lines.extend(f"  - {t}" for t in self.active_threads[:8])
        if self.mechanisms:
            lines.append("Mechanisms / concepts:")
            lines.extend(f"  - {m}" for m in self.mechanisms[:8])
        if self.uncertainties:
            lines.append("Still uncertain at cutoff (do not treat as resolved):")
            lines.extend(f"  - {u}" for u in self.uncertainties[:8])
        if self.excluded_as_post_cutoff:
            lines.append("Excluded or truncated (post-cutoff leakage prevented):")
            lines.extend(f"  - {x}" for x in self.excluded_as_post_cutoff[:6])
        if self.sources:
            lines.append(f"Sources ({len(self.sources)}): " + ", ".join(self.sources[:10]))
        return "\n".join(lines)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOP}


def rank_admissible_paths(
    question: str,
    manifest: list[str],
    *,
    max_paths: int = _MAX_PATHS,
) -> list[str]:
    """Score admissible paths by overlap with question tokens."""
    qtok = _tokenize(question)
    scored: list[tuple[float, str]] = []
    for rel in manifest:
        base = 0.0
        slug = Path(rel).stem.replace("-", " ")
        blob = f"{rel} {slug}".lower()
        hits = sum(1 for t in qtok if t in blob)
        if rel.startswith("timeline/"):
            base += 0.5
        if rel.startswith("threads/"):
            base += 1.0 + hits
        if rel.startswith("concepts/"):
            base += 0.8 + hits
        if rel in ("_forecast_instructions.md", "_procedure.md"):
            base += 2.0
        scored.append((base + hits * 2.0, rel))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[str] = []
    for _, rel in scored:
        if rel.startswith("_") and rel not in ("_forecast_instructions.md", "_procedure.md"):
            continue
        if rel not in chosen:
            chosen.append(rel)
        if len(chosen) >= max_paths:
            break
    for must in ("_forecast_instructions.md", "_procedure.md"):
        if must in manifest and must not in chosen:
            chosen.insert(0, must)
    return chosen


def gather_pit_file_bodies(
    vault_dir: Path,
    paths: list[str],
    cutoff: date,
    *,
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> tuple[str, list[str]]:
    """Read only listed paths from vault; threads already truncated in PIT snapshots."""
    vault = vault_dir.resolve()
    sections: list[str] = []
    read: list[str] = []
    budget = max_chars
    per_file = max(1200, max_chars // max(len(paths), 1))

    for rel in paths:
        if budget <= 0:
            break
        p = vault / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if rel.startswith("threads/") and "PIT: events after" not in text:
            text = truncate_thread_for_pit(text, cutoff)
        chunk = text[:per_file]
        sections.append(f"### {rel}\n{chunk}")
        read.append(rel)
        budget -= len(chunk)

    return "\n\n---\n\n".join(sections), read


def pit_search(
    question: str,
    cutoff: date,
    *,
    vault_dir: Path | None = None,
    max_results: int = 8,
) -> PitSearchResponse:
    """Deterministic PIT retrieval (no LLM). Used by synthesis and librarian preload."""
    from harness.config import VAULT_DIR

    root = Path(vault_dir or VAULT_DIR).resolve()
    manifest = list_admissible_paths(root, cutoff)
    if not manifest:
        return PitSearchResponse(cutoff=cutoff, results=[], manifest_count=0, error="empty manifest")

    ranked = rank_admissible_paths(question, manifest, max_paths=max_results)
    results: list[PitSearchResult] = []
    for rel in ranked[:max_results]:
        p = root / rel
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8", errors="replace")[:2000]
        if rel.startswith("threads/"):
            body = truncate_thread_for_pit(body, cutoff)
        results.append(PitSearchResult(path=rel, excerpt=body[:1500], score=1.0))

    return PitSearchResponse(cutoff=cutoff, results=results, manifest_count=len(manifest))


def results_to_prompt_block(results: list[PitSearchResult], cutoff: date) -> str:
    if not results:
        return f"(no PIT sources found for cutoff {cutoff.isoformat()})"
    parts = [f"PIT sources as of {cutoff.isoformat()}:"]
    for r in results:
        parts.append(f"\n#### {r.path}\n{r.excerpt[:1200]}")
    return "\n".join(parts)


def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _HERMES_PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes failed: {err}")
    return (result.stdout or "").strip()


def _parse_brief(raw: str, cutoff: date) -> PitResearchBrief:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return PitResearchBrief(
            cutoff=cutoff.isoformat(),
            conjuncture=raw[:2000],
            raw_json={"parse_error": "no json"},
        )
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        py = re.search(r'"p_yes"', m.group(0))
        return PitResearchBrief(cutoff=cutoff.isoformat(), conjuncture=raw[:1500], raw_json={})

    if not isinstance(data, dict):
        data = {}

    def _list(key: str) -> list[str]:
        val = data.get(key)
        if not isinstance(val, list):
            return []
        return [str(x) for x in val if x]

    return PitResearchBrief(
        cutoff=cutoff.isoformat(),
        conjuncture=str(data.get("conjuncture") or data.get("conjuncture_summary") or ""),
        key_events=_list("key_events"),
        active_threads=_list("active_threads"),
        mechanisms=_list("mechanisms"),
        uncertainties=_list("uncertainties"),
        excluded_as_post_cutoff=_list("excluded_as_post_cutoff"),
        sources=_list("sources"),
        raw_json=data,
    )


def build_pit_research_prompt(
    question: str,
    cutoff: date,
    *,
    context: str,
    manifest: list[str],
    vault_dir: Path,
    market_yes_at_cutoff: float | None = None,
) -> str:
    market_line = ""
    if market_yes_at_cutoff is not None:
        market_line = (
            f"\nPolymarket YES at cutoff (for calibration context only): {market_yes_at_cutoff:.4f}\n"
            "Report what the graph supports at cutoff; note if traders likely still uncertain.\n"
        )

    return "\n".join([
        "=== PIT RESEARCH LIBRARIAN ===",
        "You are a READ-ONLY sub-agent. You do NOT forecast. You do NOT write to the vault.",
        "Your ONLY job: extract facts and conjuncture knowable strictly BEFORE OR ON the cutoff.",
        "",
        f"Question (for relevance only): {question}",
        f"Cutoff: {cutoff.isoformat()}",
        market_line,
        format_admissible_block(manifest, vault_dir=vault_dir),
        "",
        "RULES:",
        "1. Use ONLY the preloaded excerpts below — no web, no files outside manifest.",
        "2. Do NOT include outcomes that traders at cutoff had not yet priced unless clearly public before cutoff.",
        "3. If vault text describes an event on cutoff day but market would still be uncertain, list under uncertainties.",
        "4. Flag any post-cutoff narrative you had to ignore.",
        "",
        "=== PRELOADED PIT EXCERPTS ===",
        context,
        "",
        "=== OUTPUT (JSON only) ===",
        """{
  "conjuncture": "2-4 sentences: forces interacting at cutoff",
  "key_events": ["event ≤ cutoff", ...],
  "active_threads": ["thread slug / dynamic", ...],
  "mechanisms": ["concept / mechanism", ...],
  "uncertainties": ["what remains open at cutoff", ...],
  "excluded_as_post_cutoff": ["what you refused to carry forward", ...],
  "sources": ["path/under/vault", ...]
}""",
    ])


def run_pit_research(
    question: str,
    cutoff: date,
    *,
    vault_dir: Path,
    market_yes_at_cutoff: float | None = None,
    use_snapshot: bool = True,
) -> tuple[PitResearchBrief, tempfile.TemporaryDirectory[str] | None]:
    """Run the PIT librarian sub-agent; returns brief and optional temp snapshot holder."""
    source = vault_dir.resolve()
    manifest = list_admissible_paths(source, cutoff)
    pit_tmp: tempfile.TemporaryDirectory[str] | None = None
    research_root = source

    if use_snapshot:
        pit_tmp = tempfile.TemporaryDirectory(prefix="pit-research-")
        research_root = Path(pit_tmp.name)
        materialize_pit_snapshot(source, research_root, cutoff)
        manifest = list_admissible_paths(research_root, cutoff)

    ranked = rank_admissible_paths(question, manifest)
    context, sources_read = gather_pit_file_bodies(research_root, ranked, cutoff)

    prompt = build_pit_research_prompt(
        question,
        cutoff,
        context=context,
        manifest=manifest,
        vault_dir=research_root,
        market_yes_at_cutoff=market_yes_at_cutoff,
    )
    raw = _call_hermes(prompt)
    brief = _parse_brief(raw, cutoff)
    if not brief.sources:
        brief.sources = sources_read
    return brief, pit_tmp
