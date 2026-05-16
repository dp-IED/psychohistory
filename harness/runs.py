"""Episode runs stored as timestamped markdown files in vault/runs/.

Each run is a `.md` file with YAML frontmatter containing all fields.
No SQLite — everything is in Obsidian.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunNote:
    question_text: str
    p_yes: float
    reasoning: str = ""
    timestamp: str = ""  # auto-filled
    cutoff: date | None = None
    source: str = ""
    category: str = ""
    brier: float | None = None
    resolution: bool | None = None
    question_id: str = ""
    pit_context: str = ""
    error: str | None = None

    @property
    def filepath(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.question_text.lower()).strip("-")[:60]
        ts_part = self.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{ts_part}-{slug}.md"


def write_run(vault_dir: str | Path, note: RunNote) -> Path:
    """Write a run note to vault/runs/ as a markdown file with YAML frontmatter."""
    runs_dir = Path(vault_dir).resolve() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not note.timestamp:
        note.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    frontmatter: dict[str, Any] = {
        "timestamp": note.timestamp,
        "p_yes": round(note.p_yes, 4),
        "reasoning": note.reasoning,
    }
    if note.cutoff:
        frontmatter["cutoff"] = note.cutoff.isoformat()
    if note.source:
        frontmatter["source"] = note.source
    if note.category:
        frontmatter["category"] = note.category
    if note.brier is not None:
        frontmatter["brier"] = round(note.brier, 4)
    if note.resolution is not None:
        frontmatter["resolution"] = note.resolution
    if note.question_id:
        frontmatter["question_id"] = note.question_id
    if note.pit_context:
        frontmatter["pit_context"] = note.pit_context[:200]  # summary in frontmatter
    if note.error:
        frontmatter["error"] = note.error

    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
    content = f"---\n{yaml_str}\n---\n\n{note.question_text}\n"
    if note.reasoning:
        content += f"\n{note.reasoning}\n"
    if note.pit_context and len(note.pit_context) > 200:
        content += f"\n## PIT Context\n\n{note.pit_context}\n"

    fpath = runs_dir / note.filepath
    fpath.write_text(content, encoding="utf-8")
    return fpath


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from a markdown file."""
    m = re.match(r"^---\s*\n(.*?\n)---\s*\n(.*)", text, re.DOTALL)
    if not m:
        return {}, text.strip()
    try:
        data: dict[str, Any] = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    return data, m.group(2).strip()


def read_all_runs(vault_dir: str | Path) -> list[dict[str, Any]]:
    """Read all run files from vault/runs/, parse frontmatter, return sorted by timestamp."""
    runs_dir = Path(vault_dir).resolve() / "runs"
    if not runs_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for f in sorted(runs_dir.glob("*.md"), reverse=True):
        fm, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
        fm["_file"] = f.name
        fm["_body"] = body
        runs.append(fm)
    return runs


def runs_count(vault_dir: str | Path) -> int:
    runs_dir = Path(vault_dir).resolve() / "runs"
    if not runs_dir.exists():
        return 0
    return len(list(runs_dir.glob("*.md")))


def mean_brier(vault_dir: str | Path) -> float | None:
    runs = read_all_runs(vault_dir)
    briers = [r["brier"] for r in runs if r.get("brier") is not None]
    return sum(briers) / len(briers) if briers else None


def brier_by_category(vault_dir: str | Path) -> dict[str, float]:
    runs = read_all_runs(vault_dir)
    by_cat: dict[str, list[float]] = {}
    for r in runs:
        cat = r.get("category", "general") or "general"
        b = r.get("brier")
        if b is not None:
            by_cat.setdefault(cat, []).append(b)
    return {cat: sum(vals) / len(vals) for cat, vals in sorted(by_cat.items())}


def worst_runs(vault_dir: str | Path, n: int = 5) -> list[dict[str, Any]]:
    runs = [r for r in read_all_runs(vault_dir) if r.get("brier") is not None]
    runs.sort(key=lambda r: r["brier"], reverse=True)
    return runs[:n]


def best_runs(vault_dir: str | Path, n: int = 5) -> list[dict[str, Any]]:
    runs = [r for r in read_all_runs(vault_dir) if r.get("brier") is not None]
    runs.sort(key=lambda r: r["brier"])
    return runs[:n]
