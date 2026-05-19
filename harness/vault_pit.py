"""Point-in-time vault filtering for forecast runs (no post-cutoff leakage)."""

from __future__ import annotations

import re
import shutil
from datetime import date, datetime
from pathlib import Path

_QUARTER_FILE_RE = re.compile(r"^(\d{4})-Q([1-4])\.md$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_PIT_CUTOFF_RE = re.compile(r"^pit_cutoff:\s*(\S+)", re.MULTILINE)
_PIT_BODY_CUTOFF_RE = re.compile(r"^pit_body_cutoff:\s*(\S+)", re.MULTILINE)
_INCEPTION_RE = re.compile(r"^inception:\s*(\S+)", re.MULTILINE)
_FIRST_OBSERVED_RE = re.compile(r"^first_observed:\s*(\S+)", re.MULTILINE)
_BULLET_EVENT_DATE_RE = re.compile(
    r"^- \*\*([A-Z][a-z]+ \d{1,2}, \d{4})\*\*:",
)

# Always admissible at any cutoff (policy + workflow; no event claims).
_ROOT_ALLOWLIST = (
    "_forecast_instructions.md",
    "_procedure.md",
    "_spec.md",
    "_index.md",
)

# Never admissible for PIT calibration (post-hoc or future-labelled).
_PIT_EXCLUDED_PREFIXES = (
    "meta/",
    "runs/",
    "forecasts/",
    ".git/",
    ".obsidian/",
)


def quarter_end_date(label: str) -> date | None:
    """Last calendar day of a quarter label like ``2024-Q3``."""
    m = re.match(r"^(\d{4})-Q([1-4])$", label.strip())
    if not m:
        return None
    year, q = int(m.group(1)), int(m.group(2))
    if q == 1:
        return date(year, 3, 31)
    if q == 2:
        return date(year, 6, 30)
    if q == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def _parse_frontmatter_date(text: str, key_re: re.Pattern[str]) -> date | None:
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return None
    m = key_re.search(fm.group(1))
    if not m:
        return None
    raw = m.group(1).strip().strip('"').strip("'")
    if raw.lower() in {"null", "ongoing", "~"}:
        return None
    if re.match(r"^\d{4}$", raw):
        return date(int(raw), 12, 31)
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _timeline_admissible(rel: str, cutoff: date) -> bool:
    name = Path(rel).name
    m = _QUARTER_FILE_RE.match(name)
    if not m:
        return False
    label = f"{m.group(1)}-Q{m.group(2)}"
    end = quarter_end_date(label)
    return end is not None and end <= cutoff


def _entity_admissible(path: Path, cutoff: date) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    pit = _parse_frontmatter_date(text, _PIT_CUTOFF_RE)
    if pit is not None:
        return pit <= cutoff
    return False


def _thread_admissible(path: Path, cutoff: date) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    inception = _parse_frontmatter_date(text, _INCEPTION_RE)
    if inception is not None:
        return inception <= cutoff
    return False


def _concept_admissible(path: Path, cutoff: date) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    first = _parse_frontmatter_date(text, _FIRST_OBSERVED_RE)
    if first is not None:
        return first <= cutoff
    return True


def is_path_admissible(vault_root: Path, rel: str, cutoff: date) -> bool:
    """Return whether a vault-relative path may be read at ``cutoff``."""
    rel = rel.replace("\\", "/").lstrip("/")
    for prefix in _PIT_EXCLUDED_PREFIXES:
        if rel.startswith(prefix):
            return False

    if rel in _ROOT_ALLOWLIST:
        return True

    if rel.startswith("timeline/") and rel.endswith(".md"):
        return _timeline_admissible(rel, cutoff)

    if rel.startswith("entities/") and rel.endswith(".md"):
        return _entity_admissible(vault_root / rel, cutoff)

    if rel.startswith("threads/") and rel.endswith(".md"):
        return _thread_admissible(vault_root / rel, cutoff)

    if rel.startswith("concepts/") and rel.endswith(".md"):
        return _concept_admissible(vault_root / rel, cutoff)

    if rel.startswith("history/"):
        return True

    if rel.startswith("agent-roles/") and rel.endswith(".md"):
        return True

    if rel.startswith("topics/") and rel.endswith(".md"):
        return _concept_admissible(vault_root / rel, cutoff)

    return False


def list_admissible_paths(vault_root: Path, cutoff: date) -> list[str]:
    """Sorted vault-relative paths readable at ``cutoff``."""
    vault_root = vault_root.resolve()
    if not vault_root.is_dir():
        return list(_ROOT_ALLOWLIST)

    found: list[str] = []
    for path in sorted(vault_root.rglob("*.md")):
        rel = path.relative_to(vault_root).as_posix()
        if is_path_admissible(vault_root, rel, cutoff):
            found.append(rel)
    return found


def format_admissible_block(paths: list[str], *, vault_dir: str | Path) -> str:
    """Prompt section listing only PIT-admissible files."""
    root = Path(vault_dir).resolve()
    lines = [
        "=== PIT VAULT MANIFEST (MANDATORY) ===",
        f"Information cutoff: enforced. You may ONLY read files under:",
        str(root),
        "",
        "Admissible paths (do not read any other file in this vault):",
    ]
    for rel in paths:
        lines.append(f"  - {rel}")
    lines += [
        "",
        "Forbidden: meta/, runs/, forecasts/, timeline quarters after cutoff,",
        "entities/threads with pit_cutoff or inception after cutoff.",
        "Web research must also respect the cutoff date.",
        "",
    ]
    return "\n".join(lines)


def _parse_bullet_event_date(line: str) -> date | None:
    m = _BULLET_EVENT_DATE_RE.match(line.strip())
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return datetime.strptime(raw, "%B %d %Y").date()
    except ValueError:
        return None


def _thread_body_cutoff(text: str, fallback: date) -> date:
    body_cut = _parse_frontmatter_date(text, _PIT_BODY_CUTOFF_RE)
    if body_cut is not None:
        return min(body_cut, fallback)
    return fallback


def truncate_thread_for_pit(text: str, cutoff: date) -> str:
    """Drop timeline bullets strictly after ``cutoff`` (post-hoc narrative leakage)."""
    effective = _thread_body_cutoff(text, cutoff)
    lines = text.splitlines()
    out: list[str] = []
    omit_rest = False
    for line in lines:
        if omit_rest:
            continue
        event_d = _parse_bullet_event_date(line)
        if event_d is not None and event_d > effective:
            omit_rest = True
            out.append(
                f"\n> PIT: events after {effective.isoformat()} omitted from this snapshot.\n"
            )
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if out and not out[-1].endswith("\n") else "")


def materialize_pit_snapshot(
    source_vault: Path,
    dest_vault: Path,
    cutoff: date,
    *,
    clear_dest: bool = True,
) -> list[str]:
    """Copy PIT-admissible markdown into ``dest_vault``. Returns paths copied."""
    source_vault = source_vault.resolve()
    dest_vault = dest_vault.resolve()
    if clear_dest and dest_vault.exists():
        shutil.rmtree(dest_vault)
    dest_vault.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for rel in list_admissible_paths(source_vault, cutoff):
        src = source_vault / rel
        if not src.is_file():
            continue
        dst = dest_vault / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith("threads/"):
            body = truncate_thread_for_pit(src.read_text(encoding="utf-8"), cutoff)
            dst.write_text(body, encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    return copied
