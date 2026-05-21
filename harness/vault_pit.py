"""Point-in-time vault filtering for forecast runs (no post-cutoff leakage).

Domain-aware: paths follow domains/<domain>/entities/, domains/<domain>/concepts/,
domains/<domain>/threads/<thread>/, etc. Content scanning blocks outcome language.

Usage:
    from harness.vault_pit import is_path_admissible, list_admissible_paths
    from harness.vault_pit import materialize_pit_snapshot, content_scan
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

_QUARTER_FILE_RE = re.compile(r"^(\d{4})-Q([1-4])\.md$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_PIT_CUTOFF_RE = re.compile(r"^pit_cutoff:\s*(\S+)", re.MULTILINE)
_PIT_BODY_CUTOFF_RE = re.compile(r"^pit_body_cutoff:\s*(\S+)", re.MULTILINE)
_INCEPTION_RE = re.compile(r"^inception:\s*(\S+)", re.MULTILINE)
_FIRST_OBSERVED_RE = re.compile(r"^first_observed:\s*(\S+)", re.MULTILINE)
_BULLET_EVENT_DATE_RE = re.compile(
    r"^- \*\*([A-Z][a-z]+ \d{1,2}, \d{4})\*\*:",
)

# ── Always admissible (no temporal constraints) ─────────────────────
_ROOT_ALLOWLIST = frozenset({
    "_forecast_instructions.md",
    "_procedure.md",
    "_spec.md",
    "_index.md",
    "_macro_gaps.md",
})

# ── Never admissible ────────────────────────────────────────────────
_PIT_EXCLUDED_PREFIXES = (
    "meta/",
    "runs/",
    "forecasts/",
    ".git/",
    ".obsidian/",
)

# ── Domain path patterns ────────────────────────────────────────────
# These regexes match the second component of a domain path, e.g.
# "domains/usa/entities/joe-biden.md" -> match "entities"
# "domains/usa/_domain.md" -> match "_domain.md"
_DOMAIN_CHILD_RE = re.compile(r"^domains/[^/]+/([^/]+)")


# ═════════════════════════════════════════════════════════════════════
# Content scanning — outcome language detection
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ContentWarning:
    pattern: str
    line: int
    snippet: str
    severity: str  # "HIGH" or "MEDIUM"


_OUTCOME_PATTERNS: list[tuple[str, str, str]] = [
    # ── HIGH severity ──
    ("would_later", r"\bwould\s+(later|go\s+on\s+to|eventually)\b", "HIGH"),
    ("culminated", r"\bculminated?\s+in\b", "HIGH"),
    ("resolved_yes_no", r"\bresolved\s+(YES|NO|yes|no)\b", "HIGH"),
    ("forward_trace", r"(?:Forward\s+[Tt]race|Forward\s+[Gg]lance)", "HIGH"),
    ("hindsight", r"\bin\s+(hindsight|retrospect)\b", "HIGH"),
    ("ex_post", r"\bex\s+post\b", "HIGH"),
    ("history_would", r"(?:history\s+would\s+(?:judge|record|show)|what\s+history\s+would)", "HIGH"),
    ("as_became_clear", r"\bas\s+(?:became|would\s+become)\s+clear\b", "HIGH"),
    ("what_followed", r"\bwhat\s+followed\b", "HIGH"),
    ("rest_is_history", r"\bthe?\s+rest\s+is\s+history\b", "HIGH"),
    ("time_would_prove", r"(?:time\s+(?:would|will)\s+prove|as\s+time\s+would\s+tell)", "HIGH"),
    ("years_later", r"\byears?\s+later\b", "HIGH"),
    ("as_of_this_writing", r"\b[Aa]s\s+of\s+this\s+writing\b", "HIGH"),
    ("in_the_aftermath", r"\bin\s+the\s+aftermath\b", "HIGH"),

    # ── MEDIUM severity ──
    ("subsequently", r"\bsubsequently\b", "MEDIUM"),
    ("what_would_become", r"\bwhat\s+would\s+become\b", "MEDIUM"),
    ("in_the_coming", r"\bin\s+the\s+coming\s+", "MEDIUM"),
    ("eventually_leading", r"\beventually\s+leading\s+to\b", "MEDIUM"),
    ("set_the_stage", r"\bset\s+the\s+stage\s+for\b", "MEDIUM"),
    ("paved_the_way", r"\bpaved?\s+the\s+way\s+for\b", "MEDIUM"),
    ("would_prove", r"\bwould\s+prove\s+to\s+be\b", "MEDIUM"),
    ("would_ultimately", r"\bwould\s+ultimately\b", "MEDIUM"),
    ("would_reverberate", r"\bwould\s+reverberate\b", "MEDIUM"),
    ("proving_ground", r"\bproving\s+ground\b", "MEDIUM"),
]


def _normalize_text(text: str) -> str:
    """NFKC normalize to catch zero-width/fullwidth Unicode bypass."""
    import unicodedata
    return unicodedata.normalize("NFKC", text)


def content_scan(
    text: str,
    *,
    strict: bool = False,
) -> list[ContentWarning]:
    """Scan text for outcome-language patterns.

    Returns list of ContentWarning. When ``strict`` is True, HIGH-severity
    warnings are considered blocking (the caller should reject the file).
    """
    warnings: list[ContentWarning] = []
    normalized = _normalize_text(text)
    lines = normalized.splitlines()

    for name, pattern, severity in _OUTCOME_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for i, line in enumerate(lines, start=1):
            m = regex.search(line)
            if m:
                start = max(0, m.start() - 30)
                end = min(len(line), m.end() + 30)
                snippet = line[start:end].strip()
                if strict and severity == "HIGH":
                    warnings.append(
                        ContentWarning(
                            pattern=name,
                            line=i,
                            snippet=snippet,
                            severity="HIGH",
                        )
                    )
                elif not strict:
                    warnings.append(
                        ContentWarning(
                            pattern=name,
                            line=i,
                            snippet=snippet,
                            severity=severity,
                        )
                    )
    return warnings


def content_scan_blocking(text: str) -> bool:
    """Return True if file should be blocked (HIGH in strict mode)."""
    return any(w.severity == "HIGH" for w in content_scan(text, strict=True))


# ═════════════════════════════════════════════════════════════════════
# Date helpers
# ═════════════════════════════════════════════════════════════════════


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


def _has_frontmatter_field(text: str, key_re: re.Pattern[str]) -> bool:
    """Return True if the frontmatter contains the given key."""
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return False
    return key_re.search(fm.group(1)) is not None


# ═════════════════════════════════════════════════════════════════════
# Domain-aware path classification
# ═════════════════════════════════════════════════════════════════════


def _classify_domain_path(rel: str) -> str | None:
    """Classify a vault-relative path under domains/.

    Returns one of: 'entity', 'concept', 'thread', 'thread_entity',
                    'thread_event', 'procedure', 'function', 'domain_meta',
                    or None for unrecognised.

    Handles both flat files (concepts/foo.md) and directory-based
    (concepts/foo/_concept.md) layouts.
    """
    m = _DOMAIN_CHILD_RE.match(rel)
    if not m:
        return None
    child = m.group(1)

    # Get the filename (last component)
    fname = Path(rel).name

    # /_domain.md at domain root
    if child == "_domain.md" and fname == "_domain.md":
        return "domain_meta"

    if child == "entities" and rel.endswith(".md"):
        return "entity"
    if child == "concepts":
        if rel.endswith(".md"):
            return "concept"
        return None
    if child == "procedures":
        if rel.endswith(".md"):
            return "procedure"
        return None
    if child == "functions":
        if rel.endswith(".md"):
            return "function"
        return None

    # thread subdirs: domains/<domain>/threads/<thread>/{entities,events,procedures}/
    if child == "threads":
        # domains/<domain>/threads/<thread>/_thread.md
        if rel.endswith("/_thread.md"):
            return "thread"
        # domains/<domain>/threads/<thread>/entities/*.md
        if "/entities/" in rel and rel.endswith(".md"):
            return "thread_entity"
        # domains/<domain>/threads/<thread>/events/*.md
        if "/events/" in rel and rel.endswith(".md"):
            return "thread_event"
        # domains/<domain>/threads/<thread>/procedures/*.md
        if "/procedures/" in rel and rel.endswith(".md"):
            return "procedure"

    return None


# ═════════════════════════════════════════════════════════════════════
# Path-level admissibility (by temporal constraint)
# ═════════════════════════════════════════════════════════════════════


def _timeline_admissible(rel: str, cutoff: date) -> bool:
    name = Path(rel).name
    m = _QUARTER_FILE_RE.match(name)
    if not m:
        return False
    label = f"{m.group(1)}-Q{m.group(2)}"
    end = quarter_end_date(label)
    return end is not None and end <= cutoff


def _read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def is_path_admissible(
    vault_root: Path,
    rel: str,
    cutoff: date,
    *,
    content_scan_enabled: bool = True,
    strict: bool = False,
) -> tuple[bool, str]:
    """Return (admissible, reason) whether a vault-relative path may be read at ``cutoff``.

    The second element is a short reason string for logging/debugging.
    """
    rel = rel.replace("\\", "/").lstrip("/")
    abs_path = vault_root.resolve() / rel

    # ── Root allowlist (skip content scan — structural files) ──
    if rel in _ROOT_ALLOWLIST:
        return True, "root allowlist"

    # ── Excluded prefixes ──
    for prefix in _PIT_EXCLUDED_PREFIXES:
        if rel.startswith(prefix):
            return False, "excluded prefix"

    # ── Content scan (non-excluded, non-allowlist files) ──
    text = _read_text_safe(abs_path)
    if text is not None and content_scan_enabled and strict:
        if content_scan_blocking(text):
            return False, "blocked by strict content scan"

    # ── Timeline ──
    if rel.startswith("timeline/") and rel.endswith(".md"):
        if _timeline_admissible(rel, cutoff):
            return True, "timeline admissible"
        return False, "timeline after cutoff"

    # ── History ──
    if rel.startswith("history/"):
        return True, "history always admissible"

    # ── Agent roles ──
    if rel.startswith("agent-roles/") and rel.endswith(".md"):
        return True, "agent-role always admissible"

    # ── Probes ──
    if rel.startswith("_probes/") or rel == "_probes":
        return True, "probes always admissible"

    # ── Domain paths ──
    if rel.startswith("domains/"):
        kind = _classify_domain_path(rel)
        if kind is None:
            return False, "unrecognised domain path"

        if kind in ("procedure", "function", "domain_meta"):
            return True, f"domain {kind} always admissible"

        if kind == "concept":
            # Concepts: first_observed <= cutoff. Missing = admissible (timeless).
            if text is None:
                return False, "unreadable concept"
            first = _parse_frontmatter_date(text, _FIRST_OBSERVED_RE)
            if first is not None and first > cutoff:
                return False, "concept first_observed after cutoff"
            return True, "concept admissible"

        if kind in ("entity", "thread_entity"):
            # Entities: pit_cutoff <= cutoff. Missing = admissible (no temporal constraint).
            if text is None:
                return False, "unreadable entity"
            pit = _parse_frontmatter_date(text, _PIT_CUTOFF_RE)
            if pit is not None and pit > cutoff:
                return False, "entity pit_cutoff after cutoff"
            return True, "entity admissible"

        if kind == "thread":
            # Threads: inception <= cutoff. Missing = REJECTED.
            if text is None:
                return False, "unreadable thread"
            inception = _parse_frontmatter_date(text, _INCEPTION_RE)
            if inception is None:
                return False, "thread has no inception"
            if inception > cutoff:
                return False, "thread inception after cutoff"
            return True, "thread admissible"

        if kind == "thread_event":
            # Events under threads: admissible if they exist (the thread
            # inception gate already constrains the parent).
            return True, "thread event admissible"

        return False, f"unhandled domain kind: {kind}"

    # ── Fallback: unrecognised path ──
    return False, "not in admissible path patterns"


def list_admissible_paths(
    vault_root: Path,
    cutoff: date,
    *,
    content_scan_enabled: bool = True,
    strict: bool = False,
) -> list[str]:
    """Sorted vault-relative paths readable at ``cutoff``.

    When ``strict`` is True, files with HIGH-severity content warnings
    are excluded from the manifest.
    """
    vault_root = vault_root.resolve()
    if not vault_root.is_dir():
        return list(_ROOT_ALLOWLIST)

    found: list[str] = []
    for path in sorted(vault_root.rglob("*.md")):
        rel = path.relative_to(vault_root).as_posix()
        admissible, _ = is_path_admissible(
            vault_root, rel, cutoff,
            content_scan_enabled=content_scan_enabled,
            strict=strict,
        )
        if admissible:
            found.append(rel)
    return found


# ═════════════════════════════════════════════════════════════════════
# PIT manifest formatting
# ═════════════════════════════════════════════════════════════════════


def format_admissible_block(
    paths: list[str],
    *,
    vault_dir: str | Path,
) -> str:
    """Prompt section listing only PIT-admissible files."""
    root = Path(vault_dir).resolve()
    lines = [
        "=== PIT VAULT MANIFEST (MANDATORY) ===\n"
        "You may ONLY read files listed in this manifest.",
        f"Vault root: {root}",
        "",
        "Admissible paths:",
    ]
    for rel in paths:
        lines.append(f"  - {rel}")
    lines += [
        "",
        "The live vault at graph-vault/ contains information from AFTER your cutoff.",
        "DO NOT read any file outside this manifest.",
        "DO NOT use the live vault directory for research.",
        "Web research must also respect the cutoff date.",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Thread body truncation
# ═════════════════════════════════════════════════════════════════════


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


# ═════════════════════════════════════════════════════════════════════
# Snapshot materialization
# ═════════════════════════════════════════════════════════════════════


def materialize_pit_snapshot(
    source_vault: Path,
    dest_vault: Path,
    cutoff: date,
    *,
    clear_dest: bool = True,
    content_scan_enabled: bool = True,
    strict: bool = False,
) -> list[str]:
    """Copy PIT-admissible markdown into ``dest_vault``. Returns paths copied.

    When ``strict`` is True, files with HIGH-severity content warnings
    are excluded from the snapshot.
    """
    source_vault = source_vault.resolve()
    dest_vault = dest_vault.resolve()
    if clear_dest and dest_vault.exists():
        shutil.rmtree(dest_vault)
    dest_vault.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for rel in list_admissible_paths(
        source_vault, cutoff,
        content_scan_enabled=content_scan_enabled,
        strict=strict,
    ):
        src = source_vault / rel
        if not src.is_file():
            continue
        dst = dest_vault / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if "threads/" in rel and rel.endswith("/_thread.md"):
            body = truncate_thread_for_pit(src.read_text(encoding="utf-8"), cutoff)
            dst.write_text(body, encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    return copied
