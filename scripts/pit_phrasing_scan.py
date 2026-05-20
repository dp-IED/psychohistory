#!/usr/bin/env python3
"""
PIT Phrasing Scan — detect retrocausal phrasing patterns in vault .md files.

Scans the graph-vault for phrases that could smuggle post-cutoff knowledge
into the PIT knowledge base. Flags four categories of leakage:

  1. Explicit retrocausal framing  ("ultimately led to", "would later", ...)
  2. Conclusion/outcome framing in resolved threads
  3. Temporal inconsistency (dates after a file's pit_cutoff)
  4. Predictive certainty about the future

Usage:
    python scripts/pit_phrasing_scan.py
    python scripts/pit_phrasing_scan.py --cutoff 2024-06-30
    python scripts/pit_phrasing_scan.py --jsonl > report.jsonl
"""

import argparse
import json
import os
import re
import sys
import yaml
from datetime import date, datetime
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_VAULT = PROJECT_ROOT / "graph-vault"
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "pit_scan" / "phrasing_patterns.yaml"

# Directories to exclude entirely
EXCLUDE_DIRS = {".git", ".obsidian", "runs", "forecasts", "meta", "agent-roles"}

# Root-level files to exclude
EXCLUDE_ROOT_FILES = {
    "_forecast_instructions.md",
    "_procedure.md",
    "_spec.md",
    "_index.md",
    "_macro_gaps.md",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> date | None:
    """Try to parse a date string in YYYY-MM-DD format. Return None on failure."""
    if not s or not isinstance(s, str):
        return None
    try:
        # Handle "YYYY-MM-DD"
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass
    # Handle bare 4-digit year used as cutoff
    try:
        if re.fullmatch(r"\d{4}", s.strip()):
            return datetime.strptime(s.strip() + "-12-31", "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass
    return None


def _extract_year_references(text: str, cutoff: date) -> list[tuple[int, str]]:
    """
    Find lines in *text* that reference a date strictly after *cutoff*.

    Rules:
      - YYYY-MM-D dates: compare directly against cutoff.
      - Bare 4-digit years: only flag if year > cutoff.year (future years).
        Do NOT flag lines whose only year reference matches the cutoff's
        own year (e.g. "2025" in a 2025-Q3 file with cutoff 2025-09-30).
      - Skip frontmatter lines and metadata-only lines.
    """
    hits: list[tuple[int, str]] = []
    in_frontmatter = False

    for lineno, line in enumerate(text.splitlines(), start=1):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Track frontmatter boundaries
        if line_stripped.startswith("---"):
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue

        # -- Check for specific YYYY-MM-DD dates --
        for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", line):
            try:
                spec_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                if spec_date > cutoff:
                    hits.append((lineno, line_stripped))
                    break
            except ValueError:
                continue
        else:
            # -- No specific date found; check bare 4-digit years --
            # Only flag if the year is strictly *after* the cutoff's year.
            for m in re.finditer(r"\b(20[2-9]\d)\b", line):
                yr = int(m.group(1))
                if yr > cutoff.year:
                    hits.append((lineno, line_stripped))
                    break

    return hits


def _is_year_reference(s: str) -> bool:
    """Check if string is a bare 4-digit year."""
    return bool(re.fullmatch(r"\d{4}", s.strip()))


def _parse_quarter_boundary(file_stem: str) -> date | None:
    """For a file like '2024-Q1', return the last day of that quarter."""
    m = re.fullmatch(r"(\d{4})-Q([1-4])", file_stem)
    if not m:
        return None
    year = int(m.group(1))
    q = int(m.group(2))
    boundaries = {
        1: date(year, 3, 31),
        2: date(year, 6, 30),
        3: date(year, 9, 30),
        4: date(year, 12, 31),
    }
    return boundaries[q]


# ── Frontmatter ──────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from a Markdown string. Returns {} on failure."""
    content = content.lstrip("\ufeff")  # strip BOM if present
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_block = content[3:end].strip()
    try:
        return yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return {}


# ── File collection ──────────────────────────────────────────────────────────

def collect_vault_files(vault_path: Path) -> list[Path]:
    """
    Walk *vault_path* and return all admissible .md files, excluding
    the EXCLUDE_DIRS and EXCLUDE_ROOT_FILES.
    """
    files: list[Path] = []
    for entry in vault_path.iterdir():
        # Skip excluded directories
        if entry.is_dir() and entry.name in EXCLUDE_DIRS:
            continue
        # Skip git/obsidian hidden dirs
        if entry.is_dir() and entry.name.startswith("."):
            continue
        # Skip agent-roles
        if entry.is_dir() and entry.name == "agent-roles":
            continue

        if entry.is_dir():
            # Walk subdirectories (threads/, concepts/, entities/, timeline/, events/)
            for root, _dirs, leafs in os.walk(entry):
                root_path = Path(root)
                # Skip excluded subdirectories
                rel_name = root_path.name
                if rel_name in EXCLUDE_DIRS or rel_name.startswith("."):
                    continue
                # Also skip any excluded dir in the path
                parts = root_path.relative_to(vault_path).parts
                if any(p in EXCLUDE_DIRS or p.startswith(".") for p in parts):
                    continue
                for leaf in leafs:
                    if leaf.endswith(".md"):
                        files.append(root_path / leaf)
        elif entry.name.endswith(".md"):
            # Root-level .md files — exclude the known meta files
            if entry.name in EXCLUDE_ROOT_FILES:
                continue
            files.append(entry)

    return sorted(files)


# ── Admissibility ────────────────────────────────────────────────────────────

def get_file_cutoff(frontmatter: dict, file_path: Path) -> date | None:
    """
    Determine the effective cutoff date for a file.

    Priority:
      1. pit_body_cutoff (most restrictive)
      2. pit_cutoff (timeline quarter boundary)
      3. conclusion (thread resolution date)
      4. For timeline files named YYYY-QN without explicit cutoff,
         derive from the quarter boundary.
    """
    # Check explicit frontmatter fields
    for key in ("pit_body_cutoff", "pit_cutoff", "conclusion"):
        val = frontmatter.get(key)
        if val:
            parsed = _parse_date(str(val)) if not _is_year_reference(str(val)) else None
            if parsed:
                return parsed

    # For quarter timeline files, derive from filename
    stem = file_path.stem
    qd = _parse_quarter_boundary(stem)
    if qd:
        return qd

    # Check for 'year' field (some timeline files have year+quarter info)
    year_val = frontmatter.get("year")
    if year_val and frontmatter.get("type") == "quarter":
        return _parse_quarter_boundary(f"{year_val}-Q1")

    return None


def is_admissible_at_cutoff(frontmatter: dict, file_path: Path,
                            global_cutoff: date | None) -> bool:
    """
    Return True if the file's content *could* be admissible given *global_cutoff*.

    When no *global_cutoff* is given, all files are admissible.
    """
    if global_cutoff is None:
        return True

    # Timeline files: pit_cutoff must be <= global_cutoff
    if frontmatter.get("type") == "quarter" or file_path.parent.name == "timeline":
        fc = get_file_cutoff(frontmatter, file_path)
        if fc is not None:
            return fc <= global_cutoff

    # Thread files: inception must be <= global_cutoff
    if frontmatter.get("type") == "thread":
        inception = frontmatter.get("inception")
        if inception:
            idate = _parse_date(str(inception))
            if idate:
                return idate <= global_cutoff
        # Also check pit_body_cutoff
        pbc = frontmatter.get("pit_body_cutoff")
        if pbc:
            pbc_date = _parse_date(str(pbc))
            if pbc_date:
                return pbc_date <= global_cutoff

    # Events/concepts/entities: always admissible (no temporal gate)
    return True


# ── Scanning ─────────────────────────────────────────────────────────────────

class ScanResult:
    """Aggregates all findings for a scan run."""

    def __init__(self, cutoff: date | None, config: dict):
        self.cutoff = cutoff
        self.config = config
        self.scanned_count = 0
        self.flagged_count = 0
        self.categories: dict[str, list[dict]] = {
            "retrocausal": [],
            "conclusion_framing": [],
            "temporal_inconsistency": [],
            "predictive_certainty": [],
        }

    def add(self, category: str, file_rel: str, lineno: int, line: str,
            phrase: str, severity: str = "medium"):
        self.categories.setdefault(category, []).append({
            "file": file_rel,
            "line": lineno,
            "text": line.strip(),
            "phrase": phrase,
            "severity": severity,
        })
        self.flagged_count += 1


def scan_file(file_path: Path, vault_root: Path, global_cutoff: date | None,
              config: dict, result: ScanResult):
    """Run all scans on a single markdown file."""
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return  # skip unreadable files

    frontmatter = parse_frontmatter(raw)
    rel_path = file_path.relative_to(vault_root).as_posix()

    # Check admissibility
    is_admitted = is_admissible_at_cutoff(frontmatter, file_path, global_cutoff)
    if not is_admitted:
        return  # skip files not admissible at this cutoff

    result.scanned_count += 1

    # Determine effective cutoff for temporal checks
    effective_cutoff = get_file_cutoff(frontmatter, file_path) or global_cutoff

    # ── Split content into lines ──
    lines = raw.splitlines()

    # ── Category 1: Retrocausal framing (all admissible files) ──
    retro_patterns = config.get("patterns", {}).get("retrocausal", [])
    _scan_pattern_category(lines, rel_path, retro_patterns, result, "retrocausal")

    # ── Category 2: Conclusion/outcome framing (resolved threads) ──
    is_resolved = (
        frontmatter.get("status") == "resolved"
        and frontmatter.get("conclusion") is not None
    )
    if is_resolved:
        concl_patterns = config.get("patterns", {}).get("conclusion_framing", [])
        _scan_pattern_category(lines, rel_path, concl_patterns, result, "conclusion_framing")

    # ── Category 3: Temporal inconsistency ──
    temporal_cfg = config.get("patterns", {}).get("temporal_inconsistency", {})
    if temporal_cfg.get("enabled", True) and effective_cutoff is not None:
        # Check for year references after the cutoff
        date_hits = _extract_year_references(raw, effective_cutoff)
        for lineno, line_text in date_hits:
            result.add(
                "temporal_inconsistency",
                rel_path,
                lineno,
                line_text,
                phrase=f"year reference > {effective_cutoff}",
                severity=temporal_cfg.get("severity", "high"),
            )

    # ── Category 4: Predictive certainty ──
    cert_patterns = config.get("patterns", {}).get("predictive_certainty", [])
    _scan_pattern_category(lines, rel_path, cert_patterns, result, "predictive_certainty")


def _scan_pattern_category(lines: list[str], rel_path: str,
                           patterns: list[dict], result: ScanResult,
                           category: str):
    """Scan *lines* for each *pattern* and add hits to *result*."""
    for pat_def in patterns:
        phrase = pat_def.get("phrase", "")
        if not phrase:
            continue
        severity = pat_def.get("severity", "medium")
        # Case-insensitive search
        for lineno, line in enumerate(lines, start=1):
            # Skip frontmatter
            if line.strip().startswith("---"):
                continue
            # Simple substring match (case-insensitive)
            if phrase.lower() in line.lower():
                result.add(category, rel_path, lineno, line, phrase, severity)


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(result: ScanResult):
    """Print human-readable scan report to stdout."""
    cutoff_str = str(result.cutoff) if result.cutoff else "(none)"
    print(f"PIT Phrasing Scan — cutoff={cutoff_str}")
    print(f"Scanned {result.scanned_count} files, {result.flagged_count} total flagged")
    print()

    category_headers = {
        "retrocausal": ("Category 1", "Retrocausal framing"),
        "conclusion_framing": ("Category 2", "Conclusion/outcome framing"),
        "temporal_inconsistency": ("Category 3", "Temporal inconsistency"),
        "predictive_certainty": ("Category 4", "Predictive certainty"),
    }

    for cat_key in category_headers:
        hits = result.categories.get(cat_key, [])
        cat_num, cat_name = category_headers[cat_key]
        plural = "es" if len(hits) != 1 else ""
        print(f"== {cat_num}: {cat_name} ({len(hits)} match{plural}) ==")
        if not hits:
            print("  (none)")
        else:
            for h in hits:
                print(f"  {h['file']}:{h['line']} - \"{h['text'][:120]}\"")
                print(f"           [{h['phrase']}] severity={h['severity']}")
        print()


def print_jsonl(result: ScanResult):
    """Print scan results as JSON Lines to stdout."""
    for cat_key, hits in result.categories.items():
        for h in hits:
            record = {
                "category": cat_key,
                "file": h["file"],
                "line": h["line"],
                "text": h["text"],
                "phrase": h["phrase"],
                "severity": h["severity"],
                "cutoff": str(result.cutoff) if result.cutoff else None,
            }
            print(json.dumps(record))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIT Phrasing Scan — detect retrocausal phrasing leakage in vault .md files"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Global cutoff date (YYYY-MM-DD). Only files admissible at this cutoff are scanned.",
    )
    parser.add_argument(
        "--vault",
        type=str,
        default=str(DEFAULT_VAULT),
        help=f"Path to graph-vault directory (default: {DEFAULT_VAULT})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Path to pattern config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Output JSON Lines instead of human-readable report",
    )
    args = parser.parse_args()

    # Resolve paths
    vault_path = Path(args.vault).resolve()
    config_path = Path(args.config).resolve()

    if not vault_path.is_dir():
        print(f"ERROR: vault directory not found: {vault_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Parse cutoff
    global_cutoff = None
    if args.cutoff:
        try:
            global_cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()
        except ValueError:
            print(f"ERROR: invalid cutoff date '{args.cutoff}'. Use YYYY-MM-DD.",
                  file=sys.stderr)
            sys.exit(1)

    # Collect files
    all_files = collect_vault_files(vault_path)
    if not all_files:
        print("WARNING: No .md files found in vault.", file=sys.stderr)

    # Scan
    result = ScanResult(cutoff=global_cutoff, config=config)
    for fp in all_files:
        scan_file(fp, vault_path, global_cutoff, config, result)

    # Output
    if args.jsonl:
        print_jsonl(result)
    else:
        print_report(result)

    # Return exit code 0 even if hits found (intentional — caller checks stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
