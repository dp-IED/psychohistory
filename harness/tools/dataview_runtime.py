"""Minimal Dataview-DQL-style execution over vault markdown (for agent runtime).

Supports a constrained subset of Obsidian Dataview so exported run notes stay
compatible with the real plugin while tests and CI run without Obsidian.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.vault_markdown import iter_vault_markdown_files, read_vault_note

_DATAVIEW_BLOCK = re.compile(r"```dataview\s*\n([\s\S]*?)```", re.IGNORECASE)
_WS = re.compile(r"\s+")
_COND = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|<=|>=|<|>)\s*(.+)\s*$",
)


@dataclass(frozen=True)
class SimpleQuery:
    table_fields: list[str]
    from_subdir: str
    where_clauses: list[tuple[str, str, Any]]
    sort_field: str | None
    sort_dir: str  # ASC or DESC
    limit: int | None


def extract_dataview_queries(text: str) -> list[str]:
    return [m.group(1).strip() for m in _DATAVIEW_BLOCK.finditer(text)]


def _parse_value(raw: str) -> Any:
    s = raw.strip()
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _normalize_path_fragment(rel: str) -> str:
    p = rel.strip().strip('"').strip("'")
    return p.replace("\\", "/")


def parse_simple_dataview(query: str) -> SimpleQuery:
    lines = [ln.strip() for ln in query.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    if not lines:
        raise ValueError("empty dataview query")

    header = lines[0]
    if not header.upper().startswith("TABLE"):
        raise ValueError("only TABLE queries are supported")

    fields_part = header[len("TABLE") :].strip()
    fields = [f.strip() for f in fields_part.split(",") if f.strip()]

    from_subdir = ""
    where_clauses: list[tuple[str, str, Any]] = []
    sort_field: str | None = None
    sort_dir = "ASC"
    limit_n: int | None = None

    for ln in lines[1:]:
        up = ln.upper()
        if up.startswith("FROM"):
            rest = ln[4:].strip()
            from_subdir = _normalize_path_fragment(rest)
        elif up.startswith("WHERE"):
            rest = ln[5:].strip()
            parts = [p.strip() for p in rest.split(" AND ") if p.strip()]
            for part in parts:
                m = _COND.match(part)
                if not m:
                    raise ValueError(f"unsupported WHERE clause: {part!r}")
                field, op, rhs = m.group(1), m.group(2), _parse_value(m.group(3))
                where_clauses.append((field, op, rhs))
        elif up.startswith("SORT"):
            rest = ln[4:].strip()
            tokens = _WS.split(rest)
            if not tokens:
                continue
            sort_field = tokens[0]
            if len(tokens) >= 2 and tokens[1].upper() in ("ASC", "DESC"):
                sort_dir = tokens[1].upper()
        elif up.startswith("LIMIT"):
            rest = ln[5:].strip()
            limit_n = int(rest)
        else:
            continue

    if not from_subdir:
        raise ValueError("FROM clause required")

    return SimpleQuery(
        table_fields=fields,
        from_subdir=from_subdir,
        where_clauses=where_clauses,
        sort_field=sort_field,
        sort_dir=sort_dir,
        limit=limit_n,
    )


def _get_field(record: dict[str, Any], key: str) -> Any:
    if key not in record:
        return None
    return record[key]


def _cmp_match(actual: Any, op: str, expected: Any) -> bool:
    if op == "=":
        if isinstance(actual, list) or isinstance(expected, list):
            return actual == expected
        return actual == expected
    if op == "!=":
        return not _cmp_match(actual, "=", expected)
    for cast in (float, int):
        try:
            a = cast(actual)  # type: ignore[arg-type]
            e = cast(expected)  # type: ignore[arg-type]
            if op == "<":
                return a < e
            if op == ">":
                return a > e
            if op == "<=":
                return a <= e
            if op == ">=":
                return a >= e
        except (TypeError, ValueError):
            continue
    return False


def _sort_key(row: dict[str, Any], field: str) -> tuple[int, Any]:
    v = row.get(field)
    if v is None:
        return (1, "")
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (0, str(v))


def substitute_query_placeholders(query: str, *, category: str, horizon_days: int) -> str:
    return (
        query.replace("{{category}}", category)
        .replace("{{horizon_days}}", str(horizon_days))
    )


def run_dataview_query(vault_root: Path, query: str) -> str:
    q = parse_simple_dataview(query)
    sub = vault_root / q.from_subdir.strip("/")
    if not sub.exists():
        return f"(no folder {q.from_subdir!r} in vault)\n"

    rows: list[dict[str, Any]] = []
    for path in iter_vault_markdown_files(sub):
        fm, _body = read_vault_note(path)
        fm = dict(fm)
        fm["_path"] = str(path.relative_to(vault_root))
        ok = True
        for field, op, expected in q.where_clauses:
            if not _cmp_match(_get_field(fm, field), op, expected):
                ok = False
                break
        if ok:
            rows.append(fm)

    if q.sort_field:
        rev = q.sort_dir.upper() == "DESC"
        rows.sort(key=lambda r: _sort_key(r, q.sort_field), reverse=rev)

    if q.limit is not None:
        rows = rows[: q.limit]

    if not rows:
        return "_No results._\n"

    header = "| " + " | ".join(q.table_fields) + " |"
    sep = "| " + " | ".join("---" for _ in q.table_fields) + " |"
    out_lines = [header, sep]
    for rec in rows:
        cells = []
        for f in q.table_fields:
            v = rec.get(f)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            cells.append("" if v is None else str(v))
        out_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(out_lines) + "\n"


__all__ = [
    "extract_dataview_queries",
    "parse_simple_dataview",
    "run_dataview_query",
    "substitute_query_placeholders",
]
