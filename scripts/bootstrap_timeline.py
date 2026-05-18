#!/usr/bin/env python3
"""Bootstrap quarters/ directory with mechanical year and quarter nodes.

Creates graph-vault/quarters/ with deterministic prev/next links.
Run once. Does not overwrite existing files.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from harness.config import VAULT_DIR

VAULT = VAULT_DIR


def _quarter_label(y: int, q: int) -> str:
    return f"{y}-Q{q}"


def _quarter_path(y: int, q: int) -> Path:
    return VAULT / "quarters" / f"{_quarter_label(y, q)}.md"


def _quarter_internal_link(y: int, q: int) -> str:
    return f"[[{_quarter_label(y, q)}]]"


def _year_path(y: int) -> Path:
    return VAULT / "quarters" / f"{y}.md"


def _year_internal_link(y: int) -> str:
    return f"[[{y}]]"


def write_year(y: int) -> None:
    fpath = _year_path(y)
    if fpath.exists():
        return

    quarters = [_quarter_label(y, q) for q in range(1, 5)]
    quarters_str = ", ".join(f'"[[{q}]]"' for q in quarters)

    content = f"""---
type: year
label: "{y}"
prev: {_year_internal_link(y - 1)}
next: {_year_internal_link(y + 1)}
quarters: [{quarters_str}]
"""
    fpath.write_text(content.lstrip(), encoding="utf-8")


def write_quarter(y: int, q: int) -> None:
    fpath = _quarter_path(y, q)
    if fpath.exists():
        return

    prev_q = q - 1 if q > 1 else 4
    prev_y = y if q > 1 else y - 1
    next_q = q + 1 if q < 4 else 1
    next_y = y if q < 4 else y + 1

    start_month = (q - 1) * 3 + 1
    start = date(y, start_month, 1)
    end_month = start_month + 2
    end = date(y, end_month, 1)
    import calendar
    last_day = calendar.monthrange(y, end_month)[1]
    end = date(y, end_month, last_day)

    content = f"""---
type: quarter
year: {_year_internal_link(y)}
label: "{_quarter_label(y, q)}"
date_range: "{start.isoformat()} to {end.isoformat()}"
prev: {_quarter_internal_link(prev_y, prev_q)}
next: {_quarter_internal_link(next_y, next_q)}
---
"""
    fpath.write_text(content.lstrip(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap timeline directory.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2030)
    args = parser.parse_args()

    timeline_dir = VAULT / "quarters"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    count = {"years": 0, "quarters": 0}
    for y in range(args.start, args.end + 1):
        write_year(y)
        count["years"] += 1
        for q in range(1, 5):
            write_quarter(y, q)
            count["quarters"] += 1

    print(f"Timeline bootstrapped: {count['years']} years, {count['quarters']} quarters ({args.start}-{args.end})")
    print(f"  Location: {timeline_dir}")


if __name__ == "__main__":
    main()
