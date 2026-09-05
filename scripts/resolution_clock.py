"""Claim-line dates that are later than the problem Resolution.

Graded on P-usca-338: the 19 Aug matching row named 22 Aug 2026 while
Resolution stayed 19 Aug, so predict stopped and the post-pause take-effect
was never a dated claim. Later reflect ticks told predict to slide, then
left the heading frozen until a still-later tick. Prose did not keep the
clock honest. This script is the clock: parse Claim lines only (not
Justification — analog dates are not answering dates). If a claim names a
calendar date strictly after that problem's Resolution, slide the heading
later. No LLM.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.ledger import Ledger, parse_ledger

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(
    r"\b(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MDY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class LaterDate:
    problem_id: str
    claim_id: str
    resolution_day: date
    named_day: date


def dates_in_claim(text: str) -> tuple[date, ...]:
    found: list[date] = []
    seen: set[date] = set()

    def add(year: int, month: int, day: int) -> None:
        try:
            value = date(year, month, day)
        except ValueError:
            return
        if value not in seen:
            seen.add(value)
            found.append(value)

    for match in _ISO.finditer(text):
        add(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    for match in _DMY.finditer(text):
        add(int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1)))
    for match in _MDY.finditer(text):
        add(int(match.group(3)), _MONTHS[match.group(1).lower()], int(match.group(2)))
    return tuple(found)


def later_than_resolution(book: Ledger) -> tuple[LaterDate, ...]:
    rows: list[LaterDate] = []
    for claim in book.claims:
        problem = book.problem(claim.problem_id)
        if problem is None:
            continue
        for named in dates_in_claim(claim.claim):
            if named > problem.resolution_day:
                rows.append(
                    LaterDate(
                        problem_id=problem.id,
                        claim_id=claim.id,
                        resolution_day=problem.resolution_day,
                        named_day=named,
                    )
                )
    return tuple(rows)


def latest_named_day(rows: tuple[LaterDate, ...], problem_id: str) -> date | None:
    named = [row.named_day for row in rows if row.problem_id == problem_id]
    return max(named) if named else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exit 1 if a Claim names a date later than Resolution."
    )
    parser.add_argument(
        "ledger",
        nargs="?",
        default="ledger.md",
        type=Path,
        help="ledger markdown (default: ledger.md)",
    )
    args = parser.parse_args(argv)
    book = parse_ledger(args.ledger.read_text(encoding="utf-8"))
    rows = later_than_resolution(book)
    if not rows:
        return 0
    for row in rows:
        print(
            f"{row.problem_id} {row.claim_id}: Claim names {row.named_day.isoformat()} "
            f"after Resolution {row.resolution_day.isoformat()}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
