"""Live latest Claim is a past-tense pause recap after Resolution already equals the named next date.

Graded on P-usca-338: the 19 Aug matching row named 22 Aug 2026, Resolution later
slid to that morning, and the Claim still only recapped the pause. Predict then
treated the outcome as unchanged, so take-effect was never a dated row. The
clock script catches a frozen heading; this script catches a recap that is not
the new-clock modal. Claim lines only. No LLM. Live problems only — a past
series keeps its recap as history.
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

from harness.ledger import Claim, Ledger, parse_ledger
from scripts.resolution_clock import dates_in_claim

_PAST_PAUSE = re.compile(
    r"\b(delayed|paused|postponed|did not take effect)\b",
    re.IGNORECASE,
)
_FUTURE_DELAY = re.compile(
    r"\b(will delay|will not apply|will not take effect)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PauseRecap:
    problem_id: str
    claim_id: str
    forecast_day: date
    resolution_day: date
    named_day: date


def latest_claim(book: Ledger, problem_id: str) -> Claim | None:
    rows = book.claims_for(problem_id)
    return rows[-1] if rows else None


def pause_recaps(book: Ledger, as_of: date) -> tuple[PauseRecap, ...]:
    found: list[PauseRecap] = []
    for problem in book.live_problems(as_of):
        claim = latest_claim(book, problem.id)
        if claim is None:
            continue
        if claim.forecast_day >= problem.resolution_day:
            continue
        if _FUTURE_DELAY.search(claim.claim):
            continue
        if not _PAST_PAUSE.search(claim.claim):
            continue
        named = dates_in_claim(claim.claim)
        if problem.resolution_day not in named:
            continue
        found.append(
            PauseRecap(
                problem_id=problem.id,
                claim_id=claim.id,
                forecast_day=claim.forecast_day,
                resolution_day=problem.resolution_day,
                named_day=problem.resolution_day,
            )
        )
    return tuple(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exit 1 if a live latest Claim recaps a pause at Resolution."
    )
    parser.add_argument(
        "ledger",
        nargs="?",
        default="ledger.md",
        type=Path,
        help="ledger markdown (default: ledger.md)",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="calendar day YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    book = parse_ledger(args.ledger.read_text(encoding="utf-8"))
    rows = pause_recaps(book, as_of)
    if not rows:
        return 0
    for row in rows:
        print(
            f"{row.problem_id} {row.claim_id}: Claim recaps a pause to "
            f"{row.named_day.isoformat()} at Resolution "
            f"{row.resolution_day.isoformat()} (forecast {row.forecast_day.isoformat()})",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
