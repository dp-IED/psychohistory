from __future__ import annotations

from datetime import date
from pathlib import Path

from harness.ledger import parse_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]


_FIXTURE = """\
# Ledger

K: 1

## Problems

### P-nato — NATO article 5 this year

Resolution: 2026-12-31

Motivation: Markets will score alliance cohesion; we need a problem row before discovery exists.

## Claims

### C-early

- Problem: P-nato
- Forecast: 2026-08-18
- Owner: nato-watcher
- Claim: NATO will not invoke article 5 in 2026.
- Justification: No article 5 trigger is on the board this year.

### C-later

- Problem: P-nato
- Forecast: 2026-09-01
- Owner: nato-watcher
- Claim: Alliance cohesion holds through year-end.
- Justification: Year-end is a different forecast day.
"""


def test_ledger_reports_k() -> None:
    book = parse_ledger(_FIXTURE)
    assert book.k == 1


def test_ledger_lists_problems_with_motivation_and_resolution() -> None:
    book = parse_ledger(_FIXTURE)
    assert len(book.problems) == 1
    problem = book.problems[0]
    assert problem.id == "P-nato"
    assert problem.title == "NATO article 5 this year"
    assert problem.resolution_day == date(2026, 12, 31)
    assert problem.motivation == (
        "Markets will score alliance cohesion; we need a problem row before discovery exists."
    )


def test_live_problems_are_those_at_or_before_resolution_day() -> None:
    book = parse_ledger(_FIXTURE)
    live = book.live_problems(date(2026, 8, 18))
    assert [p.id for p in live] == ["P-nato"]
    assert book.live_problems(date(2026, 12, 31))[0].id == "P-nato"
    assert book.live_problems(date(2027, 1, 1)) == ()


def test_after_resolution_is_claims_whose_problem_has_passed() -> None:
    book = parse_ledger(_FIXTURE)
    assert book.after_resolution(date(2026, 12, 31)) == ()
    ids = {c.id for c in book.after_resolution(date(2027, 1, 1))}
    assert ids == {"C-early", "C-later"}


def test_forecast_day_is_when_the_row_was_written() -> None:
    book = parse_ledger(_FIXTURE)
    early = next(c for c in book.claims if c.id == "C-early")
    assert early.forecast_day == date(2026, 8, 18)
    assert early.owner == "nato-watcher"
    assert early.claim == "NATO will not invoke article 5 in 2026."


def test_missing_resolution_is_an_error() -> None:
    text = _FIXTURE.replace("Resolution: 2026-12-31\n\n", "")
    try:
        parse_ledger(text)
    except ValueError as exc:
        assert "missing Resolution" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_repo_ledger_is_the_schedule_book() -> None:
    path = REPO_ROOT / "ledger.md"
    book = parse_ledger(path.read_text(encoding="utf-8"))
    assert book.k == 1
    assert book.problems
    assert all(problem.motivation and problem.resolution_day for problem in book.problems)
    for claim in book.claims:
        assert claim.owner and claim.claim and claim.justification and claim.forecast_day
        assert book.problem(claim.problem_id) is not None
