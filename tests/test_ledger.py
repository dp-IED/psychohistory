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

Motivation: Markets will score alliance cohesion; we need a problem row before discovery exists.

## Claims

### C-due

- Problem: P-nato
- Due: 2026-08-18
- Owner: nato-watcher
- Claim: NATO will not invoke article 5 in 2026.
- Justification: No article 5 trigger is on the board this year.

### C-later

- Problem: P-nato
- Due: 2026-12-31
- Owner: nato-watcher
- Claim: Alliance cohesion holds through year-end.
- Justification: Year-end is a different wakeup.
"""


def test_ledger_reports_k() -> None:
    book = parse_ledger(_FIXTURE)
    assert book.k == 1


def test_ledger_lists_problems_with_motivation() -> None:
    book = parse_ledger(_FIXTURE)
    assert len(book.problems) == 1
    problem = book.problems[0]
    assert problem.id == "P-nato"
    assert problem.title == "NATO article 5 this year"
    assert problem.motivation == (
        "Markets will score alliance cohesion; we need a problem row before discovery exists."
    )


def test_due_today_returns_claims_scheduled_on_as_of_date() -> None:
    book = parse_ledger(_FIXTURE)
    due = book.due_today(date(2026, 8, 18))
    assert len(due) == 1
    claim = due[0]
    assert claim.id == "C-due"
    assert claim.owner == "nato-watcher"
    assert claim.due == date(2026, 8, 18)
    assert claim.claim == "NATO will not invoke article 5 in 2026."
    assert claim.justification == "No article 5 trigger is on the board this year."


def test_due_today_excludes_claims_due_on_another_day() -> None:
    book = parse_ledger(_FIXTURE)
    due = book.due_today(date(2026, 8, 18))
    assert [c.id for c in due] == ["C-due"]
    assert book.due_today(date(2026, 1, 1)) == ()


def test_repo_ledger_is_the_schedule_book() -> None:
    path = REPO_ROOT / "ledger.md"
    book = parse_ledger(path.read_text(encoding="utf-8"))
    assert book.k == 1
    assert book.problems
    assert all(problem.motivation for problem in book.problems)
    assert book.claims
    assert all(
        claim.owner and claim.claim and claim.justification for claim in book.claims
    )
