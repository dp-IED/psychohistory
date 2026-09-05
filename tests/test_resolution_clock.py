from __future__ import annotations

from datetime import date
from pathlib import Path

from harness.ledger import parse_ledger
from scripts.resolution_clock import dates_in_claim, later_than_resolution, main

REPO_ROOT = Path(__file__).resolve().parents[1]

_FROZEN = """\
# Ledger

K: 1

## Problems

### P-usca-338 — US Section 338 tariffs on Canada

Resolution: 2026-08-19

Motivation: First administrative deadline.

## Claims

### C-usca-338-deal-announced

- Problem: P-usca-338
- Forecast: 2026-08-19
- Owner: claim-worker
- Claim: Duties were postponed to 12:01 a.m. ET on 22 Aug 2026.
- Justification: Analog episode 1 Jun 2018 is not the answering date.
"""


def test_claim_line_parses_iso_and_english_dates() -> None:
    assert dates_in_claim("no date") == ()
    assert dates_in_claim("postpone to 2026-08-22") == (date(2026, 8, 22),)
    assert dates_in_claim("postponed to 22 Aug 2026") == (date(2026, 8, 22),)
    assert dates_in_claim("until August 22, 2026") == (date(2026, 8, 22),)


def test_justification_dates_are_ignored() -> None:
    book = parse_ledger(_FROZEN)
    rows = later_than_resolution(book)
    assert len(rows) == 1
    assert rows[0].named_day == date(2026, 8, 22)
    assert rows[0].resolution_day == date(2026, 8, 19)


def test_same_day_as_resolution_is_not_a_slide() -> None:
    text = _FROZEN.replace("Resolution: 2026-08-19", "Resolution: 2026-08-22")
    book = parse_ledger(text)
    assert later_than_resolution(book) == ()


def test_cli_flags_a_frozen_heading(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    path.write_text(_FROZEN, encoding="utf-8")
    assert main([str(path)]) == 1


def test_repo_ledger_has_no_unnamed_later_clock() -> None:
    assert main([str(REPO_ROOT / "ledger.md")]) == 0


def test_script_file_finds_harness_without_pythonpath() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "resolution_clock.py"),
            str(REPO_ROOT / "ledger.md"),
        ],
        cwd="/tmp",
        check=False,
    )
    assert completed.returncode == 0
