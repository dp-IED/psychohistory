from __future__ import annotations

from datetime import date
from pathlib import Path

from harness.ledger import parse_ledger
from scripts.pause_recap import main, pause_recaps

REPO_ROOT = Path(__file__).resolve().parents[1]

_RECAP = """\
# Ledger

K: 1

## Problems

### P-usca-338 — US Section 338 tariffs on Canada

Resolution: 2026-08-22

Motivation: Post-pause clock.

## Claims

### C-usca-338-deal-announced

- Problem: P-usca-338
- Forecast: 2026-08-19
- Owner: claim-worker
- Claim: Before 12:01 a.m. ET on 19 Aug 2026, the United States and Canada publicly announced an arrangement that delayed (did not cancel) the scheduled 50% Section 338 tariffs on listed Canadian goods; the duties did not take effect at that deadline and were postponed to 12:01 a.m. ET on 22 Aug 2026.
- Justification: Analog episode 1 Jun 2018 is not the answering date.
"""

_TAKE_EFFECT = _RECAP.replace(
    "Before 12:01 a.m. ET on 19 Aug 2026, the United States and Canada publicly announced an arrangement that delayed (did not cancel) the scheduled 50% Section 338 tariffs on listed Canadian goods; the duties did not take effect at that deadline and were postponed to 12:01 a.m. ET on 22 Aug 2026.",
    "The 50% Section 338 duties on listed Canadian goods take effect at 12:01 a.m. ET on 22 Aug 2026.",
).replace("Forecast: 2026-08-19", "Forecast: 2026-08-20")

_FIRST_CLOCK_DELAY = """\
# Ledger

K: 1

## Problems

### P-232-poly-04 — Section 232 polysilicon MIP and duties take effect

Resolution: 2026-12-04

Motivation: First administrative deadline.

## Claims

### C-232-poly-04-delay

- Problem: P-232-poly-04
- Forecast: 2026-08-25
- Owner: claim-worker
- Claim: The Section 232 polysilicon MIP and 15% ad valorem on listed derivatives will not apply as written at 12:01 a.m. ET 4 Dec 2026; a new US suspend/amend will delay that morning's take-effect.
- Justification: First clock.
"""


def test_pause_recap_flags_live_latest_row() -> None:
    book = parse_ledger(_RECAP)
    rows = pause_recaps(book, date(2026, 8, 20))
    assert len(rows) == 1
    assert rows[0].claim_id == "C-usca-338-deal-announced"
    assert rows[0].named_day == date(2026, 8, 22)


def test_past_series_is_not_flagged() -> None:
    book = parse_ledger(_RECAP)
    assert pause_recaps(book, date(2026, 8, 23)) == ()


def test_take_effect_modal_is_not_a_recap() -> None:
    book = parse_ledger(_TAKE_EFFECT)
    assert pause_recaps(book, date(2026, 8, 20)) == ()


def test_first_clock_delay_forecast_is_not_a_recap() -> None:
    book = parse_ledger(_FIRST_CLOCK_DELAY)
    assert pause_recaps(book, date(2026, 8, 26)) == ()


def test_cli_flags_a_live_recap(tmp_path: Path) -> None:
    path = tmp_path / "ledger.md"
    path.write_text(_RECAP, encoding="utf-8")
    assert main([str(path), "--as-of", "2026-08-20"]) == 1
    assert main([str(path), "--as-of", "2026-08-23"]) == 0


def test_repo_ledger_has_no_live_pause_recap() -> None:
    assert main([str(REPO_ROOT / "ledger.md"), "--as-of", "2026-09-06"]) == 0


def test_script_file_finds_harness_without_pythonpath() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "pause_recap.py"),
            str(REPO_ROOT / "ledger.md"),
            "--as-of",
            "2026-09-06",
        ],
        cwd="/tmp",
        check=False,
    )
    assert completed.returncode == 0
