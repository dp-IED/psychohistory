"""Run baseline backtests over normalized event tapes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from baselines.metrics import brier_score, mean_absolute_error, top_k_hit_rate
from baselines.recurrence import (
    RECURRENCE_MODEL_NAMES,
    ForecastRow,
    build_recurrence_forecasts_for_origin,
)
from ingest.event_tape import load_event_tape


def weekly_origins(start: dt.date, end: dt.date) -> list[dt.date]:
    if start.weekday() != 0:
        raise ValueError(f"origin_start must be a Monday: {start.isoformat()}")
    if end.weekday() != 0:
        raise ValueError(f"origin_end must be a Monday: {end.isoformat()}")
    if start > end:
        raise ValueError(
            f"origin_start must be on or before origin_end: {start.isoformat()} > {end.isoformat()}"
        )
    origins: list[dt.date] = []
    current = start
    while current <= end:
        origins.append(current)
        current += dt.timedelta(days=7)
    return origins


def build_recurrence_backtest_rows(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
) -> list[ForecastRow]:
    records = load_event_tape(tape_path)
    rows: list[ForecastRow] = []
    for forecast_origin in weekly_origins(origin_start, origin_end):
        rows.extend(
            build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=forecast_origin,
            )
        )
    return rows


def build_audit(
    rows: list[ForecastRow],
    *,
    origin_start: dt.date,
    origin_end: dt.date,
) -> dict[str, Any]:
    model_names = [
        model_name
        for model_name in RECURRENCE_MODEL_NAMES
        if any(row.model_name == model_name for row in rows)
    ]
    return {
        "row_count": len(rows),
        "origin_start": origin_start.isoformat(),
        "origin_end": origin_end.isoformat(),
        "model_names": model_names,
        "admin1_count": len({row.admin1_code for row in rows}),
        "mean_brier_by_model": {
            model_name: brier_score(rows, model_name)
            for model_name in model_names
        },
        "mean_absolute_error_by_model": {
            model_name: mean_absolute_error(rows, model_name)
            for model_name in model_names
        },
        "top5_hit_rate_by_model": {
            model_name: top_k_hit_rate(rows, model_name, k=5)
            for model_name in model_names
        },
    }


def run_recurrence_backtest(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
    out_path: Path,
) -> dict[str, Any]:
    rows = build_recurrence_backtest_rows(
        tape_path=tape_path,
        origin_start=origin_start,
        origin_end=origin_end,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    audit = build_audit(rows, origin_start=origin_start, origin_end=origin_end)
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recurrence = subparsers.add_parser("recurrence")
    recurrence.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    recurrence.add_argument("--origin-start", default="2021-01-04")
    recurrence.add_argument("--origin-end", default="2025-12-29")
    recurrence.add_argument(
        "--out",
        default="data/gdelt/baselines/france_protest/recurrence_predictions.jsonl",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "recurrence":
        try:
            run_recurrence_backtest(
                tape_path=Path(args.tape),
                origin_start=dt.date.fromisoformat(args.origin_start),
                origin_end=dt.date.fromisoformat(args.origin_end),
                out_path=Path(args.out),
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
