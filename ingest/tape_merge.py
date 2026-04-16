"""Merge normalized event tapes into a mixed-source tape."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.gdelt_raw import format_datetime_z
from ingest.io_utils import open_text_auto


UTC = dt.timezone.utc


def _record_sort_key(record: EventTapeRecord) -> tuple[dt.datetime, dt.date, str, str]:
    return (
        record.source_available_at.astimezone(UTC),
        record.event_date,
        record.source_name,
        record.source_event_id,
    )


def merge_event_tapes(
    *,
    tape_paths: Sequence[Path],
    out_path: Path,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not tape_paths:
        raise ValueError("at least one input tape is required")

    records: list[EventTapeRecord] = []
    input_counts: dict[str, int] = {}
    for path in tape_paths:
        tape_records = load_event_tape(path, allow_empty=allow_empty)
        input_counts[str(path)] = len(tape_records)
        records.extend(tape_records)

    deduped: dict[tuple[str, str], EventTapeRecord] = {}
    duplicate_count = 0
    for record in sorted(records, key=_record_sort_key):
        key = (record.source_name, record.source_event_id)
        if key in deduped:
            duplicate_count += 1
            continue
        deduped[key] = record

    output_records = sorted(deduped.values(), key=_record_sort_key)
    if not output_records and not allow_empty:
        raise ValueError("merged event tape is empty")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for record in output_records:
            handle.write(record.model_dump_json() + "\n")

    event_dates = [record.event_date for record in output_records]
    source_times = [record.source_available_at.astimezone(UTC) for record in output_records]
    source_counts = Counter(record.source_name for record in output_records)
    audit = {
        "input_paths": [str(path) for path in tape_paths],
        "input_counts": input_counts,
        "output_path": str(out_path),
        "output_row_count": len(output_records),
        "duplicate_count": duplicate_count,
        "source_counts": dict(sorted(source_counts.items())),
        "earliest_event_date": min(event_dates).isoformat() if event_dates else None,
        "latest_event_date": max(event_dates).isoformat() if event_dates else None,
        "earliest_source_available_at": format_datetime_z(min(source_times))
        if source_times
        else None,
        "latest_source_available_at": format_datetime_z(max(source_times))
        if source_times
        else None,
    }
    out_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", action="append", required=True, help="Input event tape path.")
    merge.add_argument("--out", required=True)
    merge.add_argument("--allow-empty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "merge":
        try:
            merge_event_tapes(
                tape_paths=[Path(value) for value in args.input],
                out_path=Path(args.out),
                allow_empty=args.allow_empty,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
