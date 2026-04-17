"""Build weekly historical injection manifests over the event tape."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from ingest.event_records import load_event_records
from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.event_warehouse import query_records
from ingest.paths import resolve_data_root, warehouse_path as warehouse_db_path
from ingest.gdelt_raw import parse_datetime_utc


UTC = dt.timezone.utc


class HistoricalInjectionBatch(BaseModel):
    batch_id: str
    source_name: Literal["gdelt_v2_events"]
    source_available_start: dt.datetime
    source_available_end: dt.datetime
    record_count: int
    input_path: str
    content_sha256: str


def _format_datetime_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_week_start(value: dt.datetime) -> dt.datetime:
    utc_value = value.astimezone(UTC)
    monday = utc_value.date() - dt.timedelta(days=utc_value.weekday())
    return dt.datetime.combine(monday, dt.time(), tzinfo=UTC)


def _record_sort_key(record: EventTapeRecord) -> tuple[dt.datetime, dt.date, str]:
    return (
        record.source_available_at.astimezone(UTC),
        record.event_date,
        record.source_event_id,
    )


def _canonical_records_sha256(records: list[EventTapeRecord]) -> str:
    payload = [
        record.model_dump(mode="json")
        for record in sorted(records, key=_record_sort_key)
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_batches(
    *,
    tape_path: Path | None = None,
    warehouse_path: Path | None = None,
    data_root: Path | None = None,
    out_path: Path,
) -> list[HistoricalInjectionBatch]:
    records = load_event_records(
        tape_path=tape_path,
        warehouse_db_path=warehouse_path,
        data_root=data_root,
    )
    resolved_db = (
        Path(warehouse_path).expanduser().resolve()
        if warehouse_path is not None
        else warehouse_db_path(resolve_data_root(data_root))
    )
    resolved_input = str(tape_path) if tape_path is not None else str(resolved_db)
    grouped: dict[dt.datetime, list[EventTapeRecord]] = {}
    for record in records:
        start = source_week_start(record.source_available_at)
        grouped.setdefault(start, []).append(record)

    batches: list[HistoricalInjectionBatch] = []
    for start in sorted(grouped):
        end = start + dt.timedelta(days=7)
        start_date = start.date().isoformat()
        batch_records = grouped[start]
        batches.append(
            HistoricalInjectionBatch(
                batch_id=f"gdelt_v2_events__source_week__{start_date}",
                source_name="gdelt_v2_events",
                source_available_start=start,
                source_available_end=end,
                record_count=len(batch_records),
                input_path=resolved_input,
                content_sha256=_canonical_records_sha256(batch_records),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for batch in batches:
            handle.write(batch.model_dump_json() + "\n")
    return batches


def load_batches(
    path: Path,
    *,
    allow_missing: bool = False,
    allow_empty: bool = False,
) -> list[HistoricalInjectionBatch]:
    batches: list[HistoricalInjectionBatch] = []
    if not path.exists():
        if allow_missing:
            return batches
        raise FileNotFoundError(f"missing batch manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                batches.append(HistoricalInjectionBatch.model_validate_json(line))
    if not batches and not allow_empty:
        raise ValueError(f"batch manifest is empty: {path}")
    return batches


def _load_records_for_batch_input(path_str: str) -> list[EventTapeRecord]:
    path = Path(path_str)
    if path.suffix.lower() == ".duckdb":
        return query_records(db_path=path)
    return load_event_tape(path)


def replay_records_for_cutoff(
    *,
    batches_path: Path,
    cutoff: dt.datetime | str,
) -> list[EventTapeRecord]:
    cutoff_dt = parse_datetime_utc(cutoff) if isinstance(cutoff, str) else cutoff.astimezone(UTC)
    selected_batches = [
        batch
        for batch in load_batches(batches_path)
        if batch.source_available_start.astimezone(UTC) < cutoff_dt
    ]
    records_by_path: dict[str, list[EventTapeRecord]] = {}
    replayed: list[EventTapeRecord] = []
    seen_ids: set[str] = set()
    for batch in selected_batches:
        if batch.input_path not in records_by_path:
            records_by_path[batch.input_path] = _load_records_for_batch_input(batch.input_path)
        records = records_by_path[batch.input_path]
        start = batch.source_available_start.astimezone(UTC)
        end = batch.source_available_end.astimezone(UTC)
        for record in records:
            available_at = record.source_available_at.astimezone(UTC)
            if start <= available_at < end and available_at < cutoff_dt:
                if record.source_event_id not in seen_ids:
                    seen_ids.add(record.source_event_id)
                    replayed.append(record)
    return sorted(replayed, key=_record_sort_key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-batches")
    build.add_argument(
        "--tape",
        default=None,
        help=(
            "JSONL event tape. If omitted, load from the DuckDB warehouse "
            "at <data-root>/warehouse/events.duckdb (see PSYCHOHISTORY_DATA_ROOT)."
        ),
    )
    build.add_argument(
        "--data-root",
        default=None,
        help="Root directory for data/ layout; sets warehouse path when --tape is omitted.",
    )
    build.add_argument(
        "--warehouse-path",
        default=None,
        help="Path to events.duckdb (overrides --data-root when --tape is omitted).",
    )
    build.add_argument("--out", default="data/gdelt/injection/france_protest/batches.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-batches":
        try:
            build_batches(
                tape_path=Path(args.tape) if args.tape else None,
                warehouse_path=Path(args.warehouse_path) if args.warehouse_path else None,
                data_root=Path(args.data_root) if args.data_root else None,
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
