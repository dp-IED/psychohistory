"""Normalize filtered GDELT raw fragments into a point-in-time event tape."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from ingest.io_utils import open_text_auto
from ingest.gdelt_raw import parse_datetime_utc, parse_sql_date


UTC = dt.timezone.utc


class EventTapeRecord(BaseModel):
    source_name: Literal["gdelt_v2_events", "acled", "acled_v3"]
    source_event_id: str
    event_date: dt.date
    source_available_at: dt.datetime
    retrieved_at: dt.datetime
    country_code: str
    admin1_code: str
    location_name: str | None
    latitude: float | None
    longitude: float | None
    event_class: Literal["protest"]
    event_code: str
    event_base_code: str
    event_root_code: str
    quad_class: int | None
    goldstein_scale: float | None
    num_mentions: int | None
    num_sources: int | None
    num_articles: int | None
    avg_tone: float | None
    actor1_name: str | None
    actor1_country_code: str | None
    actor2_name: str | None
    actor2_country_code: str | None
    source_url: str | None
    raw: dict[str, Any]


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(row: dict[str, Any], key: str) -> str:
    value = _none_if_blank(row.get(key))
    if value is None:
        raise ValueError(f"missing required field: {key}")
    return value


def _optional_int(row: dict[str, Any], key: str) -> int | None:
    value = _none_if_blank(row.get(key))
    if value is None:
        return None
    return int(value)


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = _none_if_blank(row.get(key))
    if value is None:
        return None
    return float(value)


def _format_datetime_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _event_sort_key(record: EventTapeRecord) -> tuple[dt.datetime, dt.date, str]:
    return (
        record.source_available_at.astimezone(UTC),
        record.event_date,
        record.source_event_id,
    )


def _dedupe_tie_key(record: EventTapeRecord) -> tuple[dt.datetime, str, str, str]:
    return (
        record.source_available_at.astimezone(UTC),
        record.source_event_id,
        str(record.raw.get("_source_file_timestamp") or ""),
        str(record.raw.get("_source_file_url") or ""),
    )


def _manifest_fragment_paths(raw_dir: Path, *, allow_partial: bool = False) -> list[Path]:
    manifest_path = raw_dir / "fetch_manifest.jsonl"
    if not manifest_path.exists():
        return []

    metadata_path = raw_dir / "fetch_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing fetch metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = metadata.get("run_id")
    if not run_id:
        raise ValueError("fetch metadata is missing run_id; rerun the raw fetcher")
    if (
        int(metadata.get("failed_file_count") or 0)
        and not metadata.get("allow_partial")
        and not allow_partial
    ):
        raise ValueError("raw fetch has failed files; rerun fetch or use --allow-partial")

    paths: list[Path] = []
    seen: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                continue
            if row.get("status") not in {"ok", "skipped"}:
                continue
            fragment = row.get("fragment_path")
            if not fragment:
                continue
            path = Path(fragment)
            if not path.is_absolute():
                path = raw_dir / path
            if path not in seen:
                paths.append(path)
                seen.add(path)

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manifest references missing fragment: {missing[0]}")
    return sorted(paths)


def _raw_fragment_paths(raw_dir: Path, *, allow_partial: bool = False) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"missing raw directory: {raw_dir}")
    manifest_paths = _manifest_fragment_paths(raw_dir, allow_partial=allow_partial)
    if manifest_paths:
        return manifest_paths
    if (raw_dir / "fetch_manifest.jsonl").exists():
        return []
    fragment_dir = raw_dir / "fragments"
    return sorted([*fragment_dir.glob("**/*.jsonl"), *fragment_dir.glob("**/*.jsonl.gz")])


def normalize_raw_row(
    row: dict[str, Any],
    *,
    event_start: dt.date = dt.date(2019, 1, 1),
    event_end: dt.date = dt.date(2026, 1, 4),
) -> EventTapeRecord | None:
    if row.get("ActionGeo_CountryCode") != "FR":
        return None
    if row.get("EventRootCode") != "14":
        return None
    event_date = parse_sql_date(_required_text(row, "SQLDATE"))
    if not (event_start <= event_date <= event_end):
        return None

    admin1_code = _none_if_blank(row.get("ActionGeo_ADM1Code")) or "FR_UNKNOWN"
    source_event_id = f"gdelt:{_required_text(row, 'GLOBALEVENTID')}"
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=parse_datetime_utc(_required_text(row, "DATEADDED")),
        retrieved_at=parse_datetime_utc(_required_text(row, "_retrieved_at")),
        country_code="FR",
        admin1_code=admin1_code,
        location_name=_none_if_blank(row.get("ActionGeo_FullName")),
        latitude=_optional_float(row, "ActionGeo_Lat"),
        longitude=_optional_float(row, "ActionGeo_Long"),
        event_class="protest",
        event_code=_required_text(row, "EventCode"),
        event_base_code=_required_text(row, "EventBaseCode"),
        event_root_code=_required_text(row, "EventRootCode"),
        quad_class=_optional_int(row, "QuadClass"),
        goldstein_scale=_optional_float(row, "GoldsteinScale"),
        num_mentions=_optional_int(row, "NumMentions"),
        num_sources=_optional_int(row, "NumSources"),
        num_articles=_optional_int(row, "NumArticles"),
        avg_tone=_optional_float(row, "AvgTone"),
        actor1_name=_none_if_blank(row.get("Actor1Name")),
        actor1_country_code=_none_if_blank(row.get("Actor1CountryCode")),
        actor2_name=_none_if_blank(row.get("Actor2Name")),
        actor2_country_code=_none_if_blank(row.get("Actor2CountryCode")),
        source_url=_none_if_blank(row.get("SOURCEURL")),
        raw=row,
    )


def _iter_raw_fragment_rows(
    raw_dir: Path,
    *,
    allow_empty: bool,
    allow_partial: bool = False,
) -> tuple[int, list[EventTapeRecord], int, int]:
    input_count = 0
    filtered_count = 0
    invalid_count = 0
    records: list[EventTapeRecord] = []
    fragment_paths = _raw_fragment_paths(raw_dir, allow_partial=allow_partial)
    if not fragment_paths and not allow_empty:
        raise ValueError(f"no raw GDELT fragments found under {raw_dir}")
    for path in fragment_paths:
        with open_text_auto(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                input_count += 1
                row = json.loads(line)
                try:
                    record = normalize_raw_row(row)
                except (TypeError, ValueError):
                    invalid_count += 1
                    continue
                if record is None:
                    filtered_count += 1
                    continue
                records.append(record)
    if input_count == 0 and not allow_empty:
        raise ValueError(f"no raw GDELT rows found under {raw_dir}")
    return input_count, records, filtered_count, invalid_count


def write_event_tape(
    *,
    raw_dir: Path,
    out_path: Path,
    allow_empty: bool = False,
    allow_partial: bool = False,
    compress: bool = False,
) -> dict[str, Any]:
    input_count, records, filtered_count, invalid_count = _iter_raw_fragment_rows(
        raw_dir,
        allow_empty=allow_empty,
        allow_partial=allow_partial,
    )

    deduped: dict[str, EventTapeRecord] = {}
    duplicate_count = 0
    for record in sorted(records, key=_dedupe_tie_key):
        existing = deduped.get(record.source_event_id)
        if existing is None:
            deduped[record.source_event_id] = record
            continue
        duplicate_count += 1
        if _dedupe_tie_key(record) < _dedupe_tie_key(existing):
            deduped[record.source_event_id] = record

    output_records = sorted(deduped.values(), key=_event_sort_key)
    if not output_records and not allow_empty:
        raise ValueError("event tape is empty after normalization")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress and out_path.suffix != ".gz":
        output_handle = gzip.open(out_path, "wt", encoding="utf-8")
    else:
        output_handle = open_text_auto(out_path, "w")
    with output_handle as handle:
        for record in output_records:
            handle.write(record.model_dump_json() + "\n")

    event_dates = [record.event_date for record in output_records]
    source_times = [record.source_available_at.astimezone(UTC) for record in output_records]
    audit = {
        "input_row_count": input_count,
        "output_row_count": len(output_records),
        "filtered_count": filtered_count,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "earliest_event_date": min(event_dates).isoformat() if event_dates else None,
        "latest_event_date": max(event_dates).isoformat() if event_dates else None,
        "earliest_source_available_at": _format_datetime_z(min(source_times))
        if source_times
        else None,
        "latest_source_available_at": _format_datetime_z(max(source_times))
        if source_times
        else None,
    }
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def load_event_tape(
    path: Path,
    *,
    allow_missing: bool = False,
    allow_empty: bool = False,
) -> list[EventTapeRecord]:
    records: list[EventTapeRecord] = []
    if not path.exists():
        if allow_missing:
            return records
        raise FileNotFoundError(f"missing event tape: {path}")
    with open_text_auto(path, "r") as handle:
        for line in handle:
            if line.strip():
                records.append(EventTapeRecord.model_validate_json(line))
    if not records and not allow_empty:
        raise ValueError(f"event tape is empty: {path}")
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalize = subparsers.add_parser("normalize-france-protests")
    normalize.add_argument("--raw", default="data/gdelt/raw/france_protest")
    normalize.add_argument("--out", default="data/gdelt/tape/france_protest/events.jsonl")
    normalize.add_argument("--allow-empty", action="store_true")
    normalize.add_argument("--allow-partial", action="store_true")
    normalize.add_argument("--compress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "normalize-france-protests":
        try:
            write_event_tape(
                raw_dir=Path(args.raw),
                out_path=Path(args.out),
                allow_empty=args.allow_empty,
                allow_partial=args.allow_partial,
                compress=args.compress,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
