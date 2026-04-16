"""Normalize ACLED raw fragments into the shared event tape."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Sequence

from ingest.event_tape import EventTapeRecord
from ingest.gdelt_raw import format_datetime_z, parse_datetime_utc
from ingest.io_utils import open_text_auto


UTC = dt.timezone.utc
SOURCE_NAME = "acled"
AvailabilityPolicy = Literal["timestamp", "event_date_lag", "retrieved_at"]


FRANCE_ADMIN1_NAME_TO_GDELT_CODE = {
    "alsace": "FRC1",
    "aquitaine": "FR97",
    "auvergne": "FR98",
    "basse normandie": "FR99",
    "bourgogne": "FRA1",
    "bretagne": "FRA2",
    "burgundy": "FRA1",
    "centre": "FRA3",
    "centre val de loire": "FRA3",
    "champagne ardenne": "FRA4",
    "corse": "FRA5",
    "corsica": "FRA5",
    "franche comte": "FRA6",
    "haute normandie": "FRA7",
    "ile de france": "FRA8",
    "languedoc roussillon": "FRA9",
    "limousin": "FRB1",
    "lorraine": "FRB2",
    "midi pyrenees": "FRB3",
    "nord pas de calais": "FRB4",
    "pays de la loire": "FRB5",
    "picardie": "FRB6",
    "poitou charentes": "FRB7",
    "provence alpes cote d azur": "FRB8",
    "rhone alpes": "FRB9",
}

FRANCE_ADMIN2_NAME_TO_GDELT_CODE = {
    "ain": "FRB9",
    "aisne": "FRB6",
    "allier": "FR98",
    "alpes de haute provence": "FRB8",
    "alpes maritimes": "FRB8",
    "ardeche": "FRB9",
    "ardennes": "FRA4",
    "ariege": "FRB3",
    "aube": "FRA4",
    "aude": "FRA9",
    "aveyron": "FRB3",
    "bas rhin": "FRC1",
    "bouches du rhone": "FRB8",
    "calvados": "FR99",
    "cantal": "FR98",
    "charente": "FRB7",
    "charente maritime": "FRB7",
    "cher": "FRA3",
    "correze": "FRB1",
    "corse du sud": "FRA5",
    "cote d or": "FRA1",
    "cotes d armor": "FRA2",
    "creuse": "FRB1",
    "deux sevres": "FRB7",
    "dordogne": "FR97",
    "doubs": "FRA6",
    "drome": "FRB9",
    "essonne": "FRA8",
    "eure": "FRA7",
    "eure et loir": "FRA3",
    "finistere": "FRA2",
    "gard": "FRA9",
    "gers": "FRB3",
    "gironde": "FR97",
    "haute corse": "FRA5",
    "haute garonne": "FRB3",
    "haute loire": "FR98",
    "haute marne": "FRA4",
    "haute saone": "FRA6",
    "haute savoie": "FRB9",
    "haute vienne": "FRB1",
    "hautes alpes": "FRB8",
    "hautes pyrenees": "FRB3",
    "hauts de seine": "FRA8",
    "herault": "FRA9",
    "ille et vilaine": "FRA2",
    "indre": "FRA3",
    "indre et loire": "FRA3",
    "isere": "FRB9",
    "jura": "FRA6",
    "landes": "FR97",
    "loir et cher": "FRA3",
    "loire": "FRB9",
    "loire atlantique": "FRB5",
    "loiret": "FRA3",
    "lot": "FRB3",
    "lot et garonne": "FR97",
    "lozere": "FRA9",
    "maine et loire": "FRB5",
    "manche": "FR99",
    "marne": "FRA4",
    "mayenne": "FRB5",
    "meurthe et moselle": "FRB2",
    "meuse": "FRB2",
    "morbihan": "FRA2",
    "moselle": "FRB2",
    "nievre": "FRA1",
    "nord": "FRB4",
    "oise": "FRB6",
    "orne": "FR99",
    "paris": "FRA8",
    "pas de calais": "FRB4",
    "puy de dome": "FR98",
    "pyrenees atlantiques": "FR97",
    "pyrenees orientales": "FRA9",
    "rhone": "FRB9",
    "saone et loire": "FRA1",
    "sarthe": "FRB5",
    "savoie": "FRB9",
    "seine et marne": "FRA8",
    "seine maritime": "FRA7",
    "seine saint denis": "FRA8",
    "somme": "FRB6",
    "tarn": "FRB3",
    "tarn et garonne": "FRB3",
    "territoire de belfort": "FRA6",
    "val d oise": "FRA8",
    "val de marne": "FRA8",
    "var": "FRB8",
    "vaucluse": "FRB8",
    "vendee": "FRB5",
    "vienne": "FRB7",
    "vosges": "FRB2",
    "yonne": "FRA1",
    "yvelines": "FRA8",
}

CURRENT_ADMIN1_FALLBACK_TO_GDELT_CODE = {
    "auvergne rhone alpes": "FRB9",
    "bourgogne franche comte": "FRA1",
    "grand est": "FRB2",
    "hauts de france": "FRB4",
    "normandie": "FRA7",
    "nouvelle aquitaine": "FR97",
    "occitanie": "FRB3",
    "provence alpes cote d azur": "FRB8",
}


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
    return int(float(value))


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = _none_if_blank(row.get(key))
    if value is None:
        return None
    return float(value)


def _normalized_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in ascii_text)
    return " ".join(cleaned.split())


def acled_admin_to_gdelt_admin1_code(admin1: str | None, admin2: str | None = None) -> str:
    admin2_key = _normalized_key(admin2)
    if admin2_key in FRANCE_ADMIN2_NAME_TO_GDELT_CODE:
        return FRANCE_ADMIN2_NAME_TO_GDELT_CODE[admin2_key]
    admin1_key = _normalized_key(admin1)
    if admin1_key in FRANCE_ADMIN1_NAME_TO_GDELT_CODE:
        return FRANCE_ADMIN1_NAME_TO_GDELT_CODE[admin1_key]
    if admin1_key in CURRENT_ADMIN1_FALLBACK_TO_GDELT_CODE:
        return CURRENT_ADMIN1_FALLBACK_TO_GDELT_CODE[admin1_key]
    return "FR_UNKNOWN"


def parse_acled_timestamp(value: Any, *, fallback: dt.datetime) -> dt.datetime:
    text = _none_if_blank(value)
    if text is None:
        return fallback
    try:
        numeric = float(text)
    except ValueError:
        try:
            return parse_datetime_utc(text)
        except ValueError:
            return fallback
    if numeric > 10_000_000_000:
        numeric /= 1000.0
    return dt.datetime.fromtimestamp(numeric, tz=UTC)


def acled_source_available_at(
    row: dict[str, Any],
    *,
    event_date: dt.date,
    retrieved_at: dt.datetime,
    availability_policy: AvailabilityPolicy,
    availability_lag_days: int,
) -> dt.datetime:
    if availability_policy == "timestamp":
        return parse_acled_timestamp(row.get("timestamp"), fallback=retrieved_at)
    if availability_policy == "event_date_lag":
        return dt.datetime.combine(
            event_date + dt.timedelta(days=availability_lag_days),
            dt.time(),
            tzinfo=UTC,
        )
    if availability_policy == "retrieved_at":
        return retrieved_at
    raise ValueError(f"unknown ACLED availability policy: {availability_policy}")


def normalize_acled_row(
    row: dict[str, Any],
    *,
    event_start: dt.date = dt.date(2019, 1, 1),
    event_end: dt.date = dt.date(2026, 1, 4),
    availability_policy: AvailabilityPolicy = "event_date_lag",
    availability_lag_days: int = 7,
) -> EventTapeRecord | None:
    if _none_if_blank(row.get("country")) != "France":
        return None
    if _none_if_blank(row.get("event_type")) != "Protests":
        return None
    event_date = dt.date.fromisoformat(_required_text(row, "event_date"))
    if not (event_start <= event_date <= event_end):
        return None

    retrieved_at = parse_datetime_utc(_required_text(row, "_retrieved_at"))
    source_event_id = f"acled:{_required_text(row, 'event_id_cnty')}"
    admin1_code = acled_admin_to_gdelt_admin1_code(
        _none_if_blank(row.get("admin1")),
        _none_if_blank(row.get("admin2")),
    )
    return EventTapeRecord(
        source_name=SOURCE_NAME,
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=acled_source_available_at(
            row,
            event_date=event_date,
            retrieved_at=retrieved_at,
            availability_policy=availability_policy,
            availability_lag_days=availability_lag_days,
        ),
        retrieved_at=retrieved_at,
        country_code="FR",
        admin1_code=admin1_code,
        location_name=_none_if_blank(row.get("location"))
        or _none_if_blank(row.get("admin2"))
        or _none_if_blank(row.get("admin1")),
        latitude=_optional_float(row, "latitude"),
        longitude=_optional_float(row, "longitude"),
        event_class="protest",
        event_code=_required_text(row, "event_type"),
        event_base_code=_required_text(row, "event_type"),
        event_root_code=_required_text(row, "event_type"),
        quad_class=None,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name=_none_if_blank(row.get("actor1")),
        actor1_country_code=None,
        actor2_name=_none_if_blank(row.get("actor2")),
        actor2_country_code=None,
        source_url=None,
        raw=row,
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
        raise ValueError("fetch metadata is missing run_id; rerun ACLED fetcher")
    if int(metadata.get("failed_page_count") or 0) and not allow_partial:
        raise ValueError("ACLED raw fetch has failed pages; rerun fetch or use --allow-partial")

    paths: list[Path] = []
    seen: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id or row.get("status") != "ok":
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
    fragment_dir = raw_dir / "fragments"
    return sorted([*fragment_dir.glob("*.jsonl"), *fragment_dir.glob("*.jsonl.gz")])


def write_acled_event_tape(
    *,
    raw_dir: Path,
    out_path: Path,
    allow_empty: bool = False,
    allow_partial: bool = False,
    event_start: dt.date = dt.date(2019, 1, 1),
    event_end: dt.date = dt.date(2026, 1, 4),
    availability_policy: AvailabilityPolicy = "event_date_lag",
    availability_lag_days: int = 7,
) -> dict[str, Any]:
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be non-negative")
    fragment_paths = _raw_fragment_paths(raw_dir, allow_partial=allow_partial)
    if not fragment_paths and not allow_empty:
        raise ValueError(f"no raw ACLED fragments found under {raw_dir}")

    input_count = 0
    filtered_count = 0
    invalid_count = 0
    records: list[EventTapeRecord] = []
    unresolved_admin1_counts: Counter[str] = Counter()
    for path in fragment_paths:
        with open_text_auto(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                input_count += 1
                row = json.loads(line)
                try:
                    record = normalize_acled_row(
                        row,
                        event_start=event_start,
                        event_end=event_end,
                        availability_policy=availability_policy,
                        availability_lag_days=availability_lag_days,
                    )
                except (TypeError, ValueError):
                    invalid_count += 1
                    continue
                if record is None:
                    filtered_count += 1
                    continue
                if record.admin1_code == "FR_UNKNOWN":
                    unresolved_admin1_counts[_none_if_blank(row.get("admin1")) or ""] += 1
                records.append(record)

    deduped: dict[str, EventTapeRecord] = {}
    duplicate_count = 0
    for record in sorted(
        records,
        key=lambda r: (r.source_available_at.astimezone(UTC), r.event_date, r.source_event_id),
    ):
        if record.source_event_id in deduped:
            duplicate_count += 1
            continue
        deduped[record.source_event_id] = record
    output_records = sorted(
        deduped.values(),
        key=lambda r: (r.source_available_at.astimezone(UTC), r.event_date, r.source_event_id),
    )
    if not output_records and not allow_empty:
        raise ValueError("ACLED event tape is empty after normalization")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
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
        "availability_policy": availability_policy,
        "availability_lag_days": availability_lag_days,
        "unresolved_admin1_counts": dict(sorted(unresolved_admin1_counts.items())),
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
    normalize = subparsers.add_parser("normalize-france-protests")
    normalize.add_argument("--raw", default="data/acled/raw/france_protest")
    normalize.add_argument("--out", default="data/acled/tape/france_protest/events.jsonl")
    normalize.add_argument("--event-start", default="2019-01-01")
    normalize.add_argument("--event-end", default="2026-01-04")
    normalize.add_argument(
        "--availability-policy",
        choices=["timestamp", "event_date_lag", "retrieved_at"],
        default="event_date_lag",
        help=(
            "How to set source_available_at. Use event_date_lag for retrospective "
            "benchmarks unless historical ACLED snapshots are available."
        ),
    )
    normalize.add_argument("--availability-lag-days", type=int, default=7)
    normalize.add_argument("--allow-empty", action="store_true")
    normalize.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "normalize-france-protests":
        try:
            write_acled_event_tape(
                raw_dir=Path(args.raw),
                out_path=Path(args.out),
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                availability_policy=args.availability_policy,
                availability_lag_days=args.availability_lag_days,
                allow_empty=args.allow_empty,
                allow_partial=args.allow_partial,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
