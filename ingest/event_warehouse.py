"""DuckDB-backed canonical warehouse for normalized event tape records."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.io_utils import open_text_auto, write_json_atomic
from ingest.paths import resolve_data_root, warehouse_path


UTC = dt.timezone.utc

EVENT_COLUMNS = (
    "source_name",
    "source_event_id",
    "event_date",
    "source_available_at",
    "retrieved_at",
    "country_code",
    "admin1_code",
    "location_name",
    "latitude",
    "longitude",
    "event_class",
    "event_code",
    "event_base_code",
    "event_root_code",
    "quad_class",
    "goldstein_scale",
    "num_mentions",
    "num_sources",
    "num_articles",
    "avg_tone",
    "actor1_name",
    "actor1_country_code",
    "actor2_name",
    "actor2_country_code",
    "source_url",
    "raw_json",
    "inserted_at",
)


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency install issue.
        raise RuntimeError("duckdb is required; install project dependencies first") from exc
    return duckdb


def _ensure_aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _connect(db_path: Path):
    return _duckdb().connect(str(db_path))


def init_warehouse(db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              source_name TEXT NOT NULL,
              source_event_id TEXT NOT NULL,
              event_date DATE NOT NULL,
              source_available_at TIMESTAMPTZ NOT NULL,
              retrieved_at TIMESTAMPTZ NOT NULL,
              country_code TEXT NOT NULL,
              admin1_code TEXT NOT NULL,
              location_name TEXT,
              latitude DOUBLE,
              longitude DOUBLE,
              event_class TEXT NOT NULL,
              event_code TEXT NOT NULL,
              event_base_code TEXT NOT NULL,
              event_root_code TEXT NOT NULL,
              quad_class INTEGER,
              goldstein_scale DOUBLE,
              num_mentions INTEGER,
              num_sources INTEGER,
              num_articles INTEGER,
              avg_tone DOUBLE,
              actor1_name TEXT,
              actor1_country_code TEXT,
              actor2_name TEXT,
              actor2_country_code TEXT,
              source_url TEXT,
              raw_json TEXT NOT NULL,
              inserted_at TIMESTAMPTZ NOT NULL,
              PRIMARY KEY (source_name, source_event_id)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_event_date ON events(event_date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_available ON events(source_available_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_events_admin1 ON events(admin1_code)")


def _record_row(record: EventTapeRecord, inserted_at: dt.datetime) -> tuple[Any, ...]:
    return (
        record.source_name,
        record.source_event_id,
        record.event_date,
        _ensure_aware(record.source_available_at),
        _ensure_aware(record.retrieved_at),
        record.country_code,
        record.admin1_code,
        record.location_name,
        record.latitude,
        record.longitude,
        record.event_class,
        record.event_code,
        record.event_base_code,
        record.event_root_code,
        record.quad_class,
        record.goldstein_scale,
        record.num_mentions,
        record.num_sources,
        record.num_articles,
        record.avg_tone,
        record.actor1_name,
        record.actor1_country_code,
        record.actor2_name,
        record.actor2_country_code,
        record.source_url,
        json.dumps(record.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        inserted_at,
    )


def upsert_records(
    *,
    db_path: Path,
    records: list[EventTapeRecord],
) -> dict[str, Any]:
    init_warehouse(db_path)
    inserted_at = dt.datetime.now(tz=UTC)
    placeholders = ", ".join(["?"] * len(EVENT_COLUMNS))
    update_columns = [column for column in EVENT_COLUMNS if column not in {"source_name", "source_event_id"}]
    set_clause = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    sql = (
        f"INSERT INTO events ({', '.join(EVENT_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT (source_name, source_event_id) DO UPDATE SET {set_clause}"
    )

    unique_keys = {(record.source_name, record.source_event_id) for record in records}
    with _connect(db_path) as con:
        before_count = int(con.execute("SELECT count(*) FROM events").fetchone()[0])
        if records:
            con.executemany(sql, [_record_row(record, inserted_at) for record in records])
        after_count = int(con.execute("SELECT count(*) FROM events").fetchone()[0])
    return {
        "input_count": len(records),
        "unique_key_count": len(unique_keys),
        "duplicate_input_count": len(records) - len(unique_keys),
        "inserted_count": after_count - before_count,
        "upserted_count": len(records),
        "total_event_count": after_count,
    }


def import_tape(
    *,
    db_path: Path,
    tape_path: Path,
    source_names: set[str] | None = None,
) -> dict[str, Any]:
    records = load_event_tape(tape_path)
    if source_names is not None:
        records = [record for record in records if record.source_name in source_names]
    result = upsert_records(db_path=db_path, records=records)
    return {
        **result,
        "input_path": str(tape_path),
        "source_names": sorted(source_names) if source_names is not None else None,
    }


def _source_name_clause(source_names: set[str], params: list[Any]) -> str:
    ordered = sorted(source_names)
    params.extend(ordered)
    return f"source_name IN ({', '.join(['?'] * len(ordered))})"


def query_records(
    *,
    db_path: Path,
    source_names: set[str] | None = None,
    event_start: dt.date | None = None,
    event_end: dt.date | None = None,
    available_before: dt.datetime | None = None,
    country_code: str | None = None,
    event_class: str | None = None,
) -> list[EventTapeRecord]:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"missing event warehouse: {db_path}; run python -m ingest.event_warehouse init and import data first"
        )
    where = ["1=1"]
    params: list[Any] = []
    if source_names:
        where.append(_source_name_clause(source_names, params))
    if event_start is not None:
        where.append("event_date >= ?")
        params.append(event_start)
    if event_end is not None:
        where.append("event_date <= ?")
        params.append(event_end)
    if available_before is not None:
        where.append("source_available_at < ?")
        params.append(_ensure_aware(available_before))
    if country_code is not None:
        where.append("country_code = ?")
        params.append(country_code)
    if event_class is not None:
        where.append("event_class = ?")
        params.append(event_class)

    sql = (
        f"SELECT {', '.join(EVENT_COLUMNS)} FROM events "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY source_available_at, event_date, source_name, source_event_id"
    )
    records: list[EventTapeRecord] = []
    with _connect(db_path) as con:
        rows = con.execute(sql, params).fetchall()
    for row in rows:
        payload = dict(zip(EVENT_COLUMNS, row))
        raw_json = payload.pop("raw_json")
        payload.pop("inserted_at", None)
        payload["raw"] = json.loads(raw_json)
        records.append(EventTapeRecord.model_validate(payload))
    return records


def source_counts(db_path: Path) -> dict[str, int]:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"missing event warehouse: {db_path}; run python -m ingest.event_warehouse init and import data first"
        )
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT source_name, count(*) FROM events GROUP BY source_name ORDER BY source_name"
        ).fetchall()
    return {str(source_name): int(count) for source_name, count in rows}


def warehouse_audit(db_path: Path) -> dict[str, Any]:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"missing event warehouse: {db_path}; run python -m ingest.event_warehouse init and import data first"
        )
    with _connect(db_path) as con:
        row = con.execute(
            """
            SELECT
              count(*) AS total_event_count,
              min(event_date) AS earliest_event_date,
              max(event_date) AS latest_event_date,
              min(source_available_at) AS earliest_source_available_at,
              max(source_available_at) AS latest_source_available_at,
              count(DISTINCT admin1_code) AS admin1_count
            FROM events
            """
        ).fetchone()
    return {
        "warehouse_path": str(db_path),
        "total_event_count": int(row[0] or 0),
        "source_counts": source_counts(db_path),
        "earliest_event_date": row[1].isoformat() if row[1] is not None else None,
        "latest_event_date": row[2].isoformat() if row[2] is not None else None,
        "earliest_source_available_at": row[3].isoformat() if row[3] is not None else None,
        "latest_source_available_at": row[4].isoformat() if row[4] is not None else None,
        "admin1_count": int(row[5] or 0),
        "database_bytes": Path(db_path).stat().st_size if Path(db_path).exists() else 0,
    }


def export_tape(
    *,
    db_path: Path,
    out_path: Path,
    source_names: set[str] | None = None,
    gzip_output: bool | None = None,
) -> dict[str, Any]:
    resolved_out = Path(out_path)
    if gzip_output is True and resolved_out.suffix != ".gz":
        resolved_out = resolved_out.with_name(f"{resolved_out.name}.gz")
    records = query_records(db_path=db_path, source_names=source_names)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(resolved_out, "w") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return {
        "output_path": str(resolved_out),
        "output_count": len(records),
        "source_names": sorted(source_names) if source_names is not None else None,
        "gzip_output": resolved_out.suffix == ".gz",
    }


def _parse_source_names(value: str | None) -> set[str] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names:
        raise ValueError("--source-names must be 'all' or a comma-separated list")
    return names


def _db_path_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "warehouse_path", None):
        return Path(args.warehouse_path).expanduser().resolve()
    return warehouse_path(resolve_data_root(getattr(args, "data_root", None)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--data-root", default=None)
    init.add_argument("--warehouse-path", default=None)

    import_cmd = subparsers.add_parser("import-tape")
    import_cmd.add_argument("--input", required=True)
    import_cmd.add_argument("--source-names", default=None)
    import_cmd.add_argument("--data-root", default=None)
    import_cmd.add_argument("--warehouse-path", default=None)

    export_cmd = subparsers.add_parser("export-tape")
    export_cmd.add_argument("--source-names", default="all")
    export_cmd.add_argument("--out", required=True)
    export_cmd.add_argument("--data-root", default=None)
    export_cmd.add_argument("--warehouse-path", default=None)
    export_cmd.add_argument("--gzip-output", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--data-root", default=None)
    audit.add_argument("--warehouse-path", default=None)
    audit.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        db_path = _db_path_from_args(args)
        if args.command == "init":
            init_warehouse(db_path)
            print(json.dumps({"warehouse_path": str(db_path)}, sort_keys=True))
            return 0
        if args.command == "import-tape":
            result = import_tape(
                db_path=db_path,
                tape_path=Path(args.input),
                source_names=_parse_source_names(args.source_names),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "export-tape":
            result = export_tape(
                db_path=db_path,
                out_path=Path(args.out),
                source_names=_parse_source_names(args.source_names),
                gzip_output=args.gzip_output or None,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "audit":
            audit = warehouse_audit(db_path)
            if args.out:
                write_json_atomic(Path(args.out), audit)
            print(json.dumps(audit, indent=2, sort_keys=True))
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
