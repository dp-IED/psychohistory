#!/usr/bin/env python3
"""Apply downstream allowlist/quarantine contract to a Cebu pilot DuckDB artifact.

Creates:
- `universal_event_staging_allowlist` table (filtered rows)
- `universal_event_staging_quarantine` table (hard quarantine rows)
- JSON summary with count checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def _read_allowlist_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            ids.append(token)
    # preserve order while deduplicating
    return list(dict.fromkeys(ids))


def _read_quarantine_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        row = json.loads(token)
        event_id = str(row.get("event_id") or "").strip()
        if event_id:
            ids.append(event_id)
    return list(dict.fromkeys(ids))


def apply_allowlist(*, db_path: Path, proceed_ids_path: Path, quarantine_jsonl_path: Path, out_json: Path) -> dict[str, int | str]:
    proceed_ids = _read_allowlist_ids(proceed_ids_path)
    quarantine_ids = _read_quarantine_ids(quarantine_jsonl_path)

    if not proceed_ids:
        raise ValueError(f"no proceed IDs found in {proceed_ids_path}")

    con = duckdb.connect(str(db_path))
    try:
        total = con.execute("SELECT COUNT(*) FROM universal_event_staging").fetchone()[0]

        con.execute("DROP TABLE IF EXISTS universal_event_staging_allowlist")
        con.execute(
            """
            CREATE TABLE universal_event_staging_allowlist AS
            SELECT *
            FROM universal_event_staging
            WHERE event_id IN (SELECT unnest(?))
            """,
            [proceed_ids],
        )

        con.execute("DROP TABLE IF EXISTS universal_event_staging_quarantine")
        con.execute(
            """
            CREATE TABLE universal_event_staging_quarantine AS
            SELECT *
            FROM universal_event_staging
            WHERE event_id IN (SELECT unnest(?))
            """,
            [quarantine_ids],
        )

        proceed_count = con.execute("SELECT COUNT(*) FROM universal_event_staging_allowlist").fetchone()[0]
        quarantine_count = con.execute("SELECT COUNT(*) FROM universal_event_staging_quarantine").fetchone()[0]

        payload: dict[str, int | str] = {
            "db_path": str(db_path),
            "total_rows": int(total),
            "proceed_ids": int(len(proceed_ids)),
            "quarantine_ids": int(len(quarantine_ids)),
            "allowlist_rows": int(proceed_count),
            "quarantine_rows": int(quarantine_count),
            "excluded_rows": int(total - proceed_count),
        }

        if proceed_count != len(proceed_ids):
            payload["warning"] = "allowlist IDs do not all match DB rows"

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    finally:
        con.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--proceed-ids", required=True)
    parser.add_argument("--quarantine-jsonl", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = apply_allowlist(
        db_path=Path(args.db_path),
        proceed_ids_path=Path(args.proceed_ids),
        quarantine_jsonl_path=Path(args.quarantine_jsonl),
        out_json=Path(args.out),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
