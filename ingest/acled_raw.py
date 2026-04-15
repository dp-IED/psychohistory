"""Raw ingestion for ACLED API v1 — France protest events.

Fetches from the ACLED API export endpoint (https://api.acleddata.com/acled/read)
using a page-based cursor, applies country/event-type filters, and writes
newline-delimited JSON fragments mirroring the layout used by gdelt_raw.py:

    data/acled/raw/france_protest/
        fetch_metadata.json
        fetch_manifest.jsonl
        fragments/YYYY/MM/DD/<run_timestamp>.jsonl

Each fragment line is one raw ACLED event row as returned by the API, with four
internal metadata fields injected by the fetcher:

    _retrieved_at           ISO-8601 UTC timestamp of the HTTP response
    _run_id                 run identifier shared across all fragments in one run
    _page                   1-based page number this row was fetched on
    _source_name            "acled_api_v1"

ACLED API docs: https://acleddata.com/acled-api-documentation
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


ACLED_API_BASE = "https://api.acleddata.com/acled/read"
SOURCE_NAME = "acled_api_v1"
DOMAIN = "france_protest"

"""
ACLED event_type filter for protests.
See https://acleddata.com/acled-api-documentation — event_type field.
"""
FRANCE_PROTEST_FILTERS: dict[str, str] = {
    "country": "France",
    "event_type": "Protests",
}

UTC = dt.timezone.utc

"""
ACLED returns dates as 'DD Month YYYY' strings, e.g. '05 January 2021'.
"""
_ACLED_DATE_FORMAT = "%d %B %Y"


@dataclass(frozen=True)
class AcledPageResult:
    page: int
    rows: list[dict[str, Any]]
    total_count: int
    retrieved_at: dt.datetime


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def format_datetime_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_acled_date(value: str) -> dt.date:
    """Parse ACLED's 'DD Month YYYY' or ISO 'YYYY-MM-DD' date strings."""
    value = value.strip()
    if len(value) == 10 and value[4] == "-":
        return dt.date.fromisoformat(value)
    return dt.datetime.strptime(value, _ACLED_DATE_FORMAT).date()


def _build_request_url(
    *,
    api_key: str,
    email: str,
    event_start: dt.date,
    event_end: dt.date,
    page: int,
    page_size: int,
    base_url: str,
) -> str:
    params: dict[str, str] = {
        "key": api_key,
        "email": email,
        "country": FRANCE_PROTEST_FILTERS["country"],
        "event_type": FRANCE_PROTEST_FILTERS["event_type"],
        "event_date": f"{event_start.isoformat()}|{event_end.isoformat()}",
        "event_date_where": "BETWEEN",
        "page": str(page),
        "limit": str(page_size),
        "export_type": "json",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _fetch_page(
    url: str,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def _extract_rows(api_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the data array out of an ACLED API response envelope."""
    data = api_response.get("data")
    if data is None:
        raise ValueError(f"ACLED response missing 'data' key; keys: {list(api_response)}")
    if not isinstance(data, list):
        raise ValueError(f"ACLED response 'data' is not a list: {type(data)}")
    return data


def _extract_total_count(api_response: dict[str, Any]) -> int:
    count = api_response.get("count")
    if count is None:
        return 0
    return int(count)


def iter_acled_pages(
    *,
    api_key: str,
    email: str,
    event_start: dt.date,
    event_end: dt.date,
    page_size: int = 500,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    base_url: str = ACLED_API_BASE,
) -> Iterator[AcledPageResult]:
    """Yield one AcledPageResult per API page until all rows are exhausted."""
    page = 1
    total_seen = 0

    while True:
        url = _build_request_url(
            api_key=api_key,
            email=email,
            event_start=event_start,
            event_end=event_end,
            page=page,
            page_size=page_size,
            base_url=base_url,
        )
        retrieved_at = utc_now()
        response = _fetch_page(url, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds)
        rows = _extract_rows(response)
        total_count = _extract_total_count(response)

        if not rows:
            break

        yield AcledPageResult(
            page=page,
            rows=rows,
            total_count=total_count,
            retrieved_at=retrieved_at,
        )

        total_seen += len(rows)
        if total_seen >= total_count:
            break
        page += 1


def _fragment_path_for_run(run_id: str, retrieved_at: dt.datetime) -> Path:
    ts = retrieved_at.astimezone(UTC)
    return (
        Path("fragments")
        / f"{ts:%Y}"
        / f"{ts:%m}"
        / f"{ts:%d}"
        / f"{run_id}.jsonl"
    )


def _json_dump_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def fetch_france_protests(
    *,
    api_key: str,
    email: str,
    event_start: dt.date,
    event_end: dt.date,
    out_dir: Path,
    page_size: int = 500,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    force: bool = False,
    progress: bool = False,
    base_url: str = ACLED_API_BASE,
) -> dict[str, Any]:
    """
    Fetch ACLED France protest events and write raw fragments + manifest.

    Data is written to a new subdirectory isolated from any existing GDELT data.
    Existing fragments are never modified; each run appends to fetch_manifest.jsonl
    and writes a fresh fragment file named by run_id.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "fetch_manifest.jsonl"
    if manifest_path.exists() and not force:
        existing_lines = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed = [row for row in existing_lines if row.get("status") == "ok"]
        if completed:
            last = completed[-1]
            if (
                last.get("event_start") == event_start.isoformat()
                and last.get("event_end") == event_end.isoformat()
            ):
                if progress:
                    print(
                        f"[acled] skipping: completed run already found in {manifest_path}; "
                        "use --force to refetch",
                        file=sys.stderr,
                        flush=True,
                    )
                return json.loads((out_dir / "fetch_metadata.json").read_text(encoding="utf-8"))

    run_id = f"{utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    run_start = utc_now()

    total_rows = 0
    pages_fetched = 0
    fragment_path: Path | None = None
    status = "failed"
    error_text: str | None = None

    try:
        all_rows: list[dict[str, Any]] = []
        first_retrieved_at = run_start

        for result in iter_acled_pages(
            api_key=api_key,
            email=email,
            event_start=event_start,
            event_end=event_end,
            page_size=page_size,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            base_url=base_url,
        ):
            if pages_fetched == 0:
                first_retrieved_at = result.retrieved_at

            for row in result.rows:
                row["_retrieved_at"] = format_datetime_z(result.retrieved_at)
                row["_run_id"] = run_id
                row["_page"] = result.page
                row["_source_name"] = SOURCE_NAME
                all_rows.append(row)

            pages_fetched += 1
            total_rows += len(result.rows)

            if progress:
                print(
                    f"[acled] page={result.page} rows_this_page={len(result.rows)} "
                    f"total_so_far={total_rows} total_api_count={result.total_count}",
                    file=sys.stderr,
                    flush=True,
                )

        if all_rows:
            relative_fragment = _fragment_path_for_run(run_id, first_retrieved_at)
            absolute_fragment = out_dir / relative_fragment
            absolute_fragment.parent.mkdir(parents=True, exist_ok=True)
            temp_path = absolute_fragment.with_suffix(".jsonl.tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                for row in all_rows:
                    handle.write(_json_dump_line(row))
            temp_path.replace(absolute_fragment)
            fragment_path = relative_fragment

        status = "ok"

    except Exception as exc:
        error_text = str(exc)

    manifest_row: dict[str, Any] = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "filters": FRANCE_PROTEST_FILTERS,
        "pages_fetched": pages_fetched,
        "total_rows": total_rows,
        "fragment_path": str(fragment_path) if fragment_path else None,
        "status": status,
        "error": error_text,
        "retrieved_at": format_datetime_z(run_start),
    }

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dump_line(manifest_row))

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "filters": FRANCE_PROTEST_FILTERS,
        "pages_fetched": pages_fetched,
        "total_rows": total_rows,
        "fragment_path": str(fragment_path) if fragment_path else None,
        "status": status,
        "retrieved_at": format_datetime_z(run_start),
    }
    (out_dir / "fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if status != "ok":
        raise RuntimeError(f"ACLED fetch failed: {error_text}")

    return metadata


def normalize_acled_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Validate and lightly normalise a raw ACLED row.

    Returns None if the row should be dropped (wrong country, wrong event type,
    or unparseable date). Does not raise — callers should count None returns as
    filtered rows rather than errors.
    """
    if str(row.get("country", "")).strip().lower() != "france":
        return None
    if str(row.get("event_type", "")).strip().lower() != "protests":
        return None
    event_date_raw = str(row.get("event_date") or "").strip()
    if not event_date_raw:
        return None
    try:
        parse_acled_date(event_date_raw)
    except (ValueError, KeyError):
        return None
    return row


def load_acled_fragments(raw_dir: Path) -> list[dict[str, Any]]:
    """
    Load all rows from ACLED raw fragments under raw_dir, using the manifest
    to restrict to the most recent completed run.

    Mirrors the manifest-aware loading pattern in gdelt_raw / event_tape.
    """
    manifest_path = raw_dir / "fetch_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing ACLED fetch manifest: {manifest_path}")

    metadata_path = raw_dir / "fetch_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing ACLED fetch metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = metadata.get("run_id")
    if not run_id:
        raise ValueError("ACLED fetch metadata is missing run_id; rerun the fetcher")

    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        manifest_row = json.loads(line)
        if manifest_row.get("run_id") != run_id:
            continue
        if manifest_row.get("status") != "ok":
            continue
        fragment = manifest_row.get("fragment_path")
        if not fragment:
            continue
        fragment_path = Path(fragment)
        if not fragment_path.is_absolute():
            fragment_path = raw_dir / fragment_path
        if not fragment_path.exists():
            raise FileNotFoundError(f"manifest references missing fragment: {fragment_path}")
        with fragment_path.open("r", encoding="utf-8") as handle:
            for row_line in handle:
                if row_line.strip():
                    rows.append(json.loads(row_line))

    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-france-protests")
    fetch.add_argument("--api-key", required=True, help="ACLED API key")
    fetch.add_argument("--email", required=True, help="Email registered with ACLED")
    fetch.add_argument("--event-start", default="2019-01-01")
    fetch.add_argument("--event-end", default="2026-01-04")
    fetch.add_argument("--out", default="data/acled/raw/france_protest")
    fetch.add_argument("--page-size", type=int, default=500)
    fetch.add_argument("--max-retries", type=int, default=3)
    fetch.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    fetch.add_argument("--force", action="store_true", help="Re-fetch even if a completed run exists")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-france-protests":
        try:
            fetch_france_protests(
                api_key=args.api_key,
                email=args.email,
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                out_dir=Path(args.out),
                page_size=args.page_size,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                force=args.force,
                progress=True,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
