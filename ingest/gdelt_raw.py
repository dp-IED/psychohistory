"""Raw-file ingestion for GDELT 2.0 France protest events."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from ingest.io_utils import open_text_auto, write_json_atomic


GDELT_V2_EVENT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
SOURCE_NAME = "gdelt_v2_events"
DOMAIN = "france_protest"
FRANCE_PROTEST_FILTERS = {"ActionGeo_CountryCode": "FR", "EventRootCode": "14"}
ARAB_SPRING_CONTEXT = "arab_spring"
ARAB_SPRING_COUNTRY_CODES = frozenset({"EG", "TU", "LY", "SY"})
GDELT10_EVENTS_BASE_URL = "http://data.gdeltproject.org/events"
MIN_DISK_FREE_BYTES_WARN = 20 * 1024 * 1024 * 1024
UTC = dt.timezone.utc
RawRetention = Literal["none", "compressed", "full"]


@dataclass(frozen=True)
class MasterfileEntry:
    expected_size: int
    expected_md5: str
    url: str
    source_file_timestamp: dt.datetime


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def format_datetime_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_datetime_utc(value: str) -> dt.datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    if len(raw) == 14 and raw.isdigit():
        return dt.datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_sql_date(value: str) -> dt.date:
    return dt.datetime.strptime(value.strip(), "%Y%m%d").date()


def parse_source_timestamp_from_url(url: str) -> dt.datetime:
    name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    if not name.endswith(".export.CSV.zip"):
        raise ValueError(f"not a GDELT export zip URL: {url}")
    timestamp_text = name[:14]
    return dt.datetime.strptime(timestamp_text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def parse_masterfilelist(text: str) -> list[MasterfileEntry]:
    entries: list[MasterfileEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) != 3:
            continue
        size_text, md5_hex, url = parts
        if not url.endswith(".export.CSV.zip"):
            continue
        entries.append(
            MasterfileEntry(
                expected_size=int(size_text),
                expected_md5=md5_hex,
                url=url,
                source_file_timestamp=parse_source_timestamp_from_url(url),
            )
        )
    return sorted(entries, key=lambda entry: (entry.source_file_timestamp, entry.url))


def select_export_entries(
    entries: Sequence[MasterfileEntry],
    *,
    source_start: dt.datetime,
    source_end: dt.datetime,
) -> list[MasterfileEntry]:
    start = source_start.astimezone(UTC)
    end = source_end.astimezone(UTC)
    return [
        entry
        for entry in entries
        if start <= entry.source_file_timestamp.astimezone(UTC) <= end
    ]


def map_gdelt_event_row(fields: Sequence[str]) -> dict[str, str]:
    if len(fields) != len(GDELT_V2_EVENT_COLUMNS):
        raise ValueError(
            f"expected {len(GDELT_V2_EVENT_COLUMNS)} GDELT columns, got {len(fields)}"
        )
    return dict(zip(GDELT_V2_EVENT_COLUMNS, fields))


def row_matches_france_protest(
    row: dict[str, str],
    *,
    event_start: dt.date,
    event_end: dt.date,
) -> bool:
    if row.get("ActionGeo_CountryCode") != "FR":
        return False
    if row.get("EventRootCode") != "14":
        return False
    try:
        event_date = parse_sql_date(row.get("SQLDATE", ""))
    except ValueError:
        return False
    return event_start <= event_date <= event_end


def row_matches_arab_spring(
    row: dict[str, str],
    *,
    event_start: dt.date,
    event_end: dt.date,
) -> bool:
    cc = (row.get("ActionGeo_CountryCode") or "").strip()
    if cc not in ARAB_SPRING_COUNTRY_CODES:
        return False
    try:
        event_date = parse_sql_date(row.get("SQLDATE", ""))
    except ValueError:
        return False
    return event_start <= event_date <= event_end


def iter_gdelt10_daily_export_urls(
    *,
    event_start: dt.date,
    event_end: dt.date,
    events_base_url: str = GDELT10_EVENTS_BASE_URL,
) -> Iterator[tuple[dt.date, str]]:
    """Yield (calendar day, zip URL) for GDELT 1.0 daily event files (separate from gdeltv2 masterfile)."""

    day = event_start
    while day <= event_end:
        yield day, f"{events_base_url.rstrip('/')}/{day:%Y%m%d}.export.CSV.zip"
        day += dt.timedelta(days=1)


def _arab_spring_fragment_filename(day: dt.date) -> str:
    return f"arab_spring_{day:%Y%m%d}_000000.jsonl"


def _disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _warn_disk_low(path: Path, *, label: str) -> None:
    free = _disk_free_bytes(path)
    if free < MIN_DISK_FREE_BYTES_WARN:
        print(
            f"[gdelt arab_spring] warning: free disk space {free / (1024**3):.1f} GB "
            f"below 20 GB ({label})",
            file=sys.stderr,
            flush=True,
        )


def _arab_spring_manifest_initial() -> dict[str, Any]:
    return {
        "context": ARAB_SPRING_CONTEXT,
        "countries": sorted(ARAB_SPRING_COUNTRY_CODES),
        "date_start": "2010-01-01",
        "date_end": "2013-12-31",
        "files_fetched": 0,
        "rows_written": 0,
        "fetch_completed_at": None,
        "gdelt_version": "1.0",
    }


def fetch_arab_spring(
    *,
    out_dir: Path,
    event_start: dt.date,
    event_end: dt.date,
    events_base_url: str = GDELT10_EVENTS_BASE_URL,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    force: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """
    Fetch GDELT 1.0 daily ``.export.CSV.zip`` files (historical URL pattern, not mixed with gdeltv2 masterfile).
    Zip bytes are released immediately after parsing; only JSONL fragments remain on disk.
    """

    if event_start > event_end:
        raise ValueError("event_start must be on or before event_end")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "fetch_manifest.json"
    if manifest_path.exists():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _arab_spring_manifest_initial()
        manifest["date_start"] = event_start.isoformat()
        manifest["date_end"] = event_end.isoformat()
        write_json_atomic(manifest_path, manifest)

    files_fetched = int(manifest.get("files_fetched") or 0)
    rows_written = int(manifest.get("rows_written") or 0)

    for day, url in iter_gdelt10_daily_export_urls(
        event_start=event_start,
        event_end=event_end,
        events_base_url=events_base_url,
    ):
        fragment_name = _arab_spring_fragment_filename(day)
        fragment_path = out_dir / fragment_name
        if fragment_path.exists() and not force:
            if progress:
                print(f"[gdelt arab_spring] skip existing {fragment_name}", file=sys.stderr, flush=True)
            continue

        _warn_disk_low(out_dir, label=f"before {day.isoformat()}")
        retrieved_at = format_datetime_z(utc_now())
        metadata = {
            "_source_file_url": url,
            "_source_file_timestamp": f"{day:%Y-%m-%d}T00:00:00Z",
            "_retrieved_at": retrieved_at,
        }
        zip_bytes: bytes | None = None
        kept_count = 0
        try:
            try:
                zip_bytes = _download_url(
                    url,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    if progress:
                        print(f"[gdelt arab_spring] missing file (404): {url}", file=sys.stderr, flush=True)
                    continue
                raise
            parsed_rows = parse_gdelt_zip_bytes(zip_bytes, metadata=metadata)
            zip_bytes = None
            kept = [
                row
                for row in parsed_rows
                if row_matches_arab_spring(row, event_start=event_start, event_end=event_end)
            ]
            kept_count = len(kept)
            if kept:
                fragment_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = fragment_path.with_name(f".{fragment_path.name}.tmp")
                with temp_path.open("w", encoding="utf-8") as handle:
                    for row in kept:
                        handle.write(_json_dump_line(row))
                temp_path.replace(fragment_path)
            files_fetched += 1
            rows_written += kept_count
            manifest["files_fetched"] = files_fetched
            manifest["rows_written"] = rows_written
            manifest["fetch_completed_at"] = None
            write_json_atomic(manifest_path, manifest)
        finally:
            zip_bytes = None

        _warn_disk_low(out_dir, label=f"after {day.isoformat()}")
        if progress:
            print(
                f"[gdelt arab_spring] {day.isoformat()} rows={kept_count}",
                file=sys.stderr,
                flush=True,
            )

    manifest["fetch_completed_at"] = format_datetime_z(utc_now())
    write_json_atomic(manifest_path, manifest)
    return {
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "files_fetched": files_fetched,
        "rows_written": rows_written,
    }


def parse_gdelt_zip_bytes(
    zip_bytes: bytes,
    *,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    attached_metadata = metadata or {}
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("GDELT zip contains no CSV member")
        with archive.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text, delimiter="\t")
            for fields in reader:
                row = map_gdelt_event_row(fields)
                row.update(attached_metadata)
                rows.append(row)
    return rows


def _read_text_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(urllib.request.url2pathname(parsed.path if parsed.scheme else url))
        return path.read_text(encoding="utf-8")
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _download_url(url: str, *, max_retries: int, retry_backoff_seconds: float) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(urllib.request.url2pathname(parsed.path if parsed.scheme else url))
        return path.read_bytes()

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised by integration use.
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def _fragment_relative_path(entry: MasterfileEntry, *, raw_retention: RawRetention) -> Path:
    timestamp = entry.source_file_timestamp.astimezone(UTC)
    suffix = ".jsonl.gz" if raw_retention == "compressed" else ".jsonl"
    return (
        Path("fragments")
        / f"{timestamp:%Y}"
        / f"{timestamp:%m}"
        / f"{timestamp:%d}"
        / f"{timestamp:%Y%m%d%H%M%S}{suffix}"
    )


def _json_dump_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _fetch_config(*, event_start: dt.date, event_end: dt.date) -> dict[str, Any]:
    return {
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "filters": FRANCE_PROTEST_FILTERS,
    }


def _load_completed_manifest(
    out_dir: Path,
    *,
    fetch_config: dict[str, Any],
    raw_retention: RawRetention,
) -> dict[str, dict[str, Any]]:
    manifest_path = out_dir / "fetch_manifest.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return completed
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") not in {"ok", "skipped"}:
            continue
        if row.get("fetch_config") != fetch_config:
            continue
        previous_raw_retention = row.get("raw_retention") or "full"
        if previous_raw_retention != raw_retention:
            continue
        fragment = row.get("fragment_path")
        fragment_exists = False
        if fragment:
            fragment_path = Path(fragment)
            if not fragment_path.is_absolute():
                fragment_path = out_dir / fragment_path
            fragment_exists = fragment_path.exists()
        elif row.get("kept_row_count") == 0:
            fragment_exists = True
        if fragment_exists:
            completed[str(row.get("url"))] = row
    return completed


def _skipped_manifest_row(
    entry: MasterfileEntry,
    previous: dict[str, Any],
    *,
    run_id: str,
    fetch_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "fetch_config": fetch_config,
        "raw_retention": previous.get("raw_retention") or "full",
        "source_file_timestamp": format_datetime_z(entry.source_file_timestamp),
        "url": entry.url,
        "expected_size": entry.expected_size,
        "expected_md5": entry.expected_md5,
        "status": "skipped",
        "downloaded_size": previous.get("downloaded_size"),
        "downloaded_md5": previous.get("downloaded_md5"),
        "raw_row_count": int(previous.get("raw_row_count") or 0),
        "kept_row_count": int(previous.get("kept_row_count") or 0),
        "fragment_path": previous.get("fragment_path"),
        "error": None,
        "retrieved_at": format_datetime_z(utc_now()),
    }


def _process_entry(
    entry: MasterfileEntry,
    *,
    out_dir: Path,
    event_start: dt.date,
    event_end: dt.date,
    run_id: str,
    fetch_config: dict[str, Any],
    max_retries: int,
    retry_backoff_seconds: float,
    raw_retention: RawRetention,
) -> dict[str, Any]:
    retrieved_at = format_datetime_z(utc_now())
    relative_path = _fragment_relative_path(entry, raw_retention=raw_retention)
    absolute_path = out_dir / relative_path
    if absolute_path.exists():
        absolute_path.unlink()
    base_manifest = {
        "run_id": run_id,
        "fetch_config": fetch_config,
        "raw_retention": raw_retention,
        "source_file_timestamp": format_datetime_z(entry.source_file_timestamp),
        "url": entry.url,
        "expected_size": entry.expected_size,
        "expected_md5": entry.expected_md5,
        "status": "failed",
        "downloaded_size": None,
        "downloaded_md5": None,
        "raw_row_count": 0,
        "kept_row_count": 0,
        "fragment_path": None,
        "error": None,
        "retrieved_at": retrieved_at,
    }
    try:
        zip_bytes = _download_url(
            entry.url,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        downloaded_size = len(zip_bytes)
        downloaded_md5 = hashlib.md5(zip_bytes).hexdigest()
        base_manifest["downloaded_size"] = downloaded_size
        base_manifest["downloaded_md5"] = downloaded_md5
        if downloaded_size != entry.expected_size:
            raise ValueError(
                f"size mismatch: expected {entry.expected_size}, got {downloaded_size}"
            )
        if downloaded_md5.lower() != entry.expected_md5.lower():
            raise ValueError(
                f"md5 mismatch: expected {entry.expected_md5}, got {downloaded_md5}"
            )

        metadata = {
            "_source_file_url": entry.url,
            "_source_file_timestamp": format_datetime_z(entry.source_file_timestamp),
            "_source_file_size": entry.expected_size,
            "_source_file_md5": entry.expected_md5,
            "_retrieved_at": retrieved_at,
        }
        rows = parse_gdelt_zip_bytes(zip_bytes, metadata=metadata)
        kept_rows = [
            row
            for row in rows
            if row_matches_france_protest(row, event_start=event_start, event_end=event_end)
        ]
        fragment_path: str | None = None
        if kept_rows and raw_retention != "none":
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = absolute_path.with_name(f".{absolute_path.name}.tmp")
            if raw_retention == "compressed":
                temp_path = temp_path.with_suffix(f"{temp_path.suffix}.gz")
            with open_text_auto(temp_path, "w") as handle:
                for row in kept_rows:
                    handle.write(_json_dump_line(row))
            temp_path.replace(absolute_path)
            fragment_path = relative_path.as_posix()

        base_manifest.update(
            {
                "status": "ok",
                "raw_row_count": len(rows),
                "kept_row_count": len(kept_rows),
                "fragment_path": fragment_path,
                "error": None,
            }
        )
    except Exception as exc:
        base_manifest["error"] = str(exc)
    return base_manifest


def _write_manifest_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(_json_dump_line(row))
    handle.flush()


def _process_entries_bounded(
    entries: Sequence[MasterfileEntry],
    *,
    out_dir: Path,
    event_start: dt.date,
    event_end: dt.date,
    run_id: str,
    fetch_config: dict[str, Any],
    workers: int,
    max_retries: int,
    retry_backoff_seconds: float,
    raw_retention: RawRetention,
) -> Iterator[dict[str, Any]]:
    max_workers = max(1, workers)
    entry_iter = iter(entries)

    def submit_next(
        executor: concurrent.futures.ThreadPoolExecutor,
        pending: dict[concurrent.futures.Future[dict[str, Any]], MasterfileEntry],
    ) -> bool:
        try:
            entry = next(entry_iter)
        except StopIteration:
            return False
        future = executor.submit(
            _process_entry,
            entry,
            out_dir=out_dir,
            event_start=event_start,
            event_end=event_end,
            run_id=run_id,
            fetch_config=fetch_config,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            raw_retention=raw_retention,
        )
        pending[future] = entry
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: dict[concurrent.futures.Future[dict[str, Any]], MasterfileEntry] = {}
        for _ in range(max_workers * 2):
            if not submit_next(executor, pending):
                break
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                pending.pop(future)
                yield future.result()
                submit_next(executor, pending)


def fetch_france_protests(
    *,
    masterfilelist_url: str,
    event_start: dt.date,
    event_end: dt.date,
    source_start: dt.datetime,
    source_end: dt.datetime | str,
    out_dir: Path,
    workers: int,
    force: bool = False,
    allow_partial: bool = False,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    raw_retention: RawRetention = "none",
    progress: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw_retention not in {"none", "compressed", "full"}:
        raise ValueError(f"unknown raw retention: {raw_retention}")
    masterfile_text = _read_text_url(masterfilelist_url)
    entries = parse_masterfilelist(masterfile_text)
    if not entries:
        raise ValueError("masterfilelist contains no .export.CSV.zip entries")
    resolved_source_end = (
        max(entry.source_file_timestamp for entry in entries)
        if source_end == "latest"
        else source_end
    )
    if not isinstance(resolved_source_end, dt.datetime):
        raise TypeError("source_end must be a datetime or 'latest'")
    selected = select_export_entries(
        entries,
        source_start=source_start,
        source_end=resolved_source_end,
    )
    run_id = f"{utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    fetch_config = _fetch_config(event_start=event_start, event_end=event_end)
    completed = (
        {}
        if force
        else _load_completed_manifest(
            out_dir,
            fetch_config=fetch_config,
            raw_retention=raw_retention,
        )
    )

    to_fetch: list[MasterfileEntry] = []
    completed_count = 0
    failed_count = 0
    kept_row_count = 0
    processed_count = 0
    manifest_path = out_dir / "fetch_manifest.jsonl"

    def maybe_print_progress(entry: MasterfileEntry | None = None) -> None:
        if not progress:
            return
        if processed_count % 1000 != 0 and processed_count != len(selected):
            return
        timestamp = (
            format_datetime_z(entry.source_file_timestamp)
            if entry is not None
            else "n/a"
        )
        print(
            (
                f"[fetch] {processed_count}/{len(selected)} "
                f"completed={completed_count} failed={failed_count} "
                f"kept_rows={kept_row_count} last_source={timestamp}"
            ),
            file=sys.stderr,
            flush=True,
        )

    with manifest_path.open("a", encoding="utf-8") as handle:
        for entry in selected:
            previous = completed.get(entry.url)
            if previous is not None:
                row = _skipped_manifest_row(
                    entry,
                    previous,
                    run_id=run_id,
                    fetch_config=fetch_config,
                )
                _write_manifest_row(handle, row)
                completed_count += 1
                kept_row_count += int(row.get("kept_row_count") or 0)
                processed_count += 1
                maybe_print_progress(entry)
            else:
                to_fetch.append(entry)

        for row in _process_entries_bounded(
            to_fetch,
            out_dir=out_dir,
            event_start=event_start,
            event_end=event_end,
            run_id=run_id,
            fetch_config=fetch_config,
            workers=workers,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            raw_retention=raw_retention,
        ):
            _write_manifest_row(handle, row)
            if row["status"] in {"ok", "skipped"}:
                completed_count += 1
            elif row["status"] == "failed":
                failed_count += 1
            kept_row_count += int(row.get("kept_row_count") or 0)
            processed_count += 1
            maybe_print_progress(
                MasterfileEntry(
                    expected_size=int(row.get("expected_size") or 0),
                    expected_md5=str(row.get("expected_md5") or ""),
                    url=str(row.get("url") or ""),
                    source_file_timestamp=parse_datetime_utc(
                        str(row.get("source_file_timestamp"))
                    ),
                )
            )

    retrieved_at = format_datetime_z(utc_now())
    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "masterfilelist_url": masterfilelist_url,
        "source_start": format_datetime_z(source_start),
        "source_end": format_datetime_z(resolved_source_end),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "filters": FRANCE_PROTEST_FILTERS,
        "fetch_config": fetch_config,
        "retrieved_at": retrieved_at,
        "selected_file_count": len(selected),
        "completed_file_count": completed_count,
        "failed_file_count": failed_count,
        "allow_partial": allow_partial,
        "raw_retention": raw_retention,
        "kept_row_count": kept_row_count,
    }
    (out_dir / "fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed_count and not allow_partial:
        raise RuntimeError(f"{failed_count} GDELT files failed; rerun with --allow-partial to ignore")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-france-protests")
    fetch.add_argument("--event-start", default="2019-01-01")
    fetch.add_argument("--event-end", default="2026-01-04")
    fetch.add_argument("--source-start", default="2019-01-01T00:00:00Z")
    fetch.add_argument("--source-end", default="latest")
    fetch.add_argument("--out", default="data/gdelt/raw/france_protest")
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--masterfilelist-url", default=MASTERFILELIST_URL)
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--allow-partial", action="store_true")
    fetch.add_argument("--max-retries", type=int, default=3)
    fetch.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    fetch.add_argument(
        "--raw-retention",
        choices=["none", "compressed", "full"],
        default="none",
    )

    arab = subparsers.add_parser("fetch-arab-spring")
    arab.add_argument("--raw-dir", default="data/gdelt/raw/arab_spring")
    arab.add_argument(
        "--event-start",
        "--date-start",
        default="2010-01-01",
        help="Inclusive event-date window start (ISO date). Alias: --date-start.",
    )
    arab.add_argument(
        "--event-end",
        "--date-end",
        default="2013-12-31",
        help="Inclusive event-date window end (ISO date). Alias: --date-end.",
    )
    arab.add_argument("--events-base-url", default=GDELT10_EVENTS_BASE_URL)
    arab.add_argument("--max-retries", type=int, default=3)
    arab.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    arab.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-france-protests":
        source_end: dt.datetime | str
        if args.source_end == "latest":
            source_end = "latest"
        else:
            source_end = parse_datetime_utc(args.source_end)
        try:
            fetch_france_protests(
                masterfilelist_url=args.masterfilelist_url,
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                source_start=parse_datetime_utc(args.source_start),
                source_end=source_end,
                out_dir=Path(args.out),
                workers=args.workers,
                force=args.force,
                allow_partial=args.allow_partial,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                raw_retention=args.raw_retention,
                progress=True,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "fetch-arab-spring":
        try:
            fetch_arab_spring(
                out_dir=Path(args.raw_dir),
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                events_base_url=args.events_base_url,
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
