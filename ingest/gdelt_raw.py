"""Raw-file ingestion for GDELT 2.0 France protest events."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import shutil
import sys
import threading
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


logger = logging.getLogger(__name__)

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

# GDELT 1.0 daily /events/ exports omit the three *_Geo_Type columns that GDELT 2.0 added.
_GDELT_1_0_OMIT_FROM_V2 = ("Actor1Geo_Type", "Actor2Geo_Type", "ActionGeo_Type")
GDELT_V1_EVENT_COLUMNS = [c for c in GDELT_V2_EVENT_COLUMNS if c not in _GDELT_1_0_OMIT_FROM_V2]
assert len(GDELT_V1_EVENT_COLUMNS) == 58, "GDELT 1.0 event export is 58 columns; update column lists"
assert len(GDELT_V2_EVENT_COLUMNS) == 61, "expected 61 V2 event columns"

MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
SOURCE_NAME = "gdelt_v2_events"
ARAB_SPRING_GDELT_SOURCE_NAME = "gdelt_v1_events"
DOMAIN = "france_protest"
FRANCE_PROTEST_FILTERS = {"ActionGeo_CountryCode": "FR", "EventRootCode": "14"}
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


def map_gdelt_event_row(
    fields: Sequence[str],
    *,
    column_names: Sequence[str] | None = None,
) -> dict[str, str]:
    names = list(column_names) if column_names is not None else list(GDELT_V2_EVENT_COLUMNS)
    if len(fields) != len(names):
        raise ValueError(
            f"expected {len(names)} GDELT columns, got {len(fields)}"
        )
    return dict(zip(names, fields, strict=True))


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


ARAB_SPRING_COUNTRY_CODES = frozenset({"EG", "TU", "LY", "SY"})


def row_matches_arab_spring(
    row: dict[str, str],
    *,
    event_start: dt.date,
    event_end: dt.date,
) -> bool:
    if row.get("ActionGeo_CountryCode") not in ARAB_SPRING_COUNTRY_CODES:
        return False
    try:
        event_date = parse_sql_date(row.get("SQLDATE", ""))
    except ValueError:
        return False
    return event_start <= event_date <= event_end


def iter_gdelt10_daily_export_urls(start: dt.date, end: dt.date) -> Iterator[str]:
    d = start
    while d <= end:
        yield f"http://data.gdeltproject.org/events/{d:%Y%m%d}.export.CSV.zip"
        d += dt.timedelta(days=1)


def _source_timestamp_gdelt10_url(url: str) -> str:
    name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    if not name.endswith(".export.CSV.zip") or len(name) < 8:
        raise ValueError(f"not a GDELT 1.0 daily export URL: {url}")
    day = name[:8]
    dt0 = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC)
    return format_datetime_z(dt0)


def _http_get_bytes(
    url: str,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[int, bytes | None]:
    """Return (status, body). 404 returns (404, None). 2xx returns (code, data)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(urllib.request.url2pathname(parsed.path if parsed.scheme else url))
        return 200, path.read_bytes()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                code = int(response.getcode() or 200)
                return code, response.read()
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            if code == 404:
                return 404, None
            last_error = exc
            if code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                return code, None
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(retry_backoff_seconds * (attempt + 1))
    logger.warning("GET failed for %s after %s retries: %s", url, max_retries, last_error)
    return 0, None


def _arab_spring_fetch_config(
    *, event_start: dt.date, event_end: dt.date
) -> dict[str, Any]:
    return {
        "source_name": ARAB_SPRING_GDELT_SOURCE_NAME,
        "countries": sorted(ARAB_SPRING_COUNTRY_CODES),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "gdelt_version": "1.0",
    }


def _write_arab_spring_fetch_manifest(
    out_dir: Path,
    payload: dict[str, Any],
    *,
    lock: threading.Lock,
) -> None:
    with lock:
        write_json_atomic(out_dir / "fetch_manifest.json", payload)


def _process_gdelt10_arab_spring_day(
    url: str,
    day: dt.date,
    *,
    out_dir: Path,
    event_start: dt.date,
    event_end: dt.date,
    force: bool,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    fragment_name = f"arab_spring_{day:%Y%m%d}.jsonl"
    relative_path = Path(fragment_name)
    absolute_path = out_dir / relative_path
    if absolute_path.exists() and not force:
        n_lines = 0
        with open_text_auto(absolute_path, "r") as handle:
            for line in handle:
                if line.strip():
                    n_lines += 1
        return {
            "url": url,
            "day": day.isoformat(),
            "status": "skipped",
            "raw_row_count": 0,
            "kept_row_count": n_lines,
            "fragment_path": fragment_name,
            "error": None,
        }

    free = shutil.disk_usage(out_dir).free
    if free < 20 * 1024**3:
        logger.warning(
            "low disk: free space under 20GB on out_dir (bytes free=%s): %s",
            free,
            out_dir,
        )
    code, zip_bytes = _http_get_bytes(
        url, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds
    )
    if code == 404:
        return {
            "url": url,
            "day": day.isoformat(),
            "status": "not_found",
            "raw_row_count": 0,
            "kept_row_count": 0,
            "fragment_path": None,
            "error": None,
        }
    if code not in range(200, 300) or zip_bytes is None:
        return {
            "url": url,
            "day": day.isoformat(),
            "status": "failed",
            "raw_row_count": 0,
            "kept_row_count": 0,
            "fragment_path": None,
            "error": f"GET status {code}",
        }
    logger.warning(
        "GDELT 1.0: skipping MD5 verification for one export (no masterfile for daily URLs): %s",
        url,
    )
    retrieved_at = format_datetime_z(utc_now())
    ts = _source_timestamp_gdelt10_url(url)
    metadata = {
        "_source_file_url": url,
        "_source_file_timestamp": ts,
        "_retrieved_at": retrieved_at,
    }
    try:
        rows = parse_gdelt_zip_bytes(
            zip_bytes, metadata=metadata, gdelt_version="1.0"
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "url": url,
            "day": day.isoformat(),
            "status": "failed",
            "raw_row_count": 0,
            "kept_row_count": 0,
            "fragment_path": None,
            "error": str(exc),
        }
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        match_map = {k: str(v) for k, v in row.items() if not str(k).startswith("_")}
        if not row_matches_arab_spring(
            match_map, event_start=event_start, event_end=event_end
        ):
            continue
        kept_rows.append({str(k): str(v) for k, v in row.items()})

    fragment_path_str: str | None = None
    if kept_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_path = out_dir / f".{fragment_name}.tmp"
        with open_text_auto(temp_path, "w") as handle:
            for row in kept_rows:
                handle.write(_json_dump_line(row))
        temp_path.replace(absolute_path)
        fragment_path_str = fragment_name
    return {
        "url": url,
        "day": day.isoformat(),
        "status": "ok",
        "raw_row_count": len(rows),
        "kept_row_count": len(kept_rows),
        "fragment_path": fragment_path_str,
        "error": None,
    }


def parse_gdelt_zip_bytes(
    zip_bytes: bytes,
    *,
    metadata: dict[str, Any] | None = None,
    gdelt_version: Literal["1.0", "2.0"] = "2.0",
) -> list[dict[str, Any]]:
    attached_metadata = metadata or {}
    column_names: Sequence[str] = (
        GDELT_V1_EVENT_COLUMNS if gdelt_version == "1.0" else GDELT_V2_EVENT_COLUMNS
    )
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("GDELT zip contains no CSV member")
        with archive.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text, delimiter="\t")
            for fields in reader:
                row = map_gdelt_event_row(fields, column_names=column_names)
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


def fetch_arab_spring(
    *,
    event_start: dt.date,
    event_end: dt.date,
    out_dir: Path,
    workers: int = 2,
    force: bool = False,
    allow_partial: bool = False,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    raw_retention: RawRetention = "none",
    progress: bool = False,
) -> dict[str, Any]:
    """
    Download GDELT 1.0 daily event exports, filter to Arab Spring area rows, one JSONL per day.

    When ``raw_retention`` is ``"none"`` (the default for this pipeline as with France),
    the raw .zip is not retained; filtered rows are still written to ``arab_spring_*.jsonl``.
    """
    if event_end < event_start:
        raise ValueError("event_end before event_start")
    if raw_retention not in {"none", "compressed", "full"}:
        raise ValueError(f"unknown raw retention: {raw_retention}")
    if raw_retention != "none":
        raise NotImplementedError("Arab Spring fetch only supports raw_retention=none for v0.1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    day_iter = (event_start + dt.timedelta(days=d) for d in range((event_end - event_start).days + 1))
    days_urls = list(
        zip(day_iter, iter_gdelt10_daily_export_urls(event_start, event_end), strict=True)
    )
    config = _arab_spring_fetch_config(event_start=event_start, event_end=event_end)

    not_found_404: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    done_index = 0

    def scan_arab_spring_fragments() -> tuple[int, int]:
        """(total_jsonl_line_count, number_of_arab_spring_*.jsonl files)."""
        n_lines = 0
        n_frags = 0
        for p in sorted(out_dir.glob("arab_spring_*.jsonl")):
            n_frags += 1
            with open_text_auto(p, "r") as handle:
                n_lines += sum(1 for line in handle if line.strip())
        return n_lines, n_frags

    def current_manifest() -> dict[str, Any]:
        rows_written, files_fetched = scan_arab_spring_fragments()
        return {
            "context": "arab_spring",
            "countries": list(config["countries"]),
            "date_start": event_start.isoformat(),
            "date_end": event_end.isoformat(),
            "files_fetched": files_fetched,
            "rows_written": rows_written,
            "not_found_404_count": not_found_404,
            "failed_count": failed_count,
            "skipped_existing_count": skipped_count,
            "days_completed": done_index,
            "days_total": len(days_urls),
            "fetch_completed_at": format_datetime_z(utc_now()),
            "gdelt_version": "1.0",
            "source_name": ARAB_SPRING_GDELT_SOURCE_NAME,
        }

    _write_arab_spring_fetch_manifest(
        out_dir, current_manifest(), lock=lock
    )

    def run_day(day: dt.date, url: str) -> dict[str, Any]:
        return _process_gdelt10_arab_spring_day(
            url,
            day,
            out_dir=out_dir,
            event_start=event_start,
            event_end=event_end,
            force=force,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    max_w = max(1, workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
        future_to: dict[concurrent.futures.Future[dict[str, Any]], tuple[dt.date, str]] = {}
        it = iter(days_urls)
        for _ in range(min(max_w * 2, len(days_urls))):
            try:
                d0, u0 = next(it)
            except StopIteration:
                break
            fut = executor.submit(run_day, d0, u0)
            future_to[fut] = (d0, u0)
        while future_to:
            done, _ = concurrent.futures.wait(
                future_to, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                future_to.pop(fut)
                result = fut.result()
                st = str(result.get("status") or "")
                if st == "ok":
                    pass
                elif st == "not_found":
                    not_found_404 += 1
                elif st == "failed":
                    failed_count += 1
                elif st == "skipped":
                    skipped_count += 1
                done_index += 1
                _write_arab_spring_fetch_manifest(
                    out_dir, current_manifest(), lock=lock
                )
                if progress and (
                    done_index % 50 == 0
                    or done_index == len(days_urls)
                ):
                    n_r, n_f = scan_arab_spring_fragments()
                    print(
                        f"[fetch-arab-spring] {done_index}/{len(days_urls)} "
                        f"fragments={n_f} rows={n_r} 404s={not_found_404} failed={failed_count}",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    d1, u1 = next(it)
                except StopIteration:
                    continue
                f2 = executor.submit(run_day, d1, u1)
                future_to[f2] = (d1, u1)

    if failed_count and not allow_partial:
        raise RuntimeError(
            f"{failed_count} GDELT 1.0 day(s) failed; rerun with --allow-partial to ignore"
        )
    return current_manifest()


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
    arab.add_argument(
        "--raw-dir", default="data/gdelt/raw/arab_spring", help="Output directory for JSONL and fetch_manifest.json"
    )
    arab.add_argument("--event-start", required=True)
    arab.add_argument("--event-end", required=True)
    arab.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel download workers (default: 2; raise for more throughput on fast links)",
    )
    arab.add_argument("--force", action="store_true")
    arab.add_argument(
        "--allow-partial", action="store_true", help="Do not fail if some day downloads fail (non-404)"
    )
    arab.add_argument("--max-retries", type=int, default=3)
    arab.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    arab.add_argument(
        "--progress",
        action="store_true",
        help="Print periodic progress to stderr (known-good 20130401 is a good smoke date before a long run)",
    )
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
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                out_dir=Path(args.raw_dir),
                workers=args.workers,
                force=args.force,
                allow_partial=args.allow_partial,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                raw_retention="none",
                progress=args.progress,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
