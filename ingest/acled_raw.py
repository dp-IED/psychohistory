"""Raw-file ingestion for ACLED France protest events."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from ingest.acled_tape import AvailabilityPolicy, normalize_acled_row
from ingest.event_warehouse import upsert_records
from ingest.io_utils import open_text_auto, write_json_atomic
from ingest.paths import resolve_data_root, warehouse_path as default_warehouse_path


SOURCE_NAME = "acled"
DOMAIN = "france_protest"
TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL = "https://acleddata.com/api/acled/read"
ARAB_SPRING_ACLED_COUNTRY_PARAM = "Egypt|Tunisia|Libya|Syria"
ARAB_SPRING_ACLED_FIELDS = (
    "event_id_cnty,event_date,country,admin1,actor1,actor2,event_type,sub_event_type,fatalities,notes"
)
UTC = dt.timezone.utc
RawRetention = Literal["none", "compressed", "full"]

ACLED_FIELDS = (
    "event_id_cnty",
    "event_date",
    "year",
    "time_precision",
    "disorder_type",
    "event_type",
    "sub_event_type",
    "actor1",
    "assoc_actor_1",
    "inter1",
    "actor2",
    "assoc_actor_2",
    "inter2",
    "interaction",
    "civilian_targeting",
    "iso",
    "region",
    "country",
    "admin1",
    "admin2",
    "admin3",
    "location",
    "latitude",
    "longitude",
    "geo_precision",
    "source",
    "source_scale",
    "notes",
    "fatalities",
    "tags",
    "timestamp",
)


@dataclass(frozen=True)
class AcledCredentials:
    username: str
    password: str


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
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def credentials_from_env() -> AcledCredentials:
    username = os.environ.get("ACLED_USERNAME") or os.environ.get("ACLED_EMAIL")
    password = os.environ.get("ACLED_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "missing ACLED credentials; source ~/.config/psychohistory/acled.env "
            "and set ACLED_USERNAME/ACLED_EMAIL plus ACLED_PASSWORD"
        )
    return AcledCredentials(username=username, password=password)


def _endpoint_path(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.path or "/"


def _json_dump_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _request_json(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "psychohistory-france-protest-ingest/0.1",
        **(headers or {}),
    }
    last_error: Exception | None = None
    last_http_body: str = ""
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers=request_headers,
                method=method,
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            try:
                last_http_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                last_http_body = ""
            if 400 <= exc.code < 500:
                break
        except Exception as exc:  # pragma: no cover - exercised by integration use.
            last_error = exc
        if attempt < max_retries:
            time.sleep(retry_backoff_seconds * (attempt + 1))
    if isinstance(last_error, urllib.error.HTTPError):
        detail = last_http_body.strip()
        if len(detail) > 800:
            detail = f"{detail[:800]}…"
        extra = f" body={detail}" if detail else ""
        raise RuntimeError(
            f"ACLED request failed: status={last_error.code} endpoint={_endpoint_path(url)}{extra}"
        ) from last_error
    raise RuntimeError(f"ACLED request failed: endpoint={_endpoint_path(url)}") from last_error


def get_access_token(
    *,
    credentials: AcledCredentials,
    token_url: str = TOKEN_URL,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> str:
    """OAuth password grant per https://acleddata.com/api-documentation/getting-started"""

    form = urllib.parse.urlencode(
        {
            "username": credentials.username,
            "password": credentials.password,
            "grant_type": "password",
            "client_id": "acled",
        }
    ).encode("utf-8")
    payload = _request_json(
        token_url,
        method="POST",
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"ACLED token response missing access_token: endpoint={_endpoint_path(token_url)}")
    return token


def _fetch_config(
    *,
    event_start: dt.date,
    event_end: dt.date,
    country: str,
    event_type: str,
    limit: int,
) -> dict[str, Any]:
    return {
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "country": country,
        "event_type": event_type,
        "limit": limit,
        "fields": list(ACLED_FIELDS),
    }


def _page_relative_path(page: int, *, raw_retention: RawRetention) -> Path:
    suffix = ".jsonl.gz" if raw_retention == "compressed" else ".jsonl"
    return Path("fragments") / f"page_{page:06d}{suffix}"


def _api_params(
    *,
    event_start: dt.date,
    event_end: dt.date,
    country: str,
    event_type: str,
    limit: int,
    page: int,
) -> dict[str, str]:
    return {
        "_format": "json",
        "country": country,
        "country_where": "=",
        "event_type": event_type,
        "event_type_where": "=",
        "event_date": f"{event_start.isoformat()}|{event_end.isoformat()}",
        "event_date_where": "BETWEEN",
        "fields": "|".join(ACLED_FIELDS),
        "limit": str(limit),
        "page": str(page),
    }


def _api_params_arab_spring_oauth(
    *,
    event_start: dt.date,
    event_end: dt.date,
    limit: int,
    page: int,
) -> dict[str, str]:
    """OAuth read API: pipe-separated countries, date range, narrow field list (no event_type filter)."""

    return {
        "_format": "json",
        "country": ARAB_SPRING_ACLED_COUNTRY_PARAM,
        "country_where": "|",
        "event_date": f"{event_start.isoformat()}|{event_end.isoformat()}",
        "event_date_where": "BETWEEN",
        "fields": ARAB_SPRING_ACLED_FIELDS,
        "limit": str(limit),
        "page": str(page),
    }


def _fetch_page(
    *,
    access_token: str,
    api_url: str,
    params: dict[str, str],
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    return _request_json(
        url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def fetch_france_protests(
    *,
    event_start: dt.date,
    event_end: dt.date,
    out_dir: Path,
    country: str = "France",
    event_type: str = "Protests",
    limit: int = 5000,
    max_pages: int = 100,
    force: bool = False,
    allow_empty: bool = False,
    token_url: str = TOKEN_URL,
    api_url: str = ACLED_READ_URL,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    raw_retention: RawRetention = "none",
    normalize_to_warehouse: bool = False,
    data_root: Path | str | None = None,
    warehouse_path: Path | str | None = None,
    availability_policy: AvailabilityPolicy = "event_date_lag",
    availability_lag_days: int = 7,
    progress: bool = False,
) -> dict[str, Any]:
    if event_start > event_end:
        raise ValueError("event_start must be on or before event_end")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if raw_retention not in {"none", "compressed", "full"}:
        raise ValueError(f"unknown raw retention: {raw_retention}")
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be non-negative")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    fetch_config = _fetch_config(
        event_start=event_start,
        event_end=event_end,
        country=country,
        event_type=event_type,
        limit=limit,
    )
    if force:
        fragments_dir = out_dir / "fragments"
        if fragments_dir.exists():
            for path in [*fragments_dir.glob("*.jsonl"), *fragments_dir.glob("*.jsonl.gz")]:
                path.unlink()

    resolved_warehouse_path: Path | None = None
    if normalize_to_warehouse:
        resolved_warehouse_path = (
            Path(warehouse_path).expanduser().resolve()
            if warehouse_path is not None
            else default_warehouse_path(resolve_data_root(data_root))
        )

    credentials = credentials_from_env()
    access_token = get_access_token(
        credentials=credentials,
        token_url=token_url,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    manifest_path = out_dir / "fetch_manifest.jsonl"
    completed_pages = 0
    failed_pages = 0
    row_count = 0
    normalized_row_count = 0
    filtered_row_count = 0
    invalid_row_count = 0
    last_update: Any = None
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for page in range(1, max_pages + 1):
            retrieved_at = format_datetime_z(utc_now())
            params = _api_params(
                event_start=event_start,
                event_end=event_end,
                country=country,
                event_type=event_type,
                limit=limit,
                page=page,
            )
            row: dict[str, Any] = {
                "run_id": run_id,
                "fetch_config": fetch_config,
                "raw_retention": raw_retention,
                "normalize_to_warehouse": normalize_to_warehouse,
                "status": "failed",
                "endpoint_path": _endpoint_path(api_url),
                "page": page,
                "limit": limit,
                "row_count": 0,
                "fragment_path": None,
                "retrieved_at": retrieved_at,
                "error": None,
            }
            try:
                payload = _fetch_page(
                    access_token=access_token,
                    api_url=api_url,
                    params=params,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                if int(payload.get("status") or 0) != 200 or payload.get("success") is False:
                    raise RuntimeError(f"ACLED API returned status={payload.get('status')}")
                data = payload.get("data") or []
                if not isinstance(data, list):
                    raise RuntimeError("ACLED API response data is not a list")
                last_update = payload.get("last_update", last_update)

                fragment_path: str | None = None
                normalized_records = []
                hydrated_rows = []
                page_filtered_count = 0
                page_invalid_count = 0
                for item in data:
                    if not isinstance(item, dict):
                        page_invalid_count += 1
                        invalid_row_count += 1
                        continue
                    hydrated = {
                        **item,
                        "_retrieved_at": retrieved_at,
                        "_api_endpoint_path": _endpoint_path(api_url),
                        "_api_page": page,
                        "_api_limit": limit,
                        "_api_query": {
                            key: value
                            for key, value in params.items()
                            if key not in {"page", "limit"}
                        },
                    }
                    hydrated_rows.append(hydrated)
                    if normalize_to_warehouse:
                        try:
                            record = normalize_acled_row(
                                hydrated,
                                event_start=event_start,
                                event_end=event_end,
                                availability_policy=availability_policy,
                                availability_lag_days=availability_lag_days,
                            )
                        except (TypeError, ValueError):
                            page_invalid_count += 1
                            invalid_row_count += 1
                            continue
                        if record is None:
                            page_filtered_count += 1
                            filtered_row_count += 1
                            continue
                        normalized_records.append(record)

                if normalized_records:
                    assert resolved_warehouse_path is not None
                    upsert_records(db_path=resolved_warehouse_path, records=normalized_records)
                    normalized_row_count += len(normalized_records)

                if hydrated_rows and raw_retention != "none":
                    relative_path = _page_relative_path(page, raw_retention=raw_retention)
                    absolute_path = out_dir / relative_path
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = absolute_path.with_name(f".{absolute_path.name}.tmp")
                    if raw_retention == "compressed":
                        temp_path = temp_path.with_suffix(f"{temp_path.suffix}.gz")
                    with open_text_auto(temp_path, "w") as handle:
                        for hydrated in hydrated_rows:
                            handle.write(_json_dump_line(hydrated))
                    temp_path.replace(absolute_path)
                    fragment_path = relative_path.as_posix()

                row.update(
                    {
                        "status": "ok",
                        "row_count": len(data),
                        "normalized_row_count": len(normalized_records),
                        "filtered_row_count": page_filtered_count,
                        "invalid_row_count": page_invalid_count,
                        "fragment_path": fragment_path,
                        "error": None,
                    }
                )
                completed_pages += 1
                row_count += len(data)
                manifest.write(_json_dump_line(row))
                manifest.flush()
                if progress:
                    print(
                        f"[acled-fetch] page={page} rows={len(data)} total_rows={row_count}",
                        file=sys.stderr,
                        flush=True,
                    )
                if len(data) < limit:
                    break
            except Exception as exc:
                failed_pages += 1
                row["error"] = str(exc)
                manifest.write(_json_dump_line(row))
                manifest.flush()
                raise
        else:
            raise RuntimeError(f"ACLED fetch reached max_pages={max_pages}; increase --max-pages")

    if row_count == 0 and not allow_empty:
        raise RuntimeError("ACLED fetch returned zero rows; use --allow-empty to keep metadata only")

    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "domain": DOMAIN,
        "api_endpoint_path": _endpoint_path(api_url),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "country": country,
        "event_type": event_type,
        "limit": limit,
        "max_pages": max_pages,
        "raw_retention": raw_retention,
        "normalize_to_warehouse": normalize_to_warehouse,
        "warehouse_path": str(resolved_warehouse_path) if resolved_warehouse_path is not None else None,
        "availability_policy": availability_policy,
        "availability_lag_days": availability_lag_days,
        "completed_page_count": completed_pages,
        "failed_page_count": failed_pages,
        "row_count": row_count,
        "normalized_row_count": normalized_row_count,
        "filtered_row_count": filtered_row_count,
        "invalid_row_count": invalid_row_count,
        "last_update": last_update,
        "retrieved_at": format_datetime_z(utc_now()),
        "credentials_source": ["ACLED_USERNAME", "ACLED_EMAIL", "ACLED_PASSWORD"],
        "fetch_config": fetch_config,
    }
    (out_dir / "fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def fetch_arab_spring_acled(
    *,
    out_dir: Path,
    event_start: dt.date,
    event_end: dt.date,
    token_url: str = TOKEN_URL,
    api_url: str = ACLED_READ_URL,
    limit: int = 5000,
    max_pages: int = 10_000,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    force: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Fetch Arab Spring rows: OAuth token + Bearer GET to /api/acled/read (ACLED getting-started flow)."""

    if event_start > event_end:
        raise ValueError("event_start must be on or before event_end")
    credentials = credentials_from_env()
    access_token = get_access_token(
        credentials=credentials,
        token_url=token_url,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in out_dir.glob("acled_arab_spring_page_*.jsonl"):
            path.unlink()

    manifest_path = out_dir / "fetch_manifest.json"
    rows_total = 0
    pages_ok = 0

    for page in range(1, max_pages + 1):
        _warn_acled_disk_low(out_dir, label=f"before page {page}")
        retrieved_at = format_datetime_z(utc_now())
        params = _api_params_arab_spring_oauth(
            event_start=event_start,
            event_end=event_end,
            limit=limit,
            page=page,
        )
        payload = _fetch_page(
            access_token=access_token,
            api_url=api_url,
            params=params,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise RuntimeError("ACLED v3 response data is not a list")
        fragment_name = f"acled_arab_spring_page_{page:04d}.jsonl"
        fragment_path = out_dir / fragment_name
        hydrated: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            row = {
                **item,
                "_retrieved_at": retrieved_at,
                "_api_page": page,
            }
            hydrated.append(row)
        if hydrated:
            temp_path = fragment_path.with_name(f".{fragment_path.name}.tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                for row in hydrated:
                    handle.write(_json_dump_line(row))
            temp_path.replace(fragment_path)
        rows_total += len(data)
        pages_ok += 1
        manifest_body = {
            "context": "arab_spring",
            "countries": ["EG", "TU", "LY", "SY"],
            "date_start": event_start.isoformat(),
            "date_end": event_end.isoformat(),
            "files_fetched": pages_ok,
            "rows_written": rows_total,
            "fetch_completed_at": None,
            "acled_api": "oauth_read",
        }
        write_json_atomic(manifest_path, manifest_body)
        _warn_acled_disk_low(out_dir, label=f"after page {page}")
        if progress:
            print(
                f"[acled arab_spring] page={page} rows={len(data)} total_rows={rows_total}",
                file=sys.stderr,
                flush=True,
            )
        if len(data) == 0:
            break
    else:
        raise RuntimeError(f"ACLED fetch exceeded max_pages={max_pages}")

    manifest_final = {
        "context": "arab_spring",
        "countries": ["EG", "TU", "LY", "SY"],
        "date_start": event_start.isoformat(),
        "date_end": event_end.isoformat(),
        "files_fetched": pages_ok,
        "rows_written": rows_total,
        "fetch_completed_at": format_datetime_z(utc_now()),
        "acled_api": "oauth_read",
    }
    write_json_atomic(manifest_path, manifest_final)
    return {
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "row_count": rows_total,
        "page_count": pages_ok,
    }


def _warn_acled_disk_low(path: Path, *, label: str) -> None:
    free = int(shutil.disk_usage(path).free)
    warn_below = 20 * 1024 * 1024 * 1024
    if free < warn_below:
        print(
            f"[acled arab_spring] warning: free disk space {free / (1024**3):.1f} GB below 20 GB ({label})",
            file=sys.stderr,
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-france-protests")
    fetch.add_argument("--event-start", default="2019-01-01")
    fetch.add_argument("--event-end", default="2026-01-04")
    fetch.add_argument("--out", default="data/acled/raw/france_protest")
    fetch.add_argument("--country", default="France")
    fetch.add_argument("--event-type", default="Protests")
    fetch.add_argument("--limit", type=int, default=5000)
    fetch.add_argument("--max-pages", type=int, default=100)
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--allow-empty", action="store_true")
    fetch.add_argument("--token-url", default=TOKEN_URL)
    fetch.add_argument("--api-url", default=ACLED_READ_URL)
    fetch.add_argument("--max-retries", type=int, default=3)
    fetch.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    fetch.add_argument(
        "--raw-retention",
        choices=["none", "compressed", "full"],
        default="none",
    )
    fetch.add_argument("--normalize-to-warehouse", action="store_true")
    fetch.add_argument("--data-root", default=None)
    fetch.add_argument("--warehouse-path", default=None)
    fetch.add_argument(
        "--availability-policy",
        choices=["timestamp", "event_date_lag", "retrieved_at"],
        default="event_date_lag",
    )
    fetch.add_argument("--availability-lag-days", type=int, default=7)

    arab = subparsers.add_parser("fetch-arab-spring")
    arab.add_argument("--out", default="data/acled/raw/arab_spring")
    arab.add_argument(
        "--event-start",
        "--date-start",
        default="2010-01-01",
        help="Inclusive event_date filter start (ISO). Alias: --date-start.",
    )
    arab.add_argument(
        "--event-end",
        "--date-end",
        default="2013-12-31",
        help="Inclusive event_date filter end (ISO). Alias: --date-end.",
    )
    arab.add_argument("--token-url", default=TOKEN_URL)
    arab.add_argument("--api-url", default=ACLED_READ_URL)
    arab.add_argument("--limit", type=int, default=5000)
    arab.add_argument("--max-pages", type=int, default=10_000)
    arab.add_argument("--max-retries", type=int, default=3)
    arab.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    arab.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-france-protests":
        try:
            fetch_france_protests(
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                out_dir=Path(args.out),
                country=args.country,
                event_type=args.event_type,
                limit=args.limit,
                max_pages=args.max_pages,
                force=args.force,
                allow_empty=args.allow_empty,
                token_url=args.token_url,
                api_url=args.api_url,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                raw_retention=args.raw_retention,
                normalize_to_warehouse=args.normalize_to_warehouse,
                data_root=Path(args.data_root) if args.data_root is not None else None,
                warehouse_path=Path(args.warehouse_path) if args.warehouse_path is not None else None,
                availability_policy=args.availability_policy,
                availability_lag_days=args.availability_lag_days,
                progress=True,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "fetch-arab-spring":
        try:
            fetch_arab_spring_acled(
                out_dir=Path(args.out),
                event_start=dt.date.fromisoformat(args.event_start),
                event_end=dt.date.fromisoformat(args.event_end),
                token_url=args.token_url,
                api_url=args.api_url,
                limit=args.limit,
                max_pages=args.max_pages,
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
