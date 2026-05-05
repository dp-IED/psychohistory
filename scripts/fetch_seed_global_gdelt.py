#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from ingest.gdelt_raw import format_datetime_z, parse_gdelt_zip_bytes

UTC = dt.timezone.utc

# Seed-relevant starter set (can be expanded)
DEFAULT_COUNTRIES = ("US", "DE", "TW")
COUNTRY_NAME_HINTS = {
    "US": ("united states",),
    "DE": ("germany",),
    "TW": ("taiwan",),
}
COUNTRY_FIPS_HINTS = {
    "US": ("US",),
    "DE": ("GM", "DE"),
    "TW": ("TW",),
}


def _parse_cutoff_dates(seed_path: Path) -> list[dt.date]:
    rows = json.loads(seed_path.read_text(encoding="utf-8"))
    dates: list[dt.date] = []
    for r in rows:
        raw = str(r.get("cutoff_t") or "").strip()
        if not raw:
            continue
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            d = dt.datetime.fromisoformat(raw).astimezone(UTC).date()
        except Exception:
            continue
        dates.append(d)
    return sorted(set(dates))


def _candidate_days(cutoff_days: list[dt.date], window_days: int, max_days: int) -> list[dt.date]:
    pool: set[dt.date] = set()
    for d in cutoff_days:
        for k in range(-window_days, window_days + 1):
            pool.add(d + dt.timedelta(days=k))
    out = sorted(pool)
    if len(out) > max_days:
        # deterministic truncation: keep earliest max_days
        out = out[:max_days]
    return out


def _daily_url(day: dt.date) -> str:
    return f"http://data.gdeltproject.org/events/{day:%Y%m%d}.export.CSV.zip"


def _extract_country(row: dict[str, str], allowed: set[str]) -> str | None:
    # GDELT 1.0 daily often stores free-text in ActionGeo_CountryCode.
    raw_cc = (row.get("ActionGeo_CountryCode") or "").strip()
    raw_low = raw_cc.lower()
    for iso in sorted(allowed):
        for hint in COUNTRY_NAME_HINTS.get(iso, ()):
            if hint in raw_low:
                return iso

    for key in ("ActionGeo_ADM1Code", "Actor1CountryCode", "Actor2CountryCode"):
        v = (row.get(key) or "").strip().upper()
        if not v:
            continue
        for iso in sorted(allowed):
            for pref in COUNTRY_FIPS_HINTS.get(iso, ()):
                if v.startswith(pref):
                    return iso
    return None


def _fetch_bytes(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "psychohistory-seed-global-gdelt/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=".context/polymarket_30_seed_coverage_audit.json")
    ap.add_argument("--out-dir", default="data/gdelt/raw/seed_global_news")
    ap.add_argument("--window-days", type=int, default=3)
    ap.add_argument("--max-days", type=int, default=45)
    ap.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    ap.add_argument("--timeout", type=float, default=40.0)
    args = ap.parse_args()

    seed_path = Path(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "seed_global_gdelt.jsonl.gz"
    manifest_path = out_dir / "fetch_manifest.json"

    countries = {c.strip().upper() for c in args.countries.split(",") if c.strip()}

    cutoff_days = _parse_cutoff_dates(seed_path)
    days = _candidate_days(cutoff_days, args.window_days, args.max_days)

    stats = {
        "seed": str(seed_path.resolve()),
        "created_at": format_datetime_z(dt.datetime.now(tz=UTC)),
        "countries": sorted(countries),
        "window_days": args.window_days,
        "max_days": args.max_days,
        "candidate_days": len(days),
        "download_ok_days": 0,
        "download_404_days": 0,
        "download_failed_days": 0,
        "rows_total_parsed": 0,
        "rows_kept": 0,
        "days": [],
    }

    with gzip.open(out_jsonl, "wt", encoding="utf-8") as f:
        for day in days:
            url = _daily_url(day)
            day_info = {"day": day.isoformat(), "url": url, "status": "ok", "rows_parsed": 0, "rows_kept": 0}
            try:
                blob = _fetch_bytes(url, timeout=args.timeout)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    day_info["status"] = "404"
                    stats["download_404_days"] += 1
                    stats["days"].append(day_info)
                    continue
                day_info["status"] = f"http_{e.code}"
                stats["download_failed_days"] += 1
                stats["days"].append(day_info)
                continue
            except Exception as e:
                day_info["status"] = f"error:{e.__class__.__name__}"
                stats["download_failed_days"] += 1
                stats["days"].append(day_info)
                continue

            stats["download_ok_days"] += 1
            rows = parse_gdelt_zip_bytes(blob, metadata={"url": url}, gdelt_version="1.0")
            day_info["rows_parsed"] = len(rows)
            stats["rows_total_parsed"] += len(rows)

            for row in rows:
                iso = _extract_country(row, countries)
                if not iso:
                    continue
                # Keep minimal PIT-able news/event payload
                event = {
                    "source_name": "gdelt_v1_events",
                    "source_url": url,
                    "source_day": day.isoformat(),
                    "source_event_id": f"gdelt:{row.get('GLOBALEVENTID','')}",
                    "event_time": row.get("SQLDATE"),
                    "country_code": iso,
                    "raw_action_geo_country": row.get("ActionGeo_CountryCode"),
                    "event_root_code": row.get("EventRootCode"),
                    "event_code": row.get("EventCode"),
                    "actor1_name": row.get("Actor1Name"),
                    "actor2_name": row.get("Actor2Name"),
                    "action_lat": row.get("ActionGeo_Lat"),
                    "action_long": row.get("ActionGeo_Long"),
                    "sourceurl": row.get("SOURCEURL"),
                }
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                day_info["rows_kept"] += 1
                stats["rows_kept"] += 1

            stats["days"].append(day_info)

    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "out": str(out_jsonl.resolve()),
        "manifest": str(manifest_path.resolve()),
        "download_ok_days": stats["download_ok_days"],
        "rows_kept": stats["rows_kept"],
        "countries": sorted(countries),
    }, indent=2))


if __name__ == "__main__":
    main()
