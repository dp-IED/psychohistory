#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path
from email.utils import parsedate_to_datetime

UTC = dt.timezone.utc

# Priority-3 seed macro bundle (economics + policy context)
BLS_SERIES = {
    "CUUR0000SA0": "cpi_u_all_items_unadjusted",
    "LNS14000000": "unemployment_rate",
    "CES0000000001": "nonfarm_payrolls_total",
}

# FRED API requires key. Script will skip cleanly if absent.
FRED_SERIES = {
    "DFF": "effective_federal_funds_rate",
    "CPIAUCSL": "cpi_all_items_sa",
    "UNRATE": "unemployment_rate",
    "PAYEMS": "nonfarm_payrolls",
    "GDPC1": "real_gdp",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "T10Y2Y": "yield_curve_10y_minus_2y",
}


def fmt_z(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_cutoff_bounds(seed_path: Path) -> tuple[dt.datetime, dt.datetime]:
    rows = json.loads(seed_path.read_text(encoding="utf-8"))
    cuts = []
    for r in rows:
        raw = str(r.get("cutoff_t") or "").strip()
        if not raw:
            continue
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            cuts.append(dt.datetime.fromisoformat(raw).astimezone(UTC))
        except Exception:
            pass
    if not cuts:
        now = dt.datetime.now(tz=UTC)
        return now - dt.timedelta(days=365), now
    return min(cuts), max(cuts)


def http_get_json(url: str, timeout: float = 45.0, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "psychohistory-priority3-macro/0.1", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bls(start_year: int, end_year: int, timeout: float) -> tuple[list[dict], dict]:
    payload = json.dumps({"seriesid": list(BLS_SERIES.keys()), "startyear": str(start_year), "endyear": str(end_year)}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=payload,
        headers={"User-Agent": "psychohistory-priority3-macro/0.1", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    records = []
    status = {
        "status": obj.get("status"),
        "messages": obj.get("message", []),
        "series_returned": 0,
        "records": 0,
    }

    for s in obj.get("Results", {}).get("series", []):
        sid = s.get("seriesID")
        metric = BLS_SERIES.get(sid, sid)
        status["series_returned"] += 1
        for row in s.get("data", []):
            per = row.get("period", "")
            if per == "M13":
                continue
            year = int(row.get("year"))
            if per.startswith("M"):
                month = int(per[1:])
                release_ts = dt.datetime(year, month, 15, 13, 30, tzinfo=UTC)  # PIT proxy (official monthly release window)
            else:
                release_ts = dt.datetime(year, 1, 15, 13, 30, tzinfo=UTC)
            records.append(
                {
                    "source_name": "bls_public_api_v2",
                    "series_id": sid,
                    "metric": metric,
                    "period": f"{row.get('year')}-{row.get('period')}",
                    "value": row.get("value"),
                    "release_time": fmt_z(release_ts),
                    "observed_time": fmt_z(release_ts),
                    "footnotes": [f.get("text") for f in row.get("footnotes", []) if isinstance(f, dict) and f.get("text")],
                    "pit_policy": "release_time_proxy_monthly_1330Z",
                }
            )
    status["records"] = len(records)
    return records, status


def fetch_fred(start_date: dt.date, end_date: dt.date, api_key: str, timeout: float) -> tuple[list[dict], dict]:
    if not api_key:
        return [], {"enabled": False, "reason": "FRED_API_KEY missing", "series_returned": 0, "records": 0}

    out = []
    series_ok = 0
    series_failed = []
    for sid, metric in FRED_SERIES.items():
        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
        }
        url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)

        obj = None
        last_err = None
        for _ in range(3):
            try:
                obj = http_get_json(url, timeout=timeout)
                break
            except Exception as e:
                last_err = e

        if obj is None:
            series_failed.append({"series_id": sid, "error": str(last_err) if last_err else "unknown"})
            continue

        obs = obj.get("observations", [])
        if obs:
            series_ok += 1
        for row in obs:
            v = row.get("value")
            if v in (None, "."):
                continue
            # release_time approximation from realtime_start (first available vintage date)
            rt = row.get("realtime_start") or row.get("date")
            try:
                release_ts = dt.datetime.fromisoformat(rt + "T13:30:00+00:00")
            except Exception:
                release_ts = dt.datetime.combine(start_date, dt.time(13, 30), tzinfo=UTC)
            out.append(
                {
                    "source_name": "fred_api",
                    "series_id": sid,
                    "metric": metric,
                    "period": row.get("date"),
                    "value": v,
                    "release_time": fmt_z(release_ts),
                    "observed_time": fmt_z(release_ts),
                    "pit_policy": "release_time_from_realtime_start_proxy",
                }
            )
    return out, {"enabled": True, "series_returned": series_ok, "records": len(out), "series_failed": series_failed}


def fetch_treasury_yields(start_date: dt.date, end_date: dt.date, timeout: float) -> tuple[list[dict], dict]:
    # FiscalData endpoint with yield curve tenors
    fields = [
        "record_date",
        "avg_interest_rate_amt",
        "security_desc",
        "security_type_desc",
    ]
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
    filt = f"record_date:gte:{start_date.isoformat()},record_date:lte:{end_date.isoformat()}"
    url = f"{base}?filter={urllib.parse.quote(filt)}&page[size]=10000"
    obj = http_get_json(url, timeout=timeout)
    rows = obj.get("data", [])
    out = []
    for r in rows:
        d = r.get("record_date")
        try:
            release_ts = dt.datetime.fromisoformat(d + "T21:00:00+00:00")
        except Exception:
            continue
        out.append(
            {
                "source_name": "us_treasury_fiscaldata",
                "series_id": "UST_AVG_INTEREST_RATES",
                "metric": (r.get("security_desc") or "unknown_security").lower().replace(" ", "_"),
                "period": d,
                "value": r.get("avg_interest_rate_amt"),
                "release_time": fmt_z(release_ts),
                "observed_time": fmt_z(release_ts),
                "security_type_desc": r.get("security_type_desc"),
                "pit_policy": "release_time_proxy_daily_2100Z",
            }
        )
    return out, {"records": len(out)}


def fetch_fed_monetary_feed(start_date: dt.date, end_date: dt.date, timeout: float) -> tuple[list[dict], dict]:
    req = urllib.request.Request(
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        headers={"User-Agent": "Mozilla/5.0 (psychohistory-priority3-macro)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        xml_text = resp.read().decode("utf-8", errors="ignore")

    # Lightweight RSS parsing without external deps
    items = []
    chunks = xml_text.split("<item>")
    for ch in chunks[1:]:
        part = ch.split("</item>", 1)[0]
        def tag(name: str) -> str:
            a = part.find(f"<{name}>")
            b = part.find(f"</{name}>")
            if a == -1 or b == -1:
                return ""
            return part[a + len(name) + 2:b].strip()
        title = tag("title")
        link = tag("link")
        pub = tag("pubDate")
        if not pub:
            continue
        try:
            pdt = parsedate_to_datetime(pub).astimezone(UTC)
        except Exception:
            continue
        if pdt.date() < start_date or pdt.date() > end_date + dt.timedelta(days=365):
            continue
        items.append(
            {
                "source_name": "federal_reserve_press_monetary_rss",
                "series_id": "FOMC_OFFICIAL_COMMUNICATION",
                "metric": "fomc_communications",
                "period": pdt.date().isoformat(),
                "value": title,
                "release_time": fmt_z(pdt),
                "observed_time": fmt_z(pdt),
                "source_url": link,
                "pit_policy": "official_publication_time",
            }
        )
    return items, {"records": len(items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=".context/polymarket_30_seed_coverage_audit.json")
    ap.add_argument("--out-dir", default="data/macro/raw/seed_official_macro")
    ap.add_argument("--start-pad-days", type=int, default=60)
    ap.add_argument("--end-pad-days", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    seed = Path(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_cut, max_cut = parse_cutoff_bounds(seed)
    start_date = (min_cut - dt.timedelta(days=args.start_pad_days)).date()
    end_date = (max_cut + dt.timedelta(days=args.end_pad_days)).date()

    bls_records, bls_stats = fetch_bls(start_date.year, end_date.year, timeout=args.timeout)
    fred_records, fred_stats = fetch_fred(start_date, end_date, api_key=(__import__("os").getenv("FRED_API_KEY", "")), timeout=args.timeout)
    treasury_records, treasury_stats = fetch_treasury_yields(start_date, end_date, timeout=args.timeout)
    fed_records, fed_stats = fetch_fed_monetary_feed(start_date, end_date, timeout=args.timeout)

    all_records = bls_records + fred_records + treasury_records + fed_records
    all_records.sort(key=lambda r: (r.get("release_time") or "", r.get("source_name") or "", r.get("series_id") or ""))

    jsonl_path = out_dir / "seed_official_macro_events.jsonl"
    manifest_path = out_dir / "fetch_manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "ok": True,
        "created_at": fmt_z(dt.datetime.now(tz=UTC)),
        "seed": str(seed.resolve()),
        "date_window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "pit_policy": "release/publication-time observability",
        "sources": {
            "bls": bls_stats,
            "fred": fred_stats,
            "treasury": treasury_stats,
            "fed_monetary_feed": fed_stats,
        },
        "records_total": len(all_records),
        "outputs": {
            "jsonl": str(jsonl_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "notes": [
            "FRED ingestion is conditional on FRED_API_KEY.",
            "BLS/Treasury release timestamps currently use deterministic PIT proxies where exact intraday release time is not exposed by endpoint response.",
        ],
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "records_total": len(all_records),
        "bls_records": bls_stats.get("records", 0),
        "fred_records": fred_stats.get("records", 0),
        "treasury_records": treasury_stats.get("records", 0),
        "fed_records": fed_stats.get("records", 0),
        "jsonl": str(jsonl_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
