"""Normalize filtered GDELT raw fragments into a point-in-time event tape."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Final, Literal, Sequence

from pydantic import BaseModel

from ingest.io_utils import open_text_auto, write_json_atomic
from ingest.gdelt_raw import (
    ARAB_SPRING_COUNTRY_CODES,
    ARAB_SPRING_GDELT_SOURCE_NAME,
    _action_geo_in_arab_spring_area,
    format_datetime_z,
    parse_datetime_utc,
    parse_sql_date,
)


UTC = dt.timezone.utc

ARAB_SPRING_GDELT_NORMALIZE: dict[str, Any] = {
    "source_name": ARAB_SPRING_GDELT_SOURCE_NAME,
    "countries": sorted(ARAB_SPRING_COUNTRY_CODES),
    "description": "GDELT 1.0 daily exports; all CAMEO root codes; ActionGeo in EG/TU/LY/SY",
}

# Corrupt 2013 daily zips: skip in normalization audit; do not count line failures as ``invalid``.
KNOWN_BAD_FRAGMENTS: Final[frozenset[str]] = frozenset(
    {
        "arab_spring_20130901.jsonl",
        "arab_spring_20130902.jsonl",
    }
)

_log = logging.getLogger(__name__)


class EventTapeRecord(BaseModel):
    source_name: Literal["gdelt_v1_events", "gdelt_v2_events", "acled", "acled_v3"]
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


_ARAB_SPRING_FN_MONTHLY = re.compile(r"^arab_spring_(\d{6})_monthly\.jsonl$")
_ARAB_SPRING_FN_DAILY = re.compile(r"^arab_spring_(\d{8})\.jsonl$")


def _gdelt_arab_spring_fragment_calendar_month_key(filename: str) -> str:
    m = _ARAB_SPRING_FN_MONTHLY.match(filename)
    if m:
        s = m.group(1)
        return f"{s[:4]}-{s[4:6]}"
    m = _ARAB_SPRING_FN_DAILY.match(filename)
    if m:
        s = m.group(1)
        return f"{s[:4]}-{s[4:6]}"
    return "unknown"


def _gdelt_arab_spring_event_date_bounds(
    raw_dir: Path,
) -> tuple[dt.date, dt.date]:
    event_start = dt.date(1, 1, 1)
    event_end = dt.date(9999, 12, 31)
    mpath = raw_dir / "fetch_manifest.json"
    if mpath.is_file():
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if m.get("date_start"):
                event_start = dt.date.fromisoformat(str(m["date_start"])[:10])
            if m.get("date_end"):
                event_end = dt.date.fromisoformat(str(m["date_end"])[:10])
        except (TypeError, ValueError, OSError, json.JSONDecodeError):
            pass
    return event_start, event_end


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


def _arab_spring_tape_fips_code(row: dict[str, Any]) -> str | None:
    """2-letter FIPS for ``country_code``; criteria match :func:`_action_geo_in_arab_spring_area` in raw fetch."""
    match = {k: str(v) for k, v in row.items() if not str(k).startswith("_")}
    if not _action_geo_in_arab_spring_area(match):
        return None
    cc = (row.get("ActionGeo_CountryCode") or "").strip()
    if cc in ARAB_SPRING_COUNTRY_CODES:
        return cc
    adm1 = (row.get("ActionGeo_ADM1Code") or "").strip()
    if len(adm1) >= 2 and adm1[:2] in ARAB_SPRING_COUNTRY_CODES:
        return adm1[:2]
    return None


def normalize_gdelt_arab_spring_row(
    row: dict[str, Any],
    *,
    event_start: dt.date,
    event_end: dt.date,
) -> EventTapeRecord | None:
    fips = _arab_spring_tape_fips_code(row)
    if fips is None:
        return None
    event_date = parse_sql_date(_required_text(row, "SQLDATE"))
    if not (event_start <= event_date <= event_end):
        return None
    admin1_code = _none_if_blank(row.get("ActionGeo_ADM1Code")) or f"{fips}_UNKNOWN"
    source_event_id = f"gdelt:{_required_text(row, 'GLOBALEVENTID')}"
    return EventTapeRecord(
        source_name="gdelt_v1_events",
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=parse_datetime_utc(_required_text(row, "DATEADDED")),
        retrieved_at=parse_datetime_utc(_required_text(row, "_retrieved_at")),
        country_code=fips,
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


def _gdelt_row_tape_reject_after_geo(
    row: dict[str, Any], *, event_start: dt.date, event_end: dt.date
) -> Literal["date"] | None:
    """
    If :func:`_arab_spring_tape_fips_code` is non-None, classify ``normalize`` non-match as date
    (outside fetch window) vs malformed (caller treats as ``invalid``).
    """
    try:
        event_date = parse_sql_date(_required_text(row, "SQLDATE"))
    except (TypeError, ValueError):
        return None
    if not (event_start <= event_date <= event_end):
        return "date"
    return None


def audit_gdelt_arab_spring_raw_normalization(
    raw_dir: Path,
    *,
    flag_min_raw: int = 10_000,
    flag_drop_rate_ge: float = 0.005,
) -> dict[str, Any]:
    """
    For each raw ``arab_spring_*.jsonl`` line, compare counts before vs after
    :func:`normalize_gdelt_arab_spring_row` (same date window as :func:`write_arab_spring_merged_tape`).

    Returns per-fragment, per calendar month (``YYYY-MM``), totals, and ``flags`` where the drop
    rate is unusually high.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw directory: {raw_dir}")
    event_start, event_end = _gdelt_arab_spring_event_date_bounds(raw_dir)
    paths = sorted(raw_dir.glob("arab_spring_*.jsonl"))
    totals: dict[str, int] = {
        "raw_lines": 0,
        "ok": 0,
        "invalid": 0,
        "filtered_geo": 0,
        "filtered_date": 0,
        "skipped_known_bad": 0,
    }
    by_month: dict[str, dict[str, int]] = {}
    by_file: list[dict[str, Any]] = []

    for path in paths:
        month_key = _gdelt_arab_spring_fragment_calendar_month_key(path.name)
        if path.name in KNOWN_BAD_FRAGMENTS:
            n_bad = 0
            with open_text_auto(path, "r") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    n_bad += 1
            _log.warning(
                "skipping known-bad GDELT Arab Spring fragment %s (%d lines)",
                path.name,
                n_bad,
            )
            fstats = {
                "raw_lines": n_bad,
                "ok": 0,
                "invalid": 0,
                "filtered_geo": 0,
                "filtered_date": 0,
                "skipped_known_bad": n_bad,
            }
            for k, v in fstats.items():
                totals[k] = totals.get(k, 0) + v
            acc = by_month.setdefault(
                month_key,
                {
                    "raw_lines": 0,
                    "ok": 0,
                    "invalid": 0,
                    "filtered_geo": 0,
                    "filtered_date": 0,
                    "skipped_known_bad": 0,
                },
            )
            for k, v in fstats.items():
                acc[k] += v
            by_file.append(
                {
                    "path": str(path.name),
                    "calendar_month": month_key,
                    **fstats,
                    "dropped": 0,
                    "drop_rate": 0.0,
                }
            )
            continue
        fstats = {
            "raw_lines": 0,
            "ok": 0,
            "invalid": 0,
            "filtered_geo": 0,
            "filtered_date": 0,
            "skipped_known_bad": 0,
        }
        with open_text_auto(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                fstats["raw_lines"] += 1
                row = json.loads(line)
                try:
                    rec = normalize_gdelt_arab_spring_row(
                        row, event_start=event_start, event_end=event_end
                    )
                except (TypeError, ValueError):
                    fstats["invalid"] += 1
                    continue
                if rec is not None:
                    fstats["ok"] += 1
                    continue
                if _arab_spring_tape_fips_code(row) is None:
                    fstats["filtered_geo"] += 1
                elif (
                    _gdelt_row_tape_reject_after_geo(
                        row, event_start=event_start, event_end=event_end
                    )
                    == "date"
                ):
                    fstats["filtered_date"] += 1
                else:
                    fstats["invalid"] += 1

        for k, v in fstats.items():
            totals[k] = totals.get(k, 0) + v
        acc = by_month.setdefault(
            month_key,
            {
                "raw_lines": 0,
                "ok": 0,
                "invalid": 0,
                "filtered_geo": 0,
                "filtered_date": 0,
                "skipped_known_bad": 0,
            },
        )
        for k, v in fstats.items():
            acc[k] += v
        sk = int(fstats.get("skipped_known_bad", 0))
        dropped = fstats["raw_lines"] - fstats["ok"] - sk
        eff = fstats["raw_lines"] - sk
        rate = (dropped / eff) if eff else 0.0
        by_file.append(
            {
                "path": str(path.name),
                "calendar_month": month_key,
                **fstats,
                "dropped": dropped,
                "drop_rate": rate,
            }
        )

    def _m_stats(m: str) -> dict[str, Any]:
        st: dict[str, int] = by_month.get(m) or {
            "raw_lines": 0,
            "ok": 0,
            "invalid": 0,
            "filtered_geo": 0,
            "filtered_date": 0,
            "skipped_known_bad": 0,
        }
        raw = st["raw_lines"]
        sk = int(st.get("skipped_known_bad", 0))
        dropped = raw - st["ok"] - sk
        eff = raw - sk
        rate = (dropped / eff) if eff else 0.0
        return {**st, "dropped": dropped, "drop_rate": rate}

    by_month_out = {m: _m_stats(m) for m in sorted(by_month)}

    total_raw = totals["raw_lines"]
    total_sk = int(totals.get("skipped_known_bad", 0))
    total_dropped = total_raw - totals["ok"] - total_sk
    eff_total = total_raw - total_sk
    total_rate = (total_dropped / eff_total) if eff_total else 0.0
    flags: list[dict[str, Any]] = []
    for m, st in by_month_out.items():
        r, dr = st["raw_lines"], st["drop_rate"]
        if r >= flag_min_raw and dr >= flag_drop_rate_ge:
            flags.append(
                {
                    "calendar_month": m,
                    "raw_lines": r,
                    "dropped": st["dropped"],
                    "drop_rate": dr,
                }
            )

    return {
        "raw_dir": str(raw_dir.resolve()),
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "totals": {**totals, "dropped": total_dropped, "drop_rate": total_rate},
        "by_month": by_month_out,
        "by_file": by_file,
        "flags": flags,
        "flag_criteria": {
            "min_raw_lines": flag_min_raw,
            "drop_rate_ge": flag_drop_rate_ge,
        },
    }


def _iter_gdelt_arab_spring_jsonl(
    raw_dir: Path,
) -> tuple[int, list[EventTapeRecord], int]:
    paths = sorted(raw_dir.glob("arab_spring_*.jsonl"))
    input_count = 0
    records: list[EventTapeRecord] = []
    invalid = 0
    event_start, event_end = _gdelt_arab_spring_event_date_bounds(raw_dir)
    for path in paths:
        with open_text_auto(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                input_count += 1
                row = json.loads(line)
                try:
                    rec = normalize_gdelt_arab_spring_row(
                        row, event_start=event_start, event_end=event_end
                    )
                except (TypeError, ValueError):
                    invalid += 1
                    continue
                if rec is None:
                    continue
                records.append(rec)
    return input_count, records, invalid


def _acled_fetch_expected_rows(acled_raw_dir: Path) -> int | None:
    """``fetch_metadata.json`` from :mod:`ingest.acled_raw` (CSV); not a single ``fetch_manifest.json``."""
    path = acled_raw_dir / "fetch_metadata.json"
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = meta.get("accepted_row_count")
    if v is None:
        return None
    return int(v)


def _gdelt_arab_fetch_expected_rows(gdelt_raw_dir: Path) -> int | None:
    path = gdelt_raw_dir / "fetch_manifest.json"
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = meta.get("rows_written")
    if v is None:
        return None
    return int(v)


def _glob_acled_page_fragments(acled_raw_dir: Path) -> list[Path]:
    frag = acled_raw_dir / "fragments"
    if not frag.is_dir():
        return []
    return sorted(
        [*frag.glob("page_*.jsonl"), *frag.glob("page_*.jsonl.gz")],
    )


def _iter_acled_fragment_tape(
    acled_raw_dir: Path,
) -> tuple[int, list[EventTapeRecord], int]:
    from ingest.acled_tape import normalize_acled_csv_row

    paths = _glob_acled_page_fragments(acled_raw_dir)
    n_in = 0
    n_invalid = 0
    out: list[EventTapeRecord] = []
    for path in paths:
        with open_text_auto(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                n_in += 1
                row = json.loads(line)
                try:
                    r_at_s = str(row.get("_retrieved_at") or "")
                    retrieved = (
                        parse_datetime_utc(r_at_s) if r_at_s else None
                    )
                except (TypeError, ValueError):
                    retrieved = None
                if retrieved is None:
                    n_invalid += 1
                    continue
                try:
                    rec = normalize_acled_csv_row(
                        row,
                        retrieved_at=retrieved,
                        input_basename=str(row.get("_csv_input_file") or path.name),
                        csv_row_index=int(row.get("_csv_row") or 0),
                    )
                except (TypeError, ValueError, KeyError):
                    n_invalid += 1
                    continue
                out.append(rec)
    return n_in, out, n_invalid


def write_arab_spring_merged_tape(
    *,
    gdelt_raw_dir: Path,
    acled_raw_dir: Path,
    out_path: Path,
    cleanup_fragments: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    gd_in, gd_recs, gd_invalid = _iter_gdelt_arab_spring_jsonl(gdelt_raw_dir)
    ac_in, ac_recs, ac_invalid = _iter_acled_fragment_tape(acled_raw_dir)
    if not gd_recs and not ac_recs and not allow_empty:
        raise ValueError("merge input is empty (no GDELT or ACLED records); use allow_empty to write empty")
    by_id: dict[str, EventTapeRecord] = {}
    dedup_dropped = 0
    for rec in gd_recs:
        if rec.source_event_id in by_id:
            continue
        by_id[rec.source_event_id] = rec
    n_after_gd = len(by_id)
    for rec in ac_recs:
        if rec.source_event_id in by_id:
            dedup_dropped += 1
            continue
        by_id[rec.source_event_id] = rec
    total = len(by_id)
    if total != n_after_gd + len(ac_recs) - dedup_dropped:
        raise ValueError("internal: dedup invariant failed")
    if total == 0 and not allow_empty:
        raise ValueError("merged event tape is empty after normalize/dedup")
    if allow_empty and total == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        man = {
            "gdelt_record_count": len(gd_recs),
            "acled_record_count": len(ac_recs),
            "total_record_count": 0,
            "dedup_dropped": dedup_dropped,
            "tape_path": str(out_path.resolve()),
            "completed_at": format_datetime_z(dt.datetime.now(tz=UTC)),
        }
        write_json_atomic(out_path.with_name("tape_manifest.json"), man)
        return {**man, "dedup_dropped": dedup_dropped}

    if cleanup_fragments:
        ex_g = _gdelt_arab_fetch_expected_rows(gdelt_raw_dir)
        ex_a = _acled_fetch_expected_rows(acled_raw_dir)
        if ex_g is None or ex_a is None:
            raise ValueError(
                "cleanup_fragments needs GDELT fetch_manifest.json rows_written and "
                "ACLED fetch_metadata.json accepted_row_count; refusing delete"
            )
        if ex_g != gd_in or ex_a != ac_in:
            raise ValueError(
                "on-disk line counts do not match fetch metadata; refusing to delete fragments"
            )
        if n_after_gd + len(ac_recs) - dedup_dropped != total:
            raise ValueError("dedup accounting mismatch; refusing to delete fragments")

    output_records = sorted(by_id.values(), key=_event_sort_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for record in output_records:
            handle.write(record.model_dump_json() + "\n")
    completed = format_datetime_z(dt.datetime.now(tz=UTC))
    manifest = {
        "gdelt_record_count": len(gd_recs),
        "acled_record_count": len(ac_recs),
        "total_record_count": total,
        "dedup_dropped": dedup_dropped,
        "tape_path": str(out_path.resolve()),
        "completed_at": completed,
    }
    write_json_atomic(out_path.with_name("tape_manifest.json"), manifest)
    if cleanup_fragments:
        for p in gdelt_raw_dir.glob("arab_spring_*.jsonl"):
            p.unlink()
        afrag = acled_raw_dir / "fragments"
        if afrag.is_dir():
            for p in afrag.glob("page_*.jsonl*"):
                p.unlink()
    return {
        **manifest,
        "gdelt_input_lines": gd_in,
        "acled_input_lines": ac_in,
        "gdelt_invalid": gd_invalid,
        "acled_invalid": ac_invalid,
    }


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
    merge = subparsers.add_parser("merge-arab-spring")
    merge.add_argument("--raw-gdelt", required=True)
    merge.add_argument("--acled-raw", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument(
        "--cleanup-fragments",
        action="store_true",
        help="After successful merge, delete arab_spring_*.jsonl and ACLED page_*.jsonl fragments if fetch metadata row counts match",
    )
    merge.add_argument("--allow-empty", action="store_true")
    audit = subparsers.add_parser(
        "audit-gdelt-arab-spring-raw",
        help="Count raw jsonl lines vs rows accepted by normalize_gdelt_arab_spring_row (per file and per calendar month)",
    )
    audit.add_argument(
        "--raw",
        type=Path,
        default=Path("data/gdelt/raw/arab_spring"),
        help="Directory containing fetch_manifest.json and arab_spring_*.jsonl",
    )
    audit.add_argument(
        "--flag-min-raw",
        type=int,
        default=10_000,
        help="Only emit month flags when raw line count in that month is at least this (default: 10000)",
    )
    audit.add_argument(
        "--flag-drop-rate-ge",
        type=float,
        default=0.005,
        help="Only emit month flags when (raw-ok)/raw >= this (default: 0.005)",
    )
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
    if args.command == "merge-arab-spring":
        try:
            write_arab_spring_merged_tape(
                gdelt_raw_dir=Path(args.raw_gdelt),
                acled_raw_dir=Path(args.acled_raw),
                out_path=Path(args.out),
                cleanup_fragments=args.cleanup_fragments,
                allow_empty=args.allow_empty,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "audit-gdelt-arab-spring-raw":
        try:
            out = audit_gdelt_arab_spring_raw_normalization(
                Path(args.raw),
                flag_min_raw=args.flag_min_raw,
                flag_drop_rate_ge=args.flag_drop_rate_ge,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
