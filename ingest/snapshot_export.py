"""Export weekly France protest forecast snapshots as graph artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from evals.graph_artifact_contract import GRAPH_ARTIFACT_FORMAT, GraphArtifactV1
from ingest.event_tape import EventTapeRecord, load_event_tape


UTC = dt.timezone.utc
LABEL_GRACE_DAYS = 14
WINDOW_DAYS = 7


def _format_datetime_z(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _record_sort_key(record: EventTapeRecord) -> tuple[dt.datetime, dt.date, str]:
    return (
        record.source_available_at.astimezone(UTC),
        record.event_date,
        record.source_event_id,
    )


def _weekly_origins(start: dt.date, end: dt.date) -> list[dt.date]:
    if start.weekday() != 0:
        raise ValueError(f"origin_start must be a Monday: {start.isoformat()}")
    if end.weekday() != 0:
        raise ValueError(f"origin_end must be a Monday: {end.isoformat()}")
    if start > end:
        raise ValueError(
            f"origin_start must be on or before origin_end: {start.isoformat()} > {end.isoformat()}"
        )
    origins: list[dt.date] = []
    current = start
    while current <= end:
        origins.append(current)
        current += dt.timedelta(days=7)
    return origins


def _split_for_origin(origin_date: dt.date) -> str:
    if dt.date(2021, 1, 4) <= origin_date <= dt.date(2024, 12, 30):
        return "development"
    if dt.date(2025, 1, 6) <= origin_date <= dt.date(2025, 12, 29):
        return "holdout"
    raise ValueError(f"origin outside configured splits: {origin_date.isoformat()}")


def actor_id(actor_name: str, country_code: str | None) -> str:
    key = f"{actor_name.strip().lower()}|{(country_code or '').strip()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"actor:{digest}"


def _location_node_id(admin1_code: str) -> str:
    return f"location:FR:{admin1_code}"


def _event_node_id(source_event_id: str) -> str:
    return f"event:{source_event_id}"


def _node_provenance() -> dict[str, list[str]]:
    return {"sources": ["gdelt_v2_events"]}


def _best_location_labels(feature_events: list[EventTapeRecord]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in sorted(feature_events, key=_record_sort_key):
        if record.location_name and record.admin1_code not in labels:
            labels[record.admin1_code] = record.location_name
    return labels


def _scoring_universe(records: list[EventTapeRecord]) -> list[str]:
    """Return the fixed country-qualified GDELT location grid for this tape."""

    return sorted({record.admin1_code for record in records} | {"FR_UNKNOWN"})


def _actor_pairs(record: EventTapeRecord) -> list[tuple[str, str | None, str]]:
    pairs: list[tuple[str, str | None, str]] = []
    if record.actor1_name:
        pairs.append((record.actor1_name, record.actor1_country_code, "actor1"))
    if record.actor2_name:
        pairs.append((record.actor2_name, record.actor2_country_code, "actor2"))
    return pairs


def build_snapshot_payload(
    *,
    records: list[EventTapeRecord],
    origin_date: dt.date,
) -> dict[str, Any]:
    origin_dt = dt.datetime.combine(origin_date, dt.time(), tzinfo=UTC)
    origin_iso = _format_datetime_z(origin_dt)
    window_end = origin_date + dt.timedelta(days=WINDOW_DAYS)
    grace_cutoff = origin_dt + dt.timedelta(days=WINDOW_DAYS + LABEL_GRACE_DAYS)
    split = _split_for_origin(origin_date)

    sorted_records = sorted(records, key=_record_sort_key)
    feature_events = [
        record
        for record in sorted_records
        if record.source_available_at.astimezone(UTC) < origin_dt
        and record.event_date < origin_date
    ]
    scoring_universe = _scoring_universe(sorted_records)
    label_by_admin = _best_location_labels(feature_events)

    primary_candidates = [
        record
        for record in sorted_records
        if origin_date <= record.event_date < window_end
        and record.source_available_at.astimezone(UTC) <= grace_cutoff
    ]
    late_events = [
        record
        for record in sorted_records
        if origin_date <= record.event_date < window_end
        and record.source_available_at.astimezone(UTC) > grace_cutoff
    ]

    target_counts = {admin1_code: 0 for admin1_code in scoring_universe}
    unscored_codes: set[str] = set()
    unscored_count = 0
    label_record_count = 0
    for record in primary_candidates:
        if record.admin1_code not in target_counts:
            unscored_count += 1
            unscored_codes.add(record.admin1_code)
            continue
        target_counts[record.admin1_code] += 1
        label_record_count += 1

    late_counts = Counter(record.admin1_code for record in late_events)

    nodes: list[dict[str, Any]] = [
        {
            "id": "source:gdelt_v2_events",
            "type": "Source",
            "label": "GDELT 2.0 Events",
            "provenance": _node_provenance(),
        }
    ]
    for admin1_code in scoring_universe:
        nodes.append(
            {
                "id": _location_node_id(admin1_code),
                "type": "Location",
                "label": label_by_admin.get(admin1_code) or admin1_code,
                "external_ids": {"gdelt_adm1": admin1_code},
                "attributes": {"country_code": "FR", "admin1_code": admin1_code},
                "provenance": _node_provenance(),
            }
        )

    actor_nodes: dict[str, dict[str, Any]] = {}
    event_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for record in feature_events:
        event_node_id = _event_node_id(record.source_event_id)
        event_time = {"start": record.event_date.isoformat(), "granularity": "day"}
        event_nodes.append(
            {
                "id": event_node_id,
                "type": "Event",
                "label": f"France protest {record.event_date.isoformat()}",
                "external_ids": {"gdelt": record.source_event_id},
                "time": event_time,
                "provenance": _node_provenance(),
                "attributes": {
                    "source_event_id": record.source_event_id,
                    "source_available_at": _format_datetime_z(record.source_available_at),
                    "event_class": record.event_class,
                    "event_code": record.event_code,
                    "event_base_code": record.event_base_code,
                    "event_root_code": record.event_root_code,
                    "admin1_code": record.admin1_code,
                    "num_mentions": record.num_mentions,
                    "num_sources": record.num_sources,
                    "num_articles": record.num_articles,
                    "avg_tone": record.avg_tone,
                    "goldstein_scale": record.goldstein_scale,
                    "source_url": record.source_url,
                },
            }
        )
        edges.append(
            {
                "source": event_node_id,
                "target": _location_node_id(record.admin1_code),
                "type": "occurs_in",
                "time": event_time,
                "provenance": _node_provenance(),
                "attributes": {"source_event_id": record.source_event_id},
            }
        )
        edges.append(
            {
                "source": "source:gdelt_v2_events",
                "target": event_node_id,
                "type": "reports",
                "time": event_time,
                "provenance": _node_provenance(),
                "attributes": {"source_event_id": record.source_event_id},
            }
        )
        for actor_name, country_code, role in _actor_pairs(record):
            node_id = actor_id(actor_name, country_code)
            actor_nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": "Actor",
                    "label": actor_name,
                    "attributes": {"country_code": country_code},
                    "provenance": _node_provenance(),
                },
            )
            edges.append(
                {
                    "source": node_id,
                    "target": event_node_id,
                    "type": "participates_in",
                    "time": event_time,
                    "provenance": _node_provenance(),
                    "attributes": {"source_event_id": record.source_event_id, "role": role},
                }
            )

    nodes.extend(event_nodes)
    nodes.extend(actor_nodes[node_id] for node_id in sorted(actor_nodes))

    target_table: list[dict[str, Any]] = []
    for admin1_code in scoring_universe:
        count_value = target_counts[admin1_code]
        common_metadata = {
            "forecast_origin": origin_iso,
            "window_start": origin_date.isoformat(),
            "window_end_exclusive": window_end.isoformat(),
            "label_grace_days": LABEL_GRACE_DAYS,
            "admin1_code": admin1_code,
        }
        location_id = _location_node_id(admin1_code)
        target_table.append(
            {
                "target_id": f"france_protest:{origin_date.isoformat()}:{admin1_code}:count_next_7d",
                "name": "target_count_next_7d",
                "value": count_value,
                "split": split,
                "slice_id": origin_date.isoformat(),
                "node_ids": [location_id],
                "metadata": common_metadata,
            }
        )
        target_table.append(
            {
                "target_id": f"france_protest:{origin_date.isoformat()}:{admin1_code}:occurs_next_7d",
                "name": "target_occurs_next_7d",
                "value": count_value >= 1,
                "split": split,
                "slice_id": origin_date.isoformat(),
                "node_ids": [location_id],
                "metadata": common_metadata,
            }
        )

    return {
        "artifact_format": GRAPH_ARTIFACT_FORMAT,
        "probe_id": f"france_protest_as_of_{origin_date.isoformat()}",
        "schema_version": "0.2.0",
        "nodes": nodes,
        "edges": edges,
        "task_labels": [],
        "target_table": target_table,
        "metadata": {
            "domain": "france_protest",
            "forecast_origin": origin_iso,
            "window_days": WINDOW_DAYS,
            "label_grace_days": LABEL_GRACE_DAYS,
            "feature_record_count": len(feature_events),
            "label_record_count": label_record_count,
            "late_label_audit": {
                "late_event_count": len(late_events),
                "late_admin1_counts": dict(sorted(late_counts.items())),
            },
            "label_audit": {
                "unscored_admin1_event_count": unscored_count,
                "unscored_admin1_codes": sorted(unscored_codes),
            },
            "scoring_universe": {
                "country_code": "FR",
                "location_id_field": "ActionGeo_ADM1Code",
                "source": "all_admin1_codes_in_event_tape",
                "admin1_count": len(scoring_universe),
            },
            "source_name": "gdelt_v2_events",
        },
    }


def export_weekly_snapshots(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
    out_dir: Path,
) -> list[Path]:
    records = load_event_tape(tape_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for origin_date in _weekly_origins(origin_start, origin_end):
        payload = build_snapshot_payload(records=records, origin_date=origin_date)
        try:
            GraphArtifactV1.model_validate(payload)
        except Exception:
            invalid_path = out_dir / f"as_of_{origin_date.isoformat()}.invalid.json"
            invalid_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        out_path = out_dir / f"as_of_{origin_date.isoformat()}.json"
        temp_path = out_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(out_path)
        written.append(out_path)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-weekly")
    export.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    export.add_argument("--origin-start", default="2021-01-04")
    export.add_argument("--origin-end", default="2025-12-29")
    export.add_argument("--out", default="data/gdelt/snapshots/france_protest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export-weekly":
        try:
            export_weekly_snapshots(
                tape_path=Path(args.tape),
                origin_start=dt.date.fromisoformat(args.origin_start),
                origin_end=dt.date.fromisoformat(args.origin_end),
                out_dir=Path(args.out),
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
