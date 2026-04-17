"""Export weekly France protest forecast snapshots as graph artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Sequence

from baselines.forecast_scaffold_windows import (
    WEEKLY_ORIGIN_DEVELOPMENT_END,
    WEEKLY_ORIGIN_DEVELOPMENT_START,
    WEEKLY_ORIGIN_HOLDOUT_END,
    WEEKLY_ORIGIN_HOLDOUT_START,
)
from evals.graph_artifact_contract import GRAPH_ARTIFACT_FORMAT, GraphArtifactV1, validate_graph_artifact_point_in_time
from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.io_utils import write_json_atomic


UTC = dt.timezone.utc
LABEL_GRACE_DAYS = 14
WINDOW_DAYS = 7
EXCLUDED_REGIONAL_ADMIN1_CODES = frozenset({"FR", "FR00", "FR_UNKNOWN"})
SOURCE_LABELS = {
    "gdelt_v2_events": "GDELT 2.0 Events",
    "acled": "ACLED",
}
SourceIdentityMode = Literal["preserve", "collapse"]


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
    if WEEKLY_ORIGIN_DEVELOPMENT_START <= origin_date <= WEEKLY_ORIGIN_DEVELOPMENT_END:
        return "development"
    if WEEKLY_ORIGIN_HOLDOUT_START <= origin_date <= WEEKLY_ORIGIN_HOLDOUT_END:
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


def source_node_id(source_name: str) -> str:
    return f"source:{source_name}"


def node_provenance(source_name: str) -> dict[str, list[str]]:
    return {"sources": [source_name]}


def _multi_source_provenance(source_names: Sequence[str]) -> dict[str, list[str]]:
    return {"sources": sorted(set(source_names))}


def _filter_records_by_sources(
    records: list[EventTapeRecord],
    source_names: set[str] | None,
) -> list[EventTapeRecord]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


def _parse_source_names(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names:
        raise ValueError("--source-names must be 'all' or a comma-separated list")
    return names


def _best_location_labels(feature_events: list[EventTapeRecord]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in sorted(feature_events, key=_record_sort_key):
        if record.location_name and record.admin1_code not in labels:
            labels[record.admin1_code] = record.location_name
    return labels


def _location_universe(records: list[EventTapeRecord]) -> list[str]:
    """Return all country-qualified GDELT location codes needed by graph edges."""

    return sorted({record.admin1_code for record in records} | {"FR_UNKNOWN"})


def _regional_scoring_universe(records: list[EventTapeRecord]) -> list[str]:
    """Return clean regional admin1 codes, excluding country/unresolved buckets."""

    return sorted(
        {
            record.admin1_code
            for record in records
            if record.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES
        }
    )


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
    source_names: set[str] | None = None,
    source_identity_mode: SourceIdentityMode = "preserve",
    grounding_cache: Path | None = None,
    grounding_request_delay_s: float = 0.25,
    grounding_log: bool = True,
) -> dict[str, Any]:
    if source_identity_mode not in {"preserve", "collapse"}:
        raise ValueError(f"unknown source identity mode: {source_identity_mode}")
    origin_dt = dt.datetime.combine(origin_date, dt.time(), tzinfo=UTC)
    origin_iso = _format_datetime_z(origin_dt)
    window_end = origin_date + dt.timedelta(days=WINDOW_DAYS)
    grace_cutoff = origin_dt + dt.timedelta(days=WINDOW_DAYS + LABEL_GRACE_DAYS)
    split = _split_for_origin(origin_date)

    selected_records = _filter_records_by_sources(records, source_names)
    sorted_records = sorted(selected_records, key=_record_sort_key)
    selected_source_names = sorted(
        source_names if source_names is not None else {record.source_name for record in sorted_records}
    )
    feature_events = [
        record
        for record in sorted_records
        if record.source_available_at.astimezone(UTC) < origin_dt
        and record.event_date < origin_date
    ]
    location_universe = _location_universe(sorted_records)
    scoring_universe = _regional_scoring_universe(sorted_records)
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
    feature_source_counts = Counter(record.source_name for record in feature_events)
    label_source_counts = Counter(record.source_name for record in primary_candidates)
    excluded_primary_counts = Counter[str]()
    excluded_feature_counts = Counter(
        record.admin1_code
        for record in feature_events
        if record.admin1_code in EXCLUDED_REGIONAL_ADMIN1_CODES
    )
    unscored_codes: set[str] = set()
    unscored_count = 0
    label_record_count = 0
    for record in primary_candidates:
        if record.admin1_code in EXCLUDED_REGIONAL_ADMIN1_CODES:
            excluded_primary_counts[record.admin1_code] += 1
            continue
        if record.admin1_code not in target_counts:
            unscored_count += 1
            unscored_codes.add(record.admin1_code)
            continue
        target_counts[record.admin1_code] += 1
        label_record_count += 1

    late_counts = Counter(record.admin1_code for record in late_events)

    if source_identity_mode == "collapse":
        source_for_reports = "source:events"
        source_provenance = _multi_source_provenance(selected_source_names)
        nodes: list[dict[str, Any]] = [
            {
                "id": source_for_reports,
                "type": "Source",
                "label": "Events",
                "provenance": source_provenance,
                "attributes": {"source_name": "events", "source_names": selected_source_names},
            }
        ]
    else:
        source_for_reports = ""
        nodes = [
            {
                "id": source_node_id(source_name),
                "type": "Source",
                "label": SOURCE_LABELS.get(source_name, source_name),
                "provenance": node_provenance(source_name),
                "attributes": {"source_name": source_name},
            }
            for source_name in selected_source_names
        ]

    location_sources: dict[str, set[str]] = {}
    for record in sorted_records:
        location_sources.setdefault(record.admin1_code, set()).add(record.source_name)
    for admin1_code in location_universe:
        location_source_names = sorted(location_sources.get(admin1_code) or selected_source_names)
        nodes.append(
            {
                "id": _location_node_id(admin1_code),
                "type": "Location",
                "label": label_by_admin.get(admin1_code) or admin1_code,
                "external_ids": {"gdelt_adm1": admin1_code},
                "attributes": {"country_code": "FR", "admin1_code": admin1_code},
                "provenance": _multi_source_provenance(location_source_names),
            }
        )

    actor_nodes: dict[str, dict[str, Any]] = {}
    actor_node_sources: dict[str, set[str]] = {}
    event_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for record in feature_events:
        event_node_id = _event_node_id(record.source_event_id)
        event_time = {"start": record.event_date.isoformat(), "granularity": "day"}
        if source_identity_mode == "collapse":
            record_provenance = source_provenance
            report_source_id = source_for_reports
        else:
            record_provenance = node_provenance(record.source_name)
            report_source_id = source_node_id(record.source_name)
        external_ids = {record.source_name: record.source_event_id}
        if record.source_name == "gdelt_v2_events":
            external_ids["gdelt"] = record.source_event_id
        elif record.source_name == "acled":
            external_ids["acled"] = record.source_event_id
        event_nodes.append(
            {
                "id": event_node_id,
                "type": "Event",
                "label": f"France protest {record.event_date.isoformat()}",
                "external_ids": external_ids,
                "time": event_time,
                "provenance": record_provenance,
                "attributes": {
                    "source_name": record.source_name,
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
                "provenance": record_provenance,
                "attributes": {
                    "source_name": record.source_name,
                    "source_event_id": record.source_event_id,
                },
            }
        )
        edges.append(
            {
                "source": report_source_id,
                "target": event_node_id,
                "type": "reports",
                "time": event_time,
                "provenance": record_provenance,
                "attributes": {
                    "source_name": record.source_name,
                    "source_event_id": record.source_event_id,
                },
            }
        )
        for actor_name, country_code, role in _actor_pairs(record):
            node_id = actor_id(actor_name, country_code)
            actor_node_sources.setdefault(node_id, set()).add(record.source_name)
            actor_nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": "Actor",
                    "label": actor_name,
                    "attributes": {"country_code": country_code},
                    "provenance": node_provenance(record.source_name),
                },
            )
            actor_nodes[node_id]["provenance"] = _multi_source_provenance(actor_node_sources[node_id])
            edges.append(
                {
                    "source": node_id,
                    "target": event_node_id,
                    "type": "participates_in",
                    "time": event_time,
                    "provenance": record_provenance,
                    "attributes": {
                        "source_name": record.source_name,
                        "source_event_id": record.source_event_id,
                        "role": role,
                    },
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

    payload = {
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
                "excluded_regional_admin1_event_count": sum(excluded_primary_counts.values()),
                "excluded_regional_admin1_counts": dict(sorted(excluded_primary_counts.items())),
            },
            "feature_audit": {
                "excluded_regional_admin1_event_count": sum(excluded_feature_counts.values()),
                "excluded_regional_admin1_counts": dict(sorted(excluded_feature_counts.items())),
            },
            "scoring_universe": {
                "country_code": "FR",
                "location_id_field": "ActionGeo_ADM1Code",
                "source": "regional_admin1_codes_excluding_country_and_unresolved",
                "admin1_count": len(scoring_universe),
                "excluded_admin1_codes": sorted(EXCLUDED_REGIONAL_ADMIN1_CODES),
            },
            "graph_location_universe": {
                "source": "all_admin1_codes_in_event_tape",
                "admin1_count": len(location_universe),
            },
            "source_name": "gdelt_v2_events"
            if selected_source_names == ["gdelt_v2_events"]
            else "mixed",
            "source_names": selected_source_names,
            "source_identity_mode": source_identity_mode,
            "feature_source_counts": {
                source_name: feature_source_counts.get(source_name, 0)
                for source_name in selected_source_names
            },
            "label_source_counts": {
                source_name: label_source_counts.get(source_name, 0)
                for source_name in selected_source_names
            },
        },
    }
    if grounding_cache is not None:
        from evals.wikidata_grounding import apply_wikidata_grounding

        if grounding_log:
            print(
                (
                    f"[snapshot] origin={origin_date.isoformat()} "
                    f"nodes={len(nodes)} edges={len(edges)} "
                    f"→ wikidata grounding"
                ),
                file=sys.stderr,
                flush=True,
            )
        apply_wikidata_grounding(
            payload,
            cache_path=grounding_cache,
            request_delay_s=grounding_request_delay_s,
            log=grounding_log,
            origin=origin_date.isoformat(),
        )
    return payload


def export_weekly_snapshots(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
    out_dir: Path,
    source_names: set[str] | None = None,
    source_identity_mode: SourceIdentityMode = "preserve",
    snapshot_format: Literal["json", "json.gz"] = "json",
    progress: bool = False,
) -> list[Path]:
    records = load_event_tape(tape_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    origins = _weekly_origins(origin_start, origin_end)
    if snapshot_format not in {"json", "json.gz"}:
        raise ValueError(f"unknown snapshot format: {snapshot_format}")
    extension = ".json.gz" if snapshot_format == "json.gz" else ".json"
    for index, origin_date in enumerate(origins, start=1):
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin_date,
            source_names=source_names,
            source_identity_mode=source_identity_mode,
        )
        try:
            artifact = GraphArtifactV1.model_validate(payload)
            validate_graph_artifact_point_in_time(artifact)
        except Exception:
            invalid_path = out_dir / f"as_of_{origin_date.isoformat()}.invalid.json"
            invalid_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        out_path = out_dir / f"as_of_{origin_date.isoformat()}{extension}"
        write_json_atomic(out_path, payload)
        written.append(out_path)
        if progress:
            print(
                (
                    f"[snapshot] {index}/{len(origins)} origin={origin_date.isoformat()} "
                    f"nodes={len(payload['nodes'])} edges={len(payload['edges'])} "
                    f"targets={len(payload['target_table'])} wrote={out_path}"
                ),
                file=sys.stderr,
                flush=True,
            )
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-weekly")
    export.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    export.add_argument("--origin-start", default="2021-01-04")
    export.add_argument("--origin-end", default="2025-12-29")
    export.add_argument("--out", default="data/gdelt/snapshots/france_protest")
    export.add_argument(
        "--source-names",
        default="all",
        help="Comma-separated source names to include, or 'all'.",
    )
    export.add_argument(
        "--source-identity-mode",
        choices=["preserve", "collapse"],
        default="preserve",
    )
    export.add_argument("--snapshot-format", choices=["json", "json.gz"], default="json")
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
                source_names=_parse_source_names(args.source_names),
                source_identity_mode=args.source_identity_mode,
                snapshot_format=args.snapshot_format,
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
