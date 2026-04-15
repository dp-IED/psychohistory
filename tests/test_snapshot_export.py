from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from evals.graph_artifact_contract import GraphArtifactV1
from ingest.event_tape import EventTapeRecord
from ingest.snapshot_export import build_snapshot_payload, export_weekly_snapshots


def _record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
    actor1_name: str | None = None,
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=dt.date.fromisoformat(event_date),
        source_available_at=dt.datetime.fromisoformat(source_available_at.replace("Z", "+00:00")),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name="Paris",
        latitude=48.8566,
        longitude=2.3522,
        event_class="protest",
        event_code="141",
        event_base_code="14",
        event_root_code="14",
        quad_class=3,
        goldstein_scale=-6.5,
        num_mentions=4,
        num_sources=2,
        num_articles=3,
        avg_tone=-1.2,
        actor1_name=actor1_name,
        actor1_country_code="FRA" if actor1_name else None,
        actor2_name=None,
        actor2_country_code=None,
        source_url="https://example.test/story",
        raw={},
    )


def _target_value(payload: dict, admin1_code: str, name: str) -> int | bool:
    for row in payload["target_table"]:
        if row["metadata"]["admin1_code"] == admin1_code and row["name"] == name:
            return row["value"]
    raise AssertionError(f"missing target {name} for {admin1_code}")


def test_reconstruction_excludes_future_available_records() -> None:
    payload = build_snapshot_payload(
        records=[
            _record(
                "gdelt:future",
                event_date="2021-01-01",
                source_available_at="2021-01-04T00:00:01Z",
            )
        ],
        origin_date=dt.date(2021, 1, 4),
    )

    assert [node for node in payload["nodes"] if node["type"] == "Event"] == []
    assert payload["metadata"]["feature_record_count"] == 0


def test_label_window_uses_event_date_not_availability_date() -> None:
    payload = build_snapshot_payload(
        records=[
            _record("gdelt:feature", event_date="2021-01-01", source_available_at="2021-01-02T00:00:00Z"),
            _record("gdelt:label", event_date="2021-01-05", source_available_at="2021-01-20T00:00:00Z"),
        ],
        origin_date=dt.date(2021, 1, 4),
    )

    assert _target_value(payload, "FR11", "target_count_next_7d") == 1
    assert _target_value(payload, "FR11", "target_occurs_next_7d") is True


def test_label_window_counts_admin1_codes_without_prior_feature_events() -> None:
    payload = build_snapshot_payload(
        records=[
            _record("gdelt:feature", event_date="2021-01-01", source_available_at="2021-01-02T00:00:00Z"),
            _record(
                "gdelt:new-admin1-label",
                event_date="2021-01-05",
                source_available_at="2021-01-06T00:00:00Z",
                admin1_code="FR22",
            ),
        ],
        origin_date=dt.date(2021, 1, 4),
    )

    assert _target_value(payload, "FR22", "target_count_next_7d") == 1
    assert _target_value(payload, "FR22", "target_occurs_next_7d") is True
    assert payload["metadata"]["label_audit"]["unscored_admin1_event_count"] == 0
    assert payload["metadata"]["scoring_universe"]["source"] == "all_admin1_codes_in_event_tape"


def test_late_labels_go_to_audit() -> None:
    payload = build_snapshot_payload(
        records=[
            _record("gdelt:feature", event_date="2021-01-01", source_available_at="2021-01-02T00:00:00Z"),
            _record("gdelt:late", event_date="2021-01-05", source_available_at="2021-01-26T00:00:00Z"),
        ],
        origin_date=dt.date(2021, 1, 4),
    )

    assert _target_value(payload, "FR11", "target_count_next_7d") == 0
    assert payload["metadata"]["late_label_audit"]["late_event_count"] == 1
    assert payload["metadata"]["late_label_audit"]["late_admin1_counts"] == {"FR11": 1}


def test_snapshot_validates_graph_artifact_contract(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    out_dir = tmp_path / "snapshots"
    record = _record(
        "gdelt:feature",
        event_date="2021-01-01",
        source_available_at="2021-01-02T00:00:00Z",
        actor1_name="Students",
    )
    tape_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    written = export_weekly_snapshots(
        tape_path=tape_path,
        origin_start=dt.date(2021, 1, 4),
        origin_end=dt.date(2021, 1, 4),
        out_dir=out_dir,
    )

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    GraphArtifactV1.model_validate(payload)
    event_nodes = [node for node in payload["nodes"] if node["type"] == "Event"]
    assert event_nodes[0]["time"]["start"] == "2021-01-01"
    assert event_nodes[0]["provenance"]["sources"] == ["gdelt_v2_events"]
    assert event_nodes[0]["external_ids"]["gdelt"] == "gdelt:feature"


def test_snapshot_excludes_event_available_at_origin() -> None:
    payload = build_snapshot_payload(
        records=[
            _record(
                "gdelt:at-origin",
                event_date="2021-01-01",
                source_available_at="2021-01-04T00:00:00Z",
            )
        ],
        origin_date=dt.date(2021, 1, 4),
    )

    assert [node for node in payload["nodes"] if node["type"] == "Event"] == []


def test_snapshot_export_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing event tape"):
        export_weekly_snapshots(
            tape_path=tmp_path / "missing.jsonl",
            origin_start=dt.date(2021, 1, 4),
            origin_end=dt.date(2021, 1, 4),
            out_dir=tmp_path / "snapshots",
        )


def test_snapshot_export_rejects_non_monday_origins(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    tape_path.write_text(
        _record("gdelt:feature", event_date="2021-01-01", source_available_at="2021-01-02T00:00:00Z").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="origin_start must be a Monday"):
        export_weekly_snapshots(
            tape_path=tape_path,
            origin_start=dt.date(2021, 1, 5),
            origin_end=dt.date(2021, 1, 11),
            out_dir=tmp_path / "snapshots",
        )
