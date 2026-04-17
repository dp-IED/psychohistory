"""Adversarial PIT checks for graph artifacts (`next_steps.md` §2.1 step A)."""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from evals.graph_artifact_contract import (
    GRAPH_ARTIFACT_FORMAT,
    GraphArtifactV1,
    assert_point_in_time_target_table,
    validate_graph_artifact_point_in_time,
)
from ingest.event_tape import EventTapeRecord
from ingest.snapshot_export import build_snapshot_payload


def _tape_record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
    source_name: str = "gdelt_v2_events",
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name=source_name,
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
        actor1_name=None,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url="https://example.test/story",
        raw={},
    )


def test_pit_rejects_target_with_future_only_observable_label() -> None:
    """Inject a resolution-style label observable only after forecast_origin — must fail closed."""
    origin = dt.date(2024, 6, 1)
    artifact = GraphArtifactV1.model_validate(
        {
            "artifact_format": GRAPH_ARTIFACT_FORMAT,
            "probe_id": "pit-adversarial",
            "nodes": [{"id": "n1", "type": "Event"}],
            "edges": [],
            "metadata": {"forecast_origin": f"{origin.isoformat()}T00:00:00Z"},
            "target_table": [
                {
                    "target_id": "t1",
                    "name": "resolution_leak_demo",
                    "value": 1.0,
                    "metadata": {
                        "observable_no_earlier_than": "2024-12-31",
                    },
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="PIT violation"):
        validate_graph_artifact_point_in_time(artifact)


def test_pit_accepts_target_observable_at_or_before_origin() -> None:
    origin = dt.date(2024, 6, 1)
    artifact = GraphArtifactV1.model_validate(
        {
            "artifact_format": GRAPH_ARTIFACT_FORMAT,
            "probe_id": "pit-ok",
            "nodes": [{"id": "n1", "type": "Event"}],
            "edges": [],
            "metadata": {"forecast_origin": f"{origin.isoformat()}T00:00:00Z"},
            "target_table": [
                {
                    "target_id": "t1",
                    "name": "ok",
                    "value": 1.0,
                    "metadata": {"observable_no_earlier_than": "2024-05-01"},
                }
            ],
        }
    )
    validate_graph_artifact_point_in_time(artifact)


def test_pit_rejects_snapshot_mutated_with_label_window_event_node() -> None:
    """A valid export payload mutated to smuggle a next-week event into the feature graph."""
    payload = build_snapshot_payload(
        records=[
            _tape_record("gdelt:ok", event_date="2021-01-01", source_available_at="2021-01-02T00:00:00Z"),
        ],
        origin_date=dt.date(2021, 1, 4),
    )
    mutant = copy.deepcopy(payload)
    mutant["nodes"].append(
        {
            "id": "event:injected-label-leak",
            "type": "Event",
            "label": "synthetic protest in label window",
            "external_ids": {"gdelt": "injected"},
            "time": {"start": "2021-01-05", "granularity": "day"},
            "provenance": {"sources": ["gdelt_v2_events"]},
            "attributes": {
                "source_name": "gdelt_v2_events",
                "source_event_id": "injected",
                "source_available_at": "2021-01-03T00:00:00Z",
            },
        }
    )
    mutant["edges"].extend(
        [
            {
                "source": "event:injected-label-leak",
                "target": "location:FR:FR11",
                "type": "occurs_in",
                "time": {"start": "2021-01-05", "granularity": "day"},
                "provenance": {"sources": ["gdelt_v2_events"]},
                "attributes": {},
            },
            {
                "source": "source:gdelt_v2_events",
                "target": "event:injected-label-leak",
                "type": "reports",
                "time": {"start": "2021-01-05", "granularity": "day"},
                "provenance": {"sources": ["gdelt_v2_events"]},
                "attributes": {},
            },
        ]
    )

    artifact = GraphArtifactV1.model_validate(mutant)
    with pytest.raises(ValueError, match="PIT violation"):
        validate_graph_artifact_point_in_time(artifact)


def test_assert_point_in_time_target_table_still_rejects_without_metadata_origin() -> None:
    """Explicit-origin API for artifacts that omit ``metadata.forecast_origin``."""
    origin = dt.date(2024, 6, 1)
    artifact = GraphArtifactV1.model_validate(
        {
            "artifact_format": GRAPH_ARTIFACT_FORMAT,
            "probe_id": "pit-adversarial",
            "nodes": [{"id": "n1", "type": "Event"}],
            "edges": [],
            "target_table": [
                {
                    "target_id": "t1",
                    "name": "resolution_leak_demo",
                    "value": 1.0,
                    "metadata": {"observable_no_earlier_than": "2024-12-31"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="PIT violation"):
        assert_point_in_time_target_table(artifact, forecast_origin=origin)
