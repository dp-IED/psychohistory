"""Adversarial PIT checks for graph artifacts (`next_steps.md` §2.1 step A)."""

from __future__ import annotations

import datetime as dt

import pytest

from evals.graph_artifact_contract import (
    GRAPH_ARTIFACT_FORMAT,
    GraphArtifactV1,
    assert_point_in_time_target_table,
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
        assert_point_in_time_target_table(artifact, forecast_origin=origin)


def test_pit_accepts_target_observable_at_or_before_origin() -> None:
    origin = dt.date(2024, 6, 1)
    artifact = GraphArtifactV1.model_validate(
        {
            "artifact_format": GRAPH_ARTIFACT_FORMAT,
            "probe_id": "pit-ok",
            "nodes": [{"id": "n1", "type": "Event"}],
            "edges": [],
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
    assert_point_in_time_target_table(artifact, forecast_origin=origin)
