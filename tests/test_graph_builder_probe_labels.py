from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from baselines.graph_builder_probe_labels import (
    FR_ADMIN1_ADJACENCY,
    LABEL_VERSION_V0,
    ProbeLabelRow,
    ProbeLabelSidecar,
    compute_probe_label_y,
    label_persistence_y,
    label_precursor_y,
    label_propagation_y,
    write_probe_labels_jsonl,
)
from ingest.event_tape import EventTapeRecord
from schemas.cameo_escalation_v0 import cameo_tier
from schemas.graph_builder_probe import AssumptionEmphasis


def _record(
    source_event_id: str,
    *,
    event_date: dt.date,
    admin1_code: str = "FR11",
    actor1_name: str | None = None,
    event_root_code: str = "14",
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=dt.datetime.combine(event_date, dt.time(12, 0), tzinfo=dt.timezone.utc),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name="Paris",
        latitude=None,
        longitude=None,
        event_class="protest",
        event_code="141",
        event_base_code="14",
        event_root_code=event_root_code,
        quad_class=3,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name=actor1_name,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def test_cameo_tier_ordering() -> None:
    assert cameo_tier("01") == 0
    assert cameo_tier("09") == 0
    assert cameo_tier("10") == 1
    assert cameo_tier("14") == 1
    assert cameo_tier("17") == 1
    assert cameo_tier("18") == 2
    assert cameo_tier("20") == 2
    assert cameo_tier("21") == 0
    assert cameo_tier("") == 0
    assert cameo_tier("Protests") == 0


def test_persistence_positive_when_week_sustains_baseline_mean() -> None:
    t = dt.date(2021, 2, 1)
    actor = "Alpha Unit"
    rows: list[EventTapeRecord] = []
    for i in range(14):
        d = t - dt.timedelta(days=14 - i)
        rows.append(_record(f"pre:{i}", event_date=d, actor1_name=actor))
    for j in range(7):
        d = t + dt.timedelta(days=j)
        rows.append(_record(f"hor:{j}", event_date=d, actor1_name=actor))

    y = label_persistence_y(
        rows,
        t,
        target_admin1="FR11",
        target_actor=actor,
        gate=AssumptionEmphasis.PERSISTENCE,
    )
    assert y is True


def test_persistence_unanswerable_when_zero_baseline() -> None:
    """No matching actor in pre-t window => baseline_mean 0 => None, not True."""
    t = dt.date(2021, 2, 1)
    actor = "NoSuchActor"
    rows: list[EventTapeRecord] = []
    for j in range(7):
        d = t + dt.timedelta(days=j)
        rows.append(_record(f"hor:{j}", event_date=d, actor1_name=actor))

    y = label_persistence_y(
        rows,
        t,
        target_admin1="FR11",
        target_actor=actor,
        gate=AssumptionEmphasis.PERSISTENCE,
    )
    assert y is None


def test_persistence_negative_when_one_day_drops_below_mean() -> None:
    t = dt.date(2021, 2, 1)
    actor = "Beta"
    rows: list[EventTapeRecord] = []
    for i in range(14):
        d = t - dt.timedelta(days=14 - i)
        rows.append(_record(f"pre:{i}", event_date=d, actor1_name=actor))
    for j in range(7):
        d = t + dt.timedelta(days=j)
        if j == 3:
            continue
        rows.append(_record(f"hor:{j}", event_date=d, actor1_name=actor))

    y = label_persistence_y(
        rows,
        t,
        target_admin1="FR11",
        target_actor=actor,
        gate=AssumptionEmphasis.PERSISTENCE,
    )
    assert y is False


def test_precursor_escalation_when_tier_exceeds_reference() -> None:
    t = dt.date(2021, 3, 1)
    rows = [
        _record("low", event_date=t + dt.timedelta(days=2), event_root_code="10"),
        _record("high", event_date=t + dt.timedelta(days=3), event_root_code="18"),
    ]
    y = label_precursor_y(
        rows,
        t,
        target_admin1="FR11",
        reference_event_root_code="10",
        gate=AssumptionEmphasis.PRECURSOR,
    )
    assert y is True


def test_precursor_negative_when_only_at_reference_tier() -> None:
    t = dt.date(2021, 3, 1)
    rows = [_record("same", event_date=t + dt.timedelta(days=1), event_root_code="14")]
    y = label_precursor_y(
        rows,
        t,
        target_admin1="FR11",
        reference_event_root_code="14",
        gate=AssumptionEmphasis.PRECURSOR,
    )
    assert y is False


def test_precursor_positive_tier1_when_reference_tier0() -> None:
    """France-style anchor ``01``: protest (tier 1) counts as escalation vs verbal."""
    t = dt.date(2021, 3, 1)
    rows = [_record("protest", event_date=t + dt.timedelta(days=1), event_root_code="14")]
    y = label_precursor_y(
        rows,
        t,
        target_admin1="FR11",
        reference_event_root_code="01",
        gate=AssumptionEmphasis.PRECURSOR,
    )
    assert y is True


def test_propagation_adjacent_same_class_after_first_origin_day() -> None:
    t = dt.date(2021, 4, 1)
    rows = [
        _record("origin", event_date=t, admin1_code="FR11"),
        _record("spread", event_date=t + dt.timedelta(days=1), admin1_code="FR24"),
    ]
    y = label_propagation_y(
        rows,
        t,
        target_admin1="FR11",
        gate=AssumptionEmphasis.PROPAGATION,
        adjacency=FR_ADMIN1_ADJACENCY,
    )
    assert y is True


def test_compute_probe_label_dispatches_sidecar() -> None:
    t = dt.date(2021, 5, 1)
    actor = "Gamma"
    rows: list[EventTapeRecord] = []
    for i in range(14):
        d = t - dt.timedelta(days=14 - i)
        rows.append(_record(f"p:{i}", event_date=d, actor1_name=actor))
    for j in range(7):
        d = t + dt.timedelta(days=j)
        rows.append(_record(f"h:{j}", event_date=d, actor1_name=actor))

    sidecar = ProbeLabelSidecar(probe_id="q1", target_admin1="FR11", actor_a=actor)
    assert (
        compute_probe_label_y(rows, t, AssumptionEmphasis.PERSISTENCE, sidecar) is True
    )


def test_write_probe_labels_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    row = ProbeLabelRow(
        probe_id="p1",
        label_version=LABEL_VERSION_V0,
        gate=AssumptionEmphasis.PRECURSOR,
        y=True,
        t0="2021-06-01",
        meta={"k": 1},
    )
    write_probe_labels_jsonl(path, [row])
    line = path.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["probe_id"] == "p1"
    assert data["y"] is True
    assert data["gate"] == "Precursor"


@pytest.mark.parametrize(
    "gate",
    [
        AssumptionEmphasis.PERSISTENCE,
        AssumptionEmphasis.PROPAGATION,
        AssumptionEmphasis.PRECURSOR,
        AssumptionEmphasis.SUPPRESSION,
        AssumptionEmphasis.COORDINATION,
    ],
)
def test_wrong_gate_returns_false_for_specific_rules(gate: AssumptionEmphasis) -> None:
    t = dt.date(2021, 7, 1)
    rows = [_record("x", event_date=t)]
    if gate != AssumptionEmphasis.PERSISTENCE:
        assert (
            label_persistence_y(
                rows, t, target_admin1="FR11", target_actor="x", gate=gate
            )
            is False
        )
    if gate != AssumptionEmphasis.PROPAGATION:
        assert (
            label_propagation_y(rows, t, target_admin1="FR11", gate=gate) is False
        )
    if gate != AssumptionEmphasis.PRECURSOR:
        assert (
            label_precursor_y(
                rows,
                t,
                target_admin1="FR11",
                reference_event_root_code="10",
                gate=gate,
            )
            is False
        )
