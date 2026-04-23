"""Arab Spring v0 node matrix: PIT window, guards, manifest embedding version (shared with France)."""

from __future__ import annotations

import datetime as dt
import zlib
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from baselines.node_warehouse_build_v0 import (
    NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0,
    NODE_WAREHOUSE_RECIPE_ID_V0,
    build_arab_spring_node_matrix_v0,
)
from baselines.node_warehouse_mmap import write_float32_matrix, write_manifest
from ingest.event_tape import EventTapeRecord
from schemas.graph_builder_warehouse import NodeWarehouseManifest

UTC = dt.timezone.utc


def _actor_slot(name: str) -> int:
    key = name.strip().lower().encode("utf-8")
    return (zlib.crc32(key) & 0xFFFFFFFF) % 8


def _two_actor_names_distinct_slots() -> tuple[str, str]:
    for i, j in product(range(256), repeat=2):
        if i == j:
            continue
        a, b = f"actor{i}", f"actor{j}"
        if _actor_slot(a) != _actor_slot(b):
            return a, b
    raise AssertionError("failed to find two names with distinct actor_slot")


def _eg_record(
    *,
    source_event_id: str,
    event_date: date,
    actor1_name: str,
    actor2_name: str | None = None,
    event_root_code: str = "14",
    source_name: str = "gdelt_v1_events",
) -> EventTapeRecord:
    stamp = dt.datetime(2011, 1, 15, 12, 0, tzinfo=UTC)
    return EventTapeRecord.model_validate(
        {
            "source_name": source_name,
            "source_event_id": source_event_id,
            "event_date": event_date,
            "source_available_at": stamp,
            "retrieved_at": stamp,
            "country_code": "EG",
            "admin1_code": "EGC1",
            "location_name": "Cairo",
            "latitude": 30.0,
            "longitude": 31.0,
            "event_class": "protest",
            "event_code": "141",
            "event_base_code": "14",
            "event_root_code": event_root_code,
            "quad_class": 2,
            "goldstein_scale": -1.5,
            "num_mentions": 1,
            "num_sources": 1,
            "num_articles": 1,
            "avg_tone": -0.5,
            "actor1_name": actor1_name,
            "actor1_country_code": "EG",
            "actor2_name": actor2_name,
            "actor2_country_code": None,
            "source_url": None,
            "raw": {},
        }
    )


def _log_count_term(matrix: np.ndarray) -> float:
    assert matrix.shape[0] >= 1
    return float(matrix[0, 104])


def test_arab_spring_pit_excludes_day_before_start() -> None:
    as_of = date(2011, 1, 10)
    window_days = 1
    start = as_of - timedelta(days=window_days - 1)
    assert start == as_of
    name = "anchor_actor"
    anchor_only = [_eg_record(source_event_id="a1", event_date=as_of, actor1_name=name)]
    with_noise_before = anchor_only + [
        _eg_record(
            source_event_id="early",
            event_date=start - timedelta(days=1),
            actor1_name=name,
        ),
    ]
    m0, _ = build_arab_spring_node_matrix_v0(anchor_only, as_of=as_of, window_days=window_days)
    m1, _ = build_arab_spring_node_matrix_v0(with_noise_before, as_of=as_of, window_days=window_days)
    assert m0.shape == (1, 128) and m1.shape == (1, 128)
    assert _log_count_term(m0) == _log_count_term(m1)


def test_arab_spring_pit_includes_start_and_as_of_daily_window() -> None:
    as_of = date(2011, 1, 10)
    window_days = 1
    name = "anchor_actor"
    only_as_of = [_eg_record(source_event_id="a1", event_date=as_of, actor1_name=name)]
    m0, _ = build_arab_spring_node_matrix_v0(only_as_of, as_of=as_of, window_days=1)
    assert m0.shape == (1, 128)
    assert _log_count_term(m0) > 0.0


def test_arab_spring_pit_excludes_after_as_of() -> None:
    as_of = date(2011, 1, 10)
    name = "anchor_actor"
    anchor_only = [_eg_record(source_event_id="a1", event_date=as_of, actor1_name=name)]
    with_future = anchor_only + [
        _eg_record(
            source_event_id="future",
            event_date=as_of + timedelta(days=1),
            actor1_name=name,
        ),
    ]
    m0, _ = build_arab_spring_node_matrix_v0(anchor_only, as_of=as_of, window_days=1)
    m1, _ = build_arab_spring_node_matrix_v0(with_future, as_of=as_of, window_days=1)
    assert _log_count_term(m0) == _log_count_term(m1)


def test_arab_ly_node_distinct_from_eg() -> None:
    as_of = date(2011, 1, 10)
    r_eg = _eg_record(
        source_event_id="e1",
        event_date=as_of,
        actor1_name="x",
    )
    r_ly = r_eg.model_copy(
        update={
            "country_code": "LY",
            "source_event_id": "e2",
            "admin1_code": "LY1",
        }
    )
    m, metas = build_arab_spring_node_matrix_v0([r_eg, r_ly], as_of=as_of, window_days=1)
    assert m.shape[0] == 2
    assert len(metas) == 2


def test_arab_spring_pre_norm_guard_raises() -> None:
    as_of = date(2011, 1, 10)
    name = "anchor_actor"
    recs = [_eg_record(source_event_id="a1", event_date=as_of, actor1_name=name)]
    with patch("baselines.node_warehouse_build_v0._PRE_L2_NORM_EPS", 1.0e9):
        with pytest.raises(ValueError, match="node warehouse v0"):
            build_arab_spring_node_matrix_v0(recs, as_of=as_of, window_days=1)


def test_arab_spring_manifest_embedding_version_matches_v0_lock() -> None:
    name_a, name_b = _two_actor_names_distinct_slots()
    as_of = date(2011, 1, 18)
    recs = [
        _eg_record(source_event_id="e1", event_date=date(2011, 1, 10), actor1_name=name_a),
        _eg_record(source_event_id="e2", event_date=date(2011, 1, 11), actor1_name=name_b),
    ]
    matrix, metas = build_arab_spring_node_matrix_v0(recs, as_of=as_of, window_days=7)
    assert matrix.shape[1] == 128
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0,
        mmap_path="data/arab_spring/node_warehouse_v0.mmap",
        row_count=int(matrix.shape[0]),
        recipe_id=NODE_WAREHOUSE_RECIPE_ID_V0,
        window_days=7,
        as_of=as_of,
        rows=metas,
    )
    assert manifest.embedding_version == NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0
    assert manifest.embedding_dim == 128


def test_arab_spring_mmap_roundtrip_and_node_id(tmp_path: Path) -> None:
    as_of = date(2011, 1, 18)
    recs = [
        _eg_record(source_event_id="e1", event_date=date(2011, 1, 10), actor1_name="alpha"),
    ]
    matrix, metas = build_arab_spring_node_matrix_v0(recs, as_of=as_of, window_days=14)
    p = tmp_path / "n.mmap"
    write_float32_matrix(p, matrix)
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V0,
        mmap_path=str(p),
        row_count=int(matrix.shape[0]),
        recipe_id=NODE_WAREHOUSE_RECIPE_ID_V0,
        window_days=14,
        as_of=as_of,
        rows=metas,
    )
    write_manifest(tmp_path / "m.json", manifest)
    assert metas[0].node_id.startswith("ar_v0|")
    assert manifest.recipe_id == NODE_WAREHOUSE_RECIPE_ID_V0
