"""Tests for France v0 node matrix build and mmap round-trip."""

from __future__ import annotations

import datetime as dt
import zlib
from datetime import date, timedelta
from itertools import product

import numpy as np
import pytest

from baselines.node_warehouse_build_v0 import (
    NODE_WAREHOUSE_RECIPE_ID_V0,
    build_france_node_matrix_v0,
)
from baselines.node_warehouse_mmap import (
    read_float32_matrix,
    write_float32_matrix,
    write_manifest,
)
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


def _fra_record(
    *,
    source_event_id: str,
    event_date: date,
    actor1_name: str,
    event_root_code: str = "14",
) -> EventTapeRecord:
    stamp = dt.datetime(2026, 4, 15, 12, 0, tzinfo=UTC)
    return EventTapeRecord.model_validate(
        {
            "source_name": "gdelt_v2_events",
            "source_event_id": source_event_id,
            "event_date": event_date,
            "source_available_at": stamp,
            "retrieved_at": stamp,
            "country_code": "FRA",
            "admin1_code": "FR-IDF",
            "location_name": "Paris",
            "latitude": 48.85,
            "longitude": 2.35,
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
            "actor1_country_code": "FRA",
            "actor2_name": None,
            "actor2_country_code": None,
            "source_url": None,
            "raw": {},
        }
    )


def test_build_france_node_matrix_v0_shape_norms_mmap_roundtrip(tmp_path) -> None:
    name_a, name_b = _two_actor_names_distinct_slots()
    as_of = date(2026, 4, 18)
    window_days = 14
    recs = [
        _fra_record(source_event_id="e1", event_date=date(2026, 4, 10), actor1_name=name_a),
        _fra_record(source_event_id="e2", event_date=date(2026, 4, 11), actor1_name=name_a),
        _fra_record(source_event_id="e3", event_date=date(2026, 4, 12), actor1_name=name_b),
    ]
    matrix, metas = build_france_node_matrix_v0(recs, as_of=as_of, window_days=window_days)
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 128)
    assert len(metas) == 2
    for meta in metas:
        assert meta.admin1_code == "FR-IDF"
        assert meta.slice_id == f"as_of_{as_of.isoformat()}"
        assert meta.node_id.startswith("fr_v0|FR-IDF|slot")

    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    bin_path = tmp_path / "nodes.f32"
    write_float32_matrix(bin_path, matrix)
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version="test_mmap_v0",
        mmap_path=str(bin_path),
        row_count=int(matrix.shape[0]),
        recipe_id=NODE_WAREHOUSE_RECIPE_ID_V0,
        window_days=window_days,
        rows=metas,
    )
    man_path = tmp_path / "manifest.json"
    write_manifest(man_path, manifest)

    mm = read_float32_matrix(bin_path, row_count=matrix.shape[0], embedding_dim=128)
    assert np.allclose(np.asarray(mm, dtype=np.float32), matrix, rtol=1e-6, atol=1e-6)


def test_build_france_node_matrix_v0_rejects_bad_window() -> None:
    rec = _fra_record(source_event_id="e0", event_date=date(2026, 4, 10), actor1_name="x")
    with pytest.raises(ValueError):
        build_france_node_matrix_v0([rec], as_of=date(2026, 4, 18), window_days=0)


def _log_count_term(matrix: np.ndarray) -> float:
    assert matrix.shape[0] >= 1
    return float(matrix[0, 104])


def test_pit_window_excludes_day_before_start_even_with_anchor() -> None:
    """Window end inclusive on ``as_of``; start = as_of - (window_days-1). Day before start is out."""
    as_of = date(2026, 4, 18)
    window_days = 7
    start = date(2026, 4, 12)
    assert as_of - timedelta(days=window_days - 1) == start
    name = "anchor_actor"
    anchor_only = [
        _fra_record(source_event_id="a1", event_date=as_of, actor1_name=name),
    ]
    with_noise_before = anchor_only + [
        _fra_record(
            source_event_id="early",
            event_date=start - timedelta(days=1),
            actor1_name=name,
        ),
    ]
    m0, _ = build_france_node_matrix_v0(anchor_only, as_of=as_of, window_days=window_days)
    m1, _ = build_france_node_matrix_v0(with_noise_before, as_of=as_of, window_days=window_days)
    assert m0.shape == (1, 128) and m1.shape == (1, 128)
    assert _log_count_term(m0) == _log_count_term(m1)


def test_pit_window_includes_start_day() -> None:
    as_of = date(2026, 4, 18)
    window_days = 7
    start = date(2026, 4, 12)
    name = "anchor_actor"
    anchor_only = [
        _fra_record(source_event_id="a1", event_date=as_of, actor1_name=name),
    ]
    with_start_day = anchor_only + [
        _fra_record(source_event_id="on_start", event_date=start, actor1_name=name),
    ]
    m0, _ = build_france_node_matrix_v0(anchor_only, as_of=as_of, window_days=window_days)
    m1, _ = build_france_node_matrix_v0(with_start_day, as_of=as_of, window_days=window_days)
    assert _log_count_term(m1) > _log_count_term(m0)


def test_pit_window_excludes_after_as_of() -> None:
    as_of = date(2026, 4, 18)
    window_days = 7
    name = "anchor_actor"
    anchor_only = [
        _fra_record(source_event_id="a1", event_date=as_of, actor1_name=name),
    ]
    with_future = anchor_only + [
        _fra_record(
            source_event_id="future",
            event_date=as_of + timedelta(days=1),
            actor1_name=name,
        ),
    ]
    m0, _ = build_france_node_matrix_v0(anchor_only, as_of=as_of, window_days=window_days)
    m1, _ = build_france_node_matrix_v0(with_future, as_of=as_of, window_days=window_days)
    assert _log_count_term(m0) == _log_count_term(m1)


def test_pit_window_includes_as_of_day() -> None:
    as_of = date(2026, 4, 18)
    window_days = 7
    name = "anchor_actor"
    only_as_of = [
        _fra_record(source_event_id="a1", event_date=as_of, actor1_name=name),
    ]
    m0, _ = build_france_node_matrix_v0(only_as_of, as_of=as_of, window_days=window_days)
    assert m0.shape == (1, 128)
    assert _log_count_term(m0) > 0.0


def test_mmap_write_rejects_zero_row_matrix(tmp_path) -> None:
    z = np.zeros((1, 128), dtype=np.float32)
    with pytest.raises(ValueError, match=r"mmap write"):
        write_float32_matrix(tmp_path / "zero.f32", z)
