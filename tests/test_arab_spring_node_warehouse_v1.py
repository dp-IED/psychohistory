"""Arab Spring v1 node warehouse: normalization, monthly grouping, manifest version lock."""

from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from baselines.node_warehouse_build_v0 import (
    NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V1,
    NODE_WAREHOUSE_RECIPE_ID_V1,
    _normalize_actor_name,
    build_arab_spring_node_matrix_v1,
    build_arab_spring_node_warehouse_v1,
)
from ingest.event_tape import EventTapeRecord
from schemas.graph_builder_warehouse import NodeWarehouseRowMeta

UTC = dt.timezone.utc


def _eg_record(
    *,
    source_event_id: str,
    event_date: date,
    actor1_name: str | None,
    admin1_code: str = "EGC1",
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
            "admin1_code": admin1_code,
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
            "actor2_name": None,
            "actor2_country_code": None,
            "source_url": None,
            "raw": {},
        }
    )


def test_normalize_actor_name_handles_none_punctuation_whitespace_and_length() -> None:
    assert _normalize_actor_name(None) == "unknown"
    assert _normalize_actor_name("") == "unknown"
    assert _normalize_actor_name("  Syrian--Arab Republic!!!  ") == "syrian arab republic"
    assert _normalize_actor_name("A" * 80) == "a" * 64
    assert _normalize_actor_name("  ...  ") == "unknown"


def test_v1_collapses_same_normalized_actor_to_one_monthly_node() -> None:
    as_of = date(2011, 2, 28)
    recs = [
        _eg_record(source_event_id="e1", event_date=date(2011, 2, 10), actor1_name="Free Syrian Army"),
        _eg_record(source_event_id="e2", event_date=date(2011, 2, 11), actor1_name=" free syrian army "),
        _eg_record(source_event_id="e3", event_date=date(2011, 2, 12), actor1_name="Free-Syrian Army!!"),
    ]
    matrix, metas = build_arab_spring_node_matrix_v1(recs, as_of=as_of, window_days=365)
    assert matrix.shape == (1, 128)
    assert len(metas) == 1
    assert metas[0].node_id == "ar_v1|free syrian army|EGC1|2011-02"
    assert metas[0].slice_id == "monthly_2011-02"
    assert metas[0].extensions["actor_name_normalized"] == "free syrian army"
    assert metas[0].extensions["time_bucket"] == "2011-02"


def test_v1_splits_same_actor_across_two_months() -> None:
    as_of = date(2011, 2, 28)
    recs = [
        _eg_record(source_event_id="jan", event_date=date(2011, 1, 20), actor1_name="Coalition"),
        _eg_record(source_event_id="feb", event_date=date(2011, 2, 20), actor1_name="Coalition"),
    ]
    matrix, metas = build_arab_spring_node_matrix_v1(recs, as_of=as_of, window_days=365)
    assert matrix.shape == (2, 128)
    assert [meta.node_id for meta in metas] == [
        "ar_v1|coalition|EGC1|2011-01",
        "ar_v1|coalition|EGC1|2011-02",
    ]


def test_v1_entity_hint_keys_keep_sorted_distinct_raw_names() -> None:
    as_of = date(2011, 2, 28)
    recs = [
        _eg_record(source_event_id="e1", event_date=date(2011, 2, 10), actor1_name="Free Syrian Army"),
        _eg_record(source_event_id="e2", event_date=date(2011, 2, 11), actor1_name="Free Syrian Army "),
        _eg_record(source_event_id="e3", event_date=date(2011, 2, 12), actor1_name="free syrian army"),
    ]
    _, metas = build_arab_spring_node_matrix_v1(recs, as_of=as_of, window_days=365)
    assert metas[0].extensions["entity_hint_keys"] == [
        "Free Syrian Army",
        "free syrian army",
    ]


def test_v1_pre_norm_guard_raises() -> None:
    as_of = date(2011, 2, 28)
    recs = [_eg_record(source_event_id="e1", event_date=date(2011, 2, 10), actor1_name="anchor_actor")]
    with patch("baselines.node_warehouse_build_v0._PRE_L2_NORM_EPS", 1.0e9):
        with pytest.raises(ValueError, match="node warehouse v1"):
            build_arab_spring_node_matrix_v1(recs, as_of=as_of, window_days=365)


def test_v1_manifest_embedding_version_and_dim(tmp_path: Path) -> None:
    as_of = date(2013, 12, 31)
    fake_matrix = np.ones((1, 128), dtype=np.float32)
    fake_meta = [
        NodeWarehouseRowMeta(
            node_id="ar_v1|actor|EGC1|2013-12",
            first_seen=as_of,
            slice_id="monthly_2013-12",
            admin1_code="EGC1",
        )
    ]

    with patch("baselines.node_warehouse_build_v0.query_records", return_value=[]), patch(
        "baselines.node_warehouse_build_v0.build_arab_spring_node_matrix_v1",
        return_value=(fake_matrix, fake_meta),
    ), patch("baselines.node_warehouse_build_v0.write_float32_matrix"), patch(
        "baselines.node_warehouse_build_v0.write_manifest"
    ) as write_manifest_mock:
        build_arab_spring_node_warehouse_v1(
            warehouse_path=tmp_path / "events.duckdb",
            out_mmap=tmp_path / "node_warehouse_v1.mmap",
            out_manifest=tmp_path / "node_warehouse_v1_manifest.json",
            show_progress=False,
        )

    manifest = write_manifest_mock.call_args.args[1]
    assert manifest.manifest_version == "v1"
    assert manifest.embedding_version == "ar_v1"
    assert manifest.embedding_version == NODE_WAREHOUSE_MMAP_EMBEDDING_VERSION_V1
    assert manifest.recipe_id == "gdelt_cameo_hist_actor1_admin1_monthly_v1"
    assert manifest.recipe_id == NODE_WAREHOUSE_RECIPE_ID_V1
    assert manifest.embedding_dim == 128
    assert manifest.as_of == as_of
