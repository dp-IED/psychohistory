"""Tests for SSL positive-pair artifact build and load."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from baselines.graph_builder_positive_pairs import (
    PAIRS_ARRAY_BASENAME,
    POSITIVE_PAIR_VERSION,
    build_positive_pairs,
    load_positive_pairs,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta


def _manifest(
    *,
    rows: list[NodeWarehouseRowMeta],
    embedding_version: str = "emb_v1",
    mmap_path: str = "nodes.f32",
    as_of: date | None = None,
    window_days: int = 30,
) -> NodeWarehouseManifest:
    return NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=embedding_version,
        mmap_path=mmap_path,
        row_count=len(rows),
        rows=rows,
        window_days=window_days,
        as_of=as_of,
    )


def test_pit_window_excludes_row_before_window_start(tmp_path: Path) -> None:
    as_of = date(2026, 4, 18)
    window_days = 30
    window_start = as_of - timedelta(days=window_days - 1)
    rows = [
        NodeWarehouseRowMeta(
            node_id="a",
            admin1_code="FR-IDF",
            first_seen=window_start - timedelta(days=1),
        ),
        NodeWarehouseRowMeta(node_id="b", admin1_code="FR-IDF", first_seen=as_of),
    ]
    manifest = _manifest(rows=rows, as_of=as_of, window_days=window_days)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, _ = load_positive_pairs(meta_path, manifest)
    assert pairs.shape == (0, 2)


def test_basic_two_admin1_regions_four_nodes(tmp_path: Path) -> None:
    as_of = date(2026, 4, 18)
    window_days = 30
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="FR-IDF", first_seen=date(2026, 4, 10)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="FR-IDF", first_seen=date(2026, 4, 11)),
        NodeWarehouseRowMeta(node_id="c", admin1_code="FR-PACA", first_seen=date(2026, 4, 10)),
        NodeWarehouseRowMeta(node_id="d", admin1_code="FR-PACA", first_seen=date(2026, 4, 11)),
    ]
    manifest = _manifest(rows=rows, as_of=as_of, window_days=window_days)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")

    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, meta = load_positive_pairs(meta_path, manifest)

    assert set(map(tuple, pairs.tolist())) == {(0, 1), (2, 3)}
    assert meta["positive_pair_version"] == POSITIVE_PAIR_VERSION
    assert meta["pairs_path"] == PAIRS_ARRAY_BASENAME
    assert meta["pairs_path"] == Path(meta["pairs_path"]).name
    assert meta["as_of"] == "2026-04-18"


def test_metadata_as_of_null_when_manifest_has_no_as_of(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="R", first_seen=date(2026, 2, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="R", first_seen=date(2026, 2, 2)),
    ]
    manifest = _manifest(rows=rows, as_of=None)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["as_of"] is None


def test_boundary_fourteen_days_included(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="FR-IDF", first_seen=date(2026, 1, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="FR-IDF", first_seen=date(2026, 1, 15)),
    ]
    manifest = _manifest(rows=rows, as_of=None)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, _ = load_positive_pairs(meta_path, manifest)
    assert pairs.shape == (1, 2)
    assert tuple(pairs[0].tolist()) == (0, 1)


def test_boundary_fifteen_days_excluded(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="FR-IDF", first_seen=date(2026, 1, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="FR-IDF", first_seen=date(2026, 1, 16)),
    ]
    manifest = _manifest(rows=rows, as_of=None)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, _ = load_positive_pairs(meta_path, manifest)
    assert pairs.shape == (0, 2)


def test_different_admin1_close_dates_excluded(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="FR-IDF", first_seen=date(2026, 4, 10)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="FR-PACA", first_seen=date(2026, 4, 11)),
    ]
    manifest = _manifest(rows=rows, as_of=None)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, _ = load_positive_pairs(meta_path, manifest)
    assert pairs.shape == (0, 2)


def test_load_embedding_version_mismatch_raises(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="X", first_seen=date(2026, 1, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="X", first_seen=date(2026, 1, 2)),
    ]
    manifest = _manifest(rows=rows, embedding_version="manifest_version")
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["embedding_version"] = "metadata_version"
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"metadata=.*manifest="):
        load_positive_pairs(meta_path, manifest)


def test_round_trip_version_match_shape_dtype(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="R", first_seen=date(2026, 2, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="R", first_seen=date(2026, 2, 2)),
    ]
    manifest = _manifest(rows=rows)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, meta = load_positive_pairs(meta_path, manifest)
    p = int(meta["pair_count"])
    assert pairs.shape == (p, 2)
    assert pairs.dtype == np.int32


def test_ordering_unique_i_less_than_j(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="R", first_seen=date(2026, 2, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="R", first_seen=date(2026, 2, 2)),
    ]
    manifest = _manifest(rows=rows)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    pairs, _ = load_positive_pairs(meta_path, manifest)
    assert list(map(tuple, pairs.tolist())) == sorted(set(map(tuple, pairs.tolist())))
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_metadata_json_basename_only(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="R", first_seen=date(2026, 2, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="R", first_seen=date(2026, 2, 2)),
    ]
    manifest = _manifest(rows=rows)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["pairs_path"] == PAIRS_ARRAY_BASENAME
    assert Path(meta["pairs_path"]).name == meta["pairs_path"]


def test_build_requires_rows_matching_row_count(tmp_path: Path) -> None:
    rows = [NodeWarehouseRowMeta(node_id="only", admin1_code="X", first_seen=date(2026, 1, 1))]
    manifest = NodeWarehouseManifest.model_construct(
        manifest_version="v0",
        embedding_version="e",
        mmap_path="/x",
        row_count=2,
        rows=rows,
        window_days=30,
        embedding_dim=128,
    )
    with pytest.raises(ValueError, match=r"rows length"):
        build_positive_pairs(manifest, tmp_path / "m.f32", tmp_path / "out")


def test_load_rejects_non_basename_pairs_path(tmp_path: Path) -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a", admin1_code="R", first_seen=date(2026, 2, 1)),
        NodeWarehouseRowMeta(node_id="b", admin1_code="R", first_seen=date(2026, 2, 2)),
    ]
    manifest = _manifest(rows=rows)
    mmap_file = tmp_path / "dummy.f32"
    mmap_file.write_bytes(b"")
    meta_path = build_positive_pairs(manifest, mmap_file, tmp_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["pairs_path"] = "subdir/pairs.npy"
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"basename"):
        load_positive_pairs(meta_path, manifest)
