"""Tests for brute ANN top-k and ANN index metadata."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from baselines.graph_builder_ann import (
    AnnIndexMetadata,
    assert_manifest_matches_ann_metadata,
    brute_topk,
    read_ann_metadata,
    write_ann_metadata,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest


def test_brute_topk_query_is_corpus_row_highest() -> None:
    rng = np.random.default_rng(0)
    n, d = 5, 128
    corpus = rng.standard_normal((n, d), dtype=np.float64).astype(np.float32)
    query = corpus[2].copy()
    k = 3
    indices, scores = brute_topk(query, corpus, k)
    assert indices[0] == 2
    assert scores[0] >= scores[1] >= scores[2]
    assert scores[0] > 0.99


def test_ann_metadata_round_trip_and_manifest_assert() -> None:
    meta = AnnIndexMetadata(
        embedding_version="emb_ci_v0",
        row_count=5,
        embedding_dim=128,
    )
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version="emb_ci_v0",
        mmap_path="/tmp/nodes.bin",
        row_count=5,
    )
    assert_manifest_matches_ann_metadata(manifest, meta)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ann_meta.json"
        write_ann_metadata(path, meta)
        loaded = read_ann_metadata(path)
    assert loaded == meta


def test_assert_manifest_mismatch_embedding_version_raises() -> None:
    meta = AnnIndexMetadata(
        embedding_version="meta_v1",
        row_count=3,
    )
    manifest = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version="manifest_v2",
        mmap_path="/x",
        row_count=3,
    )
    with pytest.raises(ValueError, match="embedding_version"):
        assert_manifest_matches_ann_metadata(manifest, meta)
