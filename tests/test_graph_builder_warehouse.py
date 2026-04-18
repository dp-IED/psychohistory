"""Tests for graph-builder node warehouse manifest schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.graph_builder_warehouse import (
    NODE_WAREHOUSE_EMBEDDING_DIM_V1,
    NodeWarehouseManifest,
    NodeWarehouseRowMeta,
)


def test_manifest_row_count_zero_empty_rows_ok() -> None:
    m = NodeWarehouseManifest(
        manifest_version="v0",
        mmap_path="/tmp/nodes.bin",
        row_count=0,
        rows=[],
    )
    assert m.row_count == 0
    assert m.rows == []


def test_manifest_row_count_two_with_two_meta_rows_ok() -> None:
    rows = [
        NodeWarehouseRowMeta(node_id="a"),
        NodeWarehouseRowMeta(node_id="b", slice_id="s1"),
    ]
    m = NodeWarehouseManifest(
        manifest_version="v0",
        mmap_path="/data/embeddings.f32",
        row_count=2,
        rows=rows,
    )
    assert m.row_count == 2
    assert len(m.rows or []) == 2


def test_manifest_row_count_two_with_one_row_raises() -> None:
    with pytest.raises(ValidationError):
        NodeWarehouseManifest(
            manifest_version="v0",
            mmap_path="/data/embeddings.f32",
            row_count=2,
            rows=[NodeWarehouseRowMeta(node_id="only")],
        )


def test_embedding_dim_must_be_128() -> None:
    with pytest.raises(ValidationError):
        NodeWarehouseManifest(
            manifest_version="v0",
            embedding_dim=64,
            mmap_path="/x",
            row_count=0,
        )


def test_default_embedding_dim_is_128() -> None:
    m = NodeWarehouseManifest(manifest_version="v0", mmap_path="/x", row_count=0)
    assert m.embedding_dim == NODE_WAREHOUSE_EMBEDDING_DIM_V1
