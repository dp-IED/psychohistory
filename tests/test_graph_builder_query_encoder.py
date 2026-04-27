"""Tests for graph-builder query encoder and warehouse hint resolution."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pytest
import torch

from baselines.graph_builder_query_encoder import (
    ENTITY_HINT_KEYS,
    QueryEncoder,
    build_hint_index,
    encode_actor_state_query,
    normalize_hint,
    warehouse_context_from_manifest,
)
from schemas.graph_builder_probe import ActorStateQuery
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta


@pytest.fixture
def embedding_version() -> str:
    return "test_enc_v0"


def _l2_normalize_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.standard_normal((n, 128)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


def _base_manifest(
    embedding_version: str,
    mmap: np.ndarray,
    rows: list[NodeWarehouseRowMeta],
) -> NodeWarehouseManifest:
    return NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=embedding_version,
        mmap_path="/tmp/test_nodes.bin",
        row_count=len(rows),
        rows=rows,
    )


def test_build_hint_index_duplicate_hint_first_wins_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real tape can map one normalized string to several node_ids; we keep the first row."""
    rows = [
        NodeWarehouseRowMeta(
            node_id="a",
            extensions={ENTITY_HINT_KEYS: ["Same"]},
        ),
        NodeWarehouseRowMeta(
            node_id="b",
            extensions={ENTITY_HINT_KEYS: ["same"]},
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="baselines.graph_builder_query_encoder"):
        idx = build_hint_index(rows)
    assert idx == {"same": "a"}
    assert "keeping first" in caplog.text or "skipping" in caplog.text


def test_valid_hint_l2_norm(
    embedding_version: str,
    rng: np.random.Generator,
) -> None:
    mmap = _l2_normalize_rows(rng, 2)
    rows = [
        NodeWarehouseRowMeta(
            node_id="n0",
            extensions={ENTITY_HINT_KEYS: ["alias0"]},
        ),
        NodeWarehouseRowMeta(node_id="n1"),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    full_ctx = warehouse_context_from_manifest(manifest, mmap)
    slice_ctx = full_ctx
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["G"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=["alias0"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="p1",
        slice_ctx=slice_ctx,
        full_ctx=full_ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)
    assert torch.allclose(torch.linalg.vector_norm(out), torch.tensor(1.0), atol=1e-5)


def test_unk_hint(embedding_version: str, rng: np.random.Generator) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    rows = [NodeWarehouseRowMeta(node_id="only")]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["X"],
        actor_type=["group"],
        as_of=date(2020, 1, 1),
        entity_hints=["not_in_index"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="p_unk",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)
    assert torch.allclose(torch.linalg.vector_norm(out), torch.tensor(1.0), atol=1e-5)


def test_future_only_hint_falls_back_to_unk(embedding_version: str, rng: np.random.Generator) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    rows = [
        NodeWarehouseRowMeta(
            node_id="early",
            first_seen=date(2021, 6, 1),
            extensions={ENTITY_HINT_KEYS: ["early_actor"]},
        ),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=["early_actor"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="probe_oob",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)
    assert torch.isfinite(out).all()


def test_temporal_oob_forced_override_still_raises(
    embedding_version: str,
    rng: np.random.Generator,
) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    rows = [
        NodeWarehouseRowMeta(
            node_id="future",
            first_seen=date(2021, 6, 1),
            extensions={ENTITY_HINT_KEYS: ["future_actor"]},
        ),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=["future_actor"],
    )
    with pytest.raises(ValueError, match="temporal out-of-bounds"):
        encode_actor_state_query(
            actor_state=actor_state,
            probe_id="probe_oob_override",
            slice_ctx=ctx,
            full_ctx=ctx,
            encoder=encoder,
            hint_index_override={normalize_hint("future_actor"): "future"},
        )


def test_temporal_same_day_allowed(embedding_version: str, rng: np.random.Generator) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    d = date(2020, 1, 1)
    rows = [
        NodeWarehouseRowMeta(
            node_id="same_day",
            first_seen=d,
            extensions={ENTITY_HINT_KEYS: ["same_actor"]},
        ),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=d,
        entity_hints=["same_actor"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="probe_same",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)
    assert torch.isfinite(out).all()


def test_slice_oob_info_log(
    embedding_version: str,
    rng: np.random.Generator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mmap_full = _l2_normalize_rows(rng, 2)
    mmap_slice = mmap_full[:1].copy()
    rows_full = [
        NodeWarehouseRowMeta(
            node_id="in_slice",
            extensions={ENTITY_HINT_KEYS: ["slice_hint"]},
        ),
        NodeWarehouseRowMeta(
            node_id="full_only",
            extensions={ENTITY_HINT_KEYS: ["full_hint"]},
        ),
    ]
    manifest_full = _base_manifest(embedding_version, mmap_full, rows_full)
    full_ctx = warehouse_context_from_manifest(manifest_full, mmap_full)
    rows_slice = [rows_full[0]]
    manifest_slice = NodeWarehouseManifest(
        manifest_version="v0",
        embedding_version=embedding_version,
        mmap_path="/tmp/slice.bin",
        row_count=1,
        rows=rows_slice,
    )
    slice_ctx = warehouse_context_from_manifest(manifest_slice, mmap_slice)

    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=["full_hint"],
    )
    caplog.set_level(logging.INFO)
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="probe_cross",
        slice_ctx=slice_ctx,
        full_ctx=full_ctx,
        encoder=encoder,
    )
    assert "cross-slice" in caplog.text
    assert out.shape == (128,)


def test_manifest_integrity_error(embedding_version: str, rng: np.random.Generator) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    rows = [NodeWarehouseRowMeta(node_id="real", extensions={ENTITY_HINT_KEYS: ["ok"]})]
    manifest = _base_manifest(embedding_version, mmap, rows)
    full_ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=["phantom"],
    )
    with pytest.raises(ValueError, match="manifest integrity"):
        encode_actor_state_query(
            actor_state=actor_state,
            probe_id="probe_bad",
            slice_ctx=full_ctx,
            full_ctx=full_ctx,
            encoder=encoder,
            hint_index_override={normalize_hint("phantom"): "ghost_id"},
        )


def test_duplicate_hint_picks_in_window_country_match(
    embedding_version: str,
    rng: np.random.Generator,
) -> None:
    """Generic hint on EG then LY: first index row can be OOB; Libya probe must use LY row."""
    mmap = _l2_normalize_rows(rng, 2)
    rows = [
        NodeWarehouseRowMeta(
            node_id="ar_v0|EG|slot0",
            first_seen=date(2010, 12, 27),
            extensions={ENTITY_HINT_KEYS: ["military"]},
        ),
        NodeWarehouseRowMeta(
            node_id="ar_v0|LY|slot0",
            first_seen=date(2010, 12, 10),
            extensions={ENTITY_HINT_KEYS: ["military"]},
        ),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    full_ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["Libya"],
        actor_type=["gov"],
        as_of=date(2010, 12, 15),
        entity_hints=["military"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="p_dup",
        slice_ctx=full_ctx,
        full_ctx=full_ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)


def test_preferred_country_no_node_falls_back_to_unk(
    embedding_version: str,
    rng: np.random.Generator,
) -> None:
    """When only other countries are in the temporal window, do not attach their node."""
    mmap = _l2_normalize_rows(rng, 1)
    rows = [
        NodeWarehouseRowMeta(
            node_id="ar_v0|EG|slot0",
            first_seen=date(2010, 1, 1),
            extensions={ENTITY_HINT_KEYS: ["military"]},
        ),
    ]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["Libya"],
        actor_type=["gov"],
        as_of=date(2010, 12, 15),
        entity_hints=["military"],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="p_wrong_co",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    )
    assert out.shape == (128,)
    assert torch.isfinite(out).all()


def test_empty_entity_hints_unit_norm(embedding_version: str, rng: np.random.Generator) -> None:
    mmap = _l2_normalize_rows(rng, 1)
    rows = [NodeWarehouseRowMeta(node_id="a")]
    manifest = _base_manifest(embedding_version, mmap, rows)
    ctx = warehouse_context_from_manifest(manifest, mmap)
    encoder = QueryEncoder()
    actor_state = ActorStateQuery(
        geography=["F"],
        actor_type=["gov"],
        as_of=date(2020, 1, 1),
        entity_hints=[],
    )
    out = encode_actor_state_query(
        actor_state=actor_state,
        probe_id="p_empty",
        slice_ctx=ctx,
        full_ctx=ctx,
        encoder=encoder,
    )
    assert torch.allclose(torch.linalg.vector_norm(out), torch.tensor(1.0), atol=1e-5)
