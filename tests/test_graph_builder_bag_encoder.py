from __future__ import annotations

import logging

import pytest
import torch

from baselines.graph_builder_bag_encoder import BAG_ENCODER_DEFAULT_DIM, BagEncoder
from schemas.graph_builder_retrieval import (
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    RetrievedGraphBatch,
)


def _empty_batch(*, batch_size: int = 2, feature_dim: int = BAG_ENCODER_DEFAULT_DIM) -> RetrievedGraphBatch:
    return RetrievedGraphBatch(
        node_feat=torch.zeros(batch_size, MAX_RETRIEVED_NODES, feature_dim, dtype=torch.float32),
        edge_index=torch.zeros(batch_size, 2, MAX_RETRIEVED_EDGES, dtype=torch.long),
        edge_weight=torch.zeros(batch_size, MAX_RETRIEVED_EDGES, dtype=torch.float32),
        node_mask=torch.zeros(batch_size, MAX_RETRIEVED_NODES, dtype=torch.bool),
        edge_mask=torch.zeros(batch_size, MAX_RETRIEVED_EDGES, dtype=torch.bool),
    )


def _clone_batch(batch: RetrievedGraphBatch) -> RetrievedGraphBatch:
    return RetrievedGraphBatch(
        node_feat=batch.node_feat.clone(),
        edge_index=batch.edge_index.clone(),
        edge_weight=batch.edge_weight.clone(),
        node_mask=batch.node_mask.clone(),
        edge_mask=batch.edge_mask.clone(),
        node_type=batch.node_type.clone() if batch.node_type is not None else None,
        slot_id=batch.slot_id.clone() if batch.slot_id is not None else None,
    )


def _set_edge(batch: RetrievedGraphBatch, *, row: int, edge_slot: int, src: int, dst: int, weight: float) -> None:
    batch.edge_index[row, 0, edge_slot] = src
    batch.edge_index[row, 1, edge_slot] = dst
    batch.edge_weight[row, edge_slot] = weight
    batch.edge_mask[row, edge_slot] = True


def _sparse_graph_batch() -> RetrievedGraphBatch:
    batch = _empty_batch(batch_size=2)

    batch.node_mask[0, :3] = True
    batch.node_feat[0, 0, :3] = torch.tensor([1.0, 0.0, 0.0])
    batch.node_feat[0, 1, :3] = torch.tensor([0.0, 2.0, 0.0])
    batch.node_feat[0, 2, :3] = torch.tensor([0.0, 0.0, 3.0])
    _set_edge(batch, row=0, edge_slot=0, src=0, dst=1, weight=2.0)
    _set_edge(batch, row=0, edge_slot=1, src=1, dst=2, weight=1.0)

    batch.node_mask[1, :2] = True
    batch.node_feat[1, 0, :3] = torch.tensor([2.0, 1.0, 0.0])
    batch.node_feat[1, 1, :3] = torch.tensor([0.0, 1.0, 2.0])
    _set_edge(batch, row=1, edge_slot=0, src=0, dst=1, weight=3.0)

    return batch


def test_bag_encoder_output_shape_and_l2_norm() -> None:
    out = BagEncoder()(_sparse_graph_batch())
    assert out.shape == (2, BAG_ENCODER_DEFAULT_DIM)
    assert torch.allclose(
        torch.linalg.vector_norm(out, ord=2, dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_masked_out_nodes_do_not_affect_output() -> None:
    batch = _sparse_graph_batch()
    mutated = _clone_batch(batch)
    mutated.node_feat[0, 7, :3] = torch.tensor([999.0, -111.0, 77.0])
    mutated.edge_index[0, :, 5] = torch.tensor([1, 7])
    mutated.edge_weight[0, 5] = 1000.0
    mutated.edge_mask[0, 5] = False

    enc = BagEncoder()
    assert torch.allclose(enc(batch), enc(mutated), atol=1e-6)


def test_nodes_with_no_edges_fall_back_to_uniform_weight() -> None:
    batch = _empty_batch(batch_size=1)
    batch.node_mask[0, :3] = True
    batch.node_feat[0, 0, :3] = torch.tensor([1.0, 0.0, 0.0])
    batch.node_feat[0, 1, :3] = torch.tensor([0.0, 2.0, 0.0])
    batch.node_feat[0, 2, :3] = torch.tensor([0.0, 0.0, 3.0])

    out = BagEncoder()(batch)
    expected = torch.zeros(BAG_ENCODER_DEFAULT_DIM)
    expected[:3] = torch.tensor([1.0 / 3.0, 2.0 / 3.0, 1.0])
    expected = expected / torch.linalg.vector_norm(expected, ord=2)
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_message_passing_preserves_shape_and_norm() -> None:
    out = BagEncoder(message_passing=True)(_sparse_graph_batch())
    assert out.shape == (2, BAG_ENCODER_DEFAULT_DIM)
    assert torch.allclose(
        torch.linalg.vector_norm(out, ord=2, dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_message_passing_changes_node_features_for_connected_graph() -> None:
    batch = _sparse_graph_batch()
    active_mask = batch.node_mask[0]
    node_feat_row = batch.node_feat[0] * active_mask.to(dtype=torch.float32).unsqueeze(-1)
    src, dst, edge_weight = BagEncoder._valid_edges_for_row(
        edge_index_row=batch.edge_index[0],
        edge_weight_row=batch.edge_weight[0],
        edge_mask_row=batch.edge_mask[0],
        node_mask_row=active_mask,
    )

    updated = BagEncoder._message_pass_row(
        node_feat_row=node_feat_row,
        active_mask=active_mask,
        src=src,
        dst=dst,
        edge_weight=edge_weight,
    )
    assert not torch.allclose(updated[active_mask], node_feat_row[active_mask])


def test_near_zero_bag_returns_zero_vector_and_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    batch = _empty_batch(batch_size=2)
    batch.node_mask[:, :2] = True
    _set_edge(batch, row=0, edge_slot=0, src=0, dst=1, weight=1.0)
    _set_edge(batch, row=1, edge_slot=0, src=0, dst=1, weight=2.0)

    with caplog.at_level(logging.WARNING):
        out = BagEncoder()(batch)

    assert torch.allclose(out, torch.zeros_like(out))
    assert "near-zero output" in caplog.text


def test_constructor_embedding_dim_64_produces_64d_output() -> None:
    batch = _empty_batch(batch_size=2, feature_dim=64)
    batch.node_mask[:, :2] = True
    batch.node_feat[0, 0, 0] = 1.0
    batch.node_feat[0, 1, 1] = 2.0
    batch.node_feat[1, 0, 2] = 3.0
    batch.node_feat[1, 1, 3] = 4.0
    _set_edge(batch, row=0, edge_slot=0, src=0, dst=1, weight=1.0)
    _set_edge(batch, row=1, edge_slot=0, src=0, dst=1, weight=1.0)

    out = BagEncoder(embedding_dim=64)(batch)
    assert out.shape == (2, 64)


@pytest.mark.parametrize(
    ("node_count_delta", "edge_count_delta"),
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_wrong_shape_contract_raises_value_error(node_count_delta: int, edge_count_delta: int) -> None:
    batch_size = 1
    feature_dim = BAG_ENCODER_DEFAULT_DIM
    bad = RetrievedGraphBatch(
        node_feat=torch.zeros(
            batch_size,
            MAX_RETRIEVED_NODES + node_count_delta,
            feature_dim,
            dtype=torch.float32,
        ),
        edge_index=torch.zeros(
            batch_size,
            2,
            MAX_RETRIEVED_EDGES + edge_count_delta,
            dtype=torch.long,
        ),
        edge_weight=torch.zeros(
            batch_size,
            MAX_RETRIEVED_EDGES + edge_count_delta,
            dtype=torch.float32,
        ),
        node_mask=torch.zeros(
            batch_size,
            MAX_RETRIEVED_NODES + node_count_delta,
            dtype=torch.bool,
        ),
        edge_mask=torch.zeros(
            batch_size,
            MAX_RETRIEVED_EDGES + edge_count_delta,
            dtype=torch.bool,
        ),
    )

    with pytest.raises(ValueError):
        BagEncoder()(bad)


def test_batch_rows_are_independent() -> None:
    baseline = _sparse_graph_batch()
    modified = _clone_batch(baseline)
    modified.node_feat[1, 0, :3] = torch.tensor([100.0, 0.0, 0.0])
    modified.node_feat[1, 1, :3] = torch.tensor([0.0, 0.0, 100.0])

    enc = BagEncoder()
    out_a = enc(baseline)
    out_b = enc(modified)

    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert not torch.allclose(out_a[1], out_b[1])


def test_bag_encoder_rejects_feature_width_mismatch() -> None:
    batch = _empty_batch(batch_size=1, feature_dim=32)
    with pytest.raises(ValueError, match="embedding_dim"):
        BagEncoder()(batch)
