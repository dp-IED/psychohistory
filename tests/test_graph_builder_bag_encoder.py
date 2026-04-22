from __future__ import annotations

import pytest
import torch

from baselines.graph_builder_bag_encoder import BagEncoder
from schemas.graph_builder_retrieval import (
    BUILDER_EMBEDDING_DIM,
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    RetrievedGraphBatch,
)


def _empty_edges(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = torch.zeros(B, 2, MAX_RETRIEVED_EDGES, dtype=torch.long)
    edge_weight = torch.zeros(B, MAX_RETRIEVED_EDGES)
    edge_mask = torch.zeros(B, MAX_RETRIEVED_EDGES, dtype=torch.bool)
    return edge_index, edge_weight, edge_mask


def test_bag_encoder_mean_pool_matches_manual() -> None:
    B, F = 2, BUILDER_EMBEDDING_DIM
    node_feat = torch.zeros(B, MAX_RETRIEVED_NODES, F)
    node_feat[0, 0, :] = 2.0
    node_feat[0, 1, :] = 4.0
    node_feat[1, 0, :] = 1.0
    node_feat[1, 1, :] = 3.0
    node_feat[1, 2, :] = 5.0

    node_mask = torch.zeros(B, MAX_RETRIEVED_NODES, dtype=torch.bool)
    node_mask[0, :2] = True
    node_mask[1, :3] = True

    ei, ew, em = _empty_edges(B)
    batch = RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=ei,
        edge_weight=ew,
        node_mask=node_mask,
        edge_mask=em,
    )

    enc = BagEncoder()
    out = enc(batch)
    assert out.shape == (B, BUILDER_EMBEDDING_DIM)
    assert torch.allclose(out[0], torch.full((F,), 3.0))
    assert torch.allclose(out[1], torch.full((F,), 3.0))


def test_bag_encoder_all_masked_row_is_zeros() -> None:
    B, F = 2, BUILDER_EMBEDDING_DIM
    node_feat = torch.randn(B, MAX_RETRIEVED_NODES, F)
    node_mask = torch.zeros(B, MAX_RETRIEVED_NODES, dtype=torch.bool)
    node_mask[0, 0] = True

    ei, ew, em = _empty_edges(B)
    batch = RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=ei,
        edge_weight=ew,
        node_mask=node_mask,
        edge_mask=em,
    )

    out = BagEncoder()(batch)
    assert torch.allclose(out[1], torch.zeros(F))


def test_bag_encoder_rejects_wrong_feature_dim() -> None:
    B = 1
    F = BUILDER_EMBEDDING_DIM + 7
    node_feat = torch.zeros(B, MAX_RETRIEVED_NODES, F)
    node_mask = torch.ones(B, MAX_RETRIEVED_NODES, dtype=torch.bool)
    ei, ew, em = _empty_edges(B)
    batch = RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=ei,
        edge_weight=ew,
        node_mask=node_mask,
        edge_mask=em,
    )
    with pytest.raises(ValueError, match="BUILDER_EMBEDDING_DIM"):
        BagEncoder()(batch)
