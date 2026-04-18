from __future__ import annotations

import pytest
import torch

from schemas.graph_builder_retrieval import (
    BUILDER_EMBEDDING_DIM,
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    RetrievedGraphBatch,
    validate_retrieved_batch_shapes,
)
from schemas.graph_builder_warehouse import NODE_WAREHOUSE_EMBEDDING_DIM_V1


def test_builder_dim_matches_warehouse_v1() -> None:
    assert BUILDER_EMBEDDING_DIM == NODE_WAREHOUSE_EMBEDDING_DIM_V1


def _minimal_batch(*, B: int = 2, F: int = BUILDER_EMBEDDING_DIM) -> RetrievedGraphBatch:
    node_feat = torch.zeros(B, MAX_RETRIEVED_NODES, F)
    node_feat[:, :3, :] = 1.0
    node_mask = torch.zeros(B, MAX_RETRIEVED_NODES, dtype=torch.bool)
    node_mask[:, :3] = True

    edge_index = torch.zeros(B, 2, MAX_RETRIEVED_EDGES, dtype=torch.long)
    edge_index[:, 0, :2] = torch.tensor([0, 1])
    edge_index[:, 1, :2] = torch.tensor([1, 2])
    edge_weight = torch.zeros(B, MAX_RETRIEVED_EDGES)
    edge_weight[:, :2] = 1.0
    edge_mask = torch.zeros(B, MAX_RETRIEVED_EDGES, dtype=torch.bool)
    edge_mask[:, :2] = True

    node_type = torch.zeros(B, MAX_RETRIEVED_NODES, dtype=torch.long)
    node_type[:, :3] = torch.tensor([1, 2, 3])
    slot_id = torch.zeros(B, MAX_RETRIEVED_NODES, dtype=torch.long)
    slot_id[:, :3] = 1

    return RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_type=node_type,
        slot_id=slot_id,
    )


def test_validate_retrieved_batch_shapes_accepts_minimal_padded_batch() -> None:
    batch = _minimal_batch(B=2, F=128)
    validate_retrieved_batch_shapes(batch, batch_size=2)


def test_validate_retrieved_batch_shapes_rejects_wrong_node_feat_shape() -> None:
    batch = _minimal_batch(B=2, F=128)
    bad = RetrievedGraphBatch(
        node_feat=torch.zeros(2, MAX_RETRIEVED_NODES - 1, 128),
        edge_index=batch.edge_index,
        edge_weight=batch.edge_weight,
        node_mask=batch.node_mask,
        edge_mask=batch.edge_mask,
        node_type=batch.node_type,
        slot_id=batch.slot_id,
    )
    with pytest.raises(ValueError, match="node_feat"):
        validate_retrieved_batch_shapes(bad, batch_size=2)
