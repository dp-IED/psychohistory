"""Tests for ANN reranking into a RetrievedGraphBatch."""

from __future__ import annotations

import numpy as np
import torch

from baselines.graph_builder_ann import brute_topk, l2_normalize_rows
from baselines.graph_builder_rerank import build_retrieved_graph_batch_from_ann
from schemas.graph_builder_retrieval import (
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    validate_retrieved_batch_shapes,
)


def _make_normalized(rows: np.ndarray) -> np.ndarray:
    return l2_normalize_rows(np.asarray(rows, dtype=np.float32))


def _batched_brute_topk(query: np.ndarray, corpus: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    batch_size = query.shape[0]
    indices = np.empty((batch_size, k), dtype=np.int64)
    scores = np.empty((batch_size, k), dtype=np.float32)
    for b in range(batch_size):
        row_indices, row_scores = brute_topk(query[b], corpus, k)
        indices[b] = row_indices
        scores[b] = row_scores
    return indices, scores


def test_rerank_shapes_b1_and_padding_semantics() -> None:
    feature_dim = 4
    corpus = _make_normalized(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
        ]
    )
    query = _make_normalized([[1.0, 0.0, 0.0, 0.0]])
    ann_indices, ann_scores = _batched_brute_topk(query, corpus, k=100)

    batch = build_retrieved_graph_batch_from_ann(
        query,
        ann_indices,
        ann_scores,
        corpus,
        debug=True,
    )

    assert batch.node_feat.shape == (1, MAX_RETRIEVED_NODES, feature_dim)
    assert batch.edge_index.shape == (1, 2, MAX_RETRIEVED_EDGES)
    assert batch.edge_weight.shape == (1, MAX_RETRIEVED_EDGES)
    assert batch.node_mask.shape == (1, MAX_RETRIEVED_NODES)
    assert batch.edge_mask.shape == (1, MAX_RETRIEVED_EDGES)
    assert batch.node_feat.dtype == torch.float32
    assert batch.edge_index.dtype == torch.long
    assert batch.edge_weight.dtype == torch.float32
    assert batch.node_mask.dtype == torch.bool
    assert batch.edge_mask.dtype == torch.bool
    assert batch.node_type is None
    assert batch.slot_id is None

    valid_node_count = int(batch.node_mask[0].sum().item())
    assert valid_node_count == 5

    padded_node_rows = ~batch.node_mask[0]
    assert torch.all(batch.node_feat[0, padded_node_rows] == 0)
    assert torch.all(torch.any(batch.node_feat[0, batch.node_mask[0]] != 0, dim=1))

    valid_edge_count = int(batch.edge_mask[0].sum().item())
    assert valid_edge_count == 10
    assert torch.all(batch.edge_weight[0, ~batch.edge_mask[0]] == 0)

    valid_edges = batch.edge_index[0, :, batch.edge_mask[0]]
    assert torch.all((valid_edges >= 0) & (valid_edges < MAX_RETRIEVED_NODES))
    assert torch.all(valid_edges[0] < valid_edges[1])
    assert torch.all(valid_edges[0] < valid_node_count)
    assert torch.all(valid_edges[1] < valid_node_count)

    validate_retrieved_batch_shapes(batch, batch_size=1)


def test_rerank_shapes_b4_edge_counts_and_edge_indices() -> None:
    feature_dim = 8
    corpus = _make_normalized(np.eye(60, feature_dim, dtype=np.float32))
    query = _make_normalized(corpus[:4].copy())
    ann_indices, ann_scores = _batched_brute_topk(query, corpus, k=100)

    batch = build_retrieved_graph_batch_from_ann(query, ann_indices, ann_scores, corpus)

    assert batch.node_feat.shape == (4, MAX_RETRIEVED_NODES, feature_dim)
    assert batch.edge_index.shape == (4, 2, MAX_RETRIEVED_EDGES)
    assert batch.edge_weight.shape == (4, MAX_RETRIEVED_EDGES)
    assert batch.node_mask.shape == (4, MAX_RETRIEVED_NODES)
    assert batch.edge_mask.shape == (4, MAX_RETRIEVED_EDGES)

    for b in range(4):
        k = int(batch.node_mask[b].sum().item())
        assert k == MAX_RETRIEVED_NODES
        assert not torch.any(~batch.node_mask[b])

        edge_count = int(batch.edge_mask[b].sum().item())
        assert edge_count == min(MAX_RETRIEVED_EDGES, k * (k - 1) // 2)

        valid_edges = batch.edge_index[b, :, batch.edge_mask[b]]
        assert torch.all((valid_edges >= 0) & (valid_edges < MAX_RETRIEVED_NODES))
        assert torch.all(valid_edges[0] < valid_edges[1])
        assert torch.all(valid_edges[0] < k)
        assert torch.all(valid_edges[1] < k)

    validate_retrieved_batch_shapes(batch, batch_size=4)
