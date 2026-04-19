"""Parameter-free ANN reranking into a padded retrieved graph batch."""

from __future__ import annotations

import numpy as np
import torch

from schemas.graph_builder_retrieval import (
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    RetrievedGraphBatch,
    validate_retrieved_batch_shapes,
)


def build_retrieved_graph_batch_from_ann(
    query: np.ndarray,
    ann_indices: np.ndarray,
    ann_scores: np.ndarray,
    mmap: np.ndarray,
    *,
    debug: bool = False,
) -> RetrievedGraphBatch:
    """Build a retrieved graph batch by reranking ANN candidates with mmap dots.

    `ann_scores` accepted for API compatibility with `brute_topk` output; node scoring recomputes from mmap dot products.
    """

    query_arr = np.asarray(query, dtype=np.float32)
    ann_indices_arr = np.asarray(ann_indices)
    if not np.issubdtype(ann_indices_arr.dtype, np.integer):
        raise ValueError(f"ann_indices must have an integer dtype, got {ann_indices_arr.dtype}")
    ann_scores_arr = np.asarray(ann_scores, dtype=np.float32)
    mmap_arr = np.asarray(mmap, dtype=np.float32)

    if query_arr.ndim != 2:
        raise ValueError(f"query must have shape (B, F), got {query_arr.shape}")
    if ann_indices_arr.ndim != 2:
        raise ValueError(f"ann_indices must have shape (B, 100), got {ann_indices_arr.shape}")
    if ann_scores_arr.ndim != 2:
        raise ValueError(f"ann_scores must have shape (B, 100), got {ann_scores_arr.shape}")
    if mmap_arr.ndim != 2:
        raise ValueError(f"mmap must have shape (N, F), got {mmap_arr.shape}")

    batch_size, feature_dim = query_arr.shape
    row_count, mmap_feature_dim = mmap_arr.shape
    expected_ann_shape = (batch_size, 100)

    if ann_indices_arr.shape != expected_ann_shape:
        raise ValueError(
            f"ann_indices must have shape {expected_ann_shape}, got {ann_indices_arr.shape}"
        )
    if ann_scores_arr.shape != expected_ann_shape:
        raise ValueError(
            f"ann_scores must have shape {expected_ann_shape}, got {ann_scores_arr.shape}"
        )
    if mmap_feature_dim != feature_dim:
        raise ValueError(
            f"mmap feature dim must match query feature dim {feature_dim}, got {mmap_feature_dim}"
        )

    node_feat = torch.zeros(
        (batch_size, MAX_RETRIEVED_NODES, feature_dim),
        dtype=torch.float32,
    )
    edge_index = torch.zeros(
        (batch_size, 2, MAX_RETRIEVED_EDGES),
        dtype=torch.long,
    )
    edge_weight = torch.zeros(
        (batch_size, MAX_RETRIEVED_EDGES),
        dtype=torch.float32,
    )
    node_mask = torch.zeros((batch_size, MAX_RETRIEVED_NODES), dtype=torch.bool)
    edge_mask = torch.zeros((batch_size, MAX_RETRIEVED_EDGES), dtype=torch.bool)

    for b in range(batch_size):
        valid_mask = (ann_indices_arr[b] >= 0) & (ann_indices_arr[b] < row_count)
        valid_indices = ann_indices_arr[b, valid_mask].astype(np.int64, copy=False)
        if valid_indices.size == 0:
            continue

        candidate_feat = mmap_arr[valid_indices]
        node_scores = (candidate_feat @ query_arr[b]).astype(np.float32, copy=False)
        node_order = np.argsort(-node_scores, kind="stable")
        take_nodes = min(MAX_RETRIEVED_NODES, int(node_order.size))

        top_feat = candidate_feat[node_order[:take_nodes]]
        node_feat[b, :take_nodes] = torch.from_numpy(top_feat)
        node_mask[b, :take_nodes] = True

        if take_nodes < 2:
            continue

        first_idx, second_idx = np.triu_indices(take_nodes, k=1)
        pair_scores = np.sum(top_feat[first_idx] * top_feat[second_idx], axis=1, dtype=np.float32)
        edge_order = np.argsort(-pair_scores, kind="stable")
        take_edges = min(MAX_RETRIEVED_EDGES, int(edge_order.size))
        top_edge_order = edge_order[:take_edges]

        edge_index[b, 0, :take_edges] = torch.from_numpy(
            first_idx[top_edge_order].astype(np.int64, copy=False)
        )
        edge_index[b, 1, :take_edges] = torch.from_numpy(
            second_idx[top_edge_order].astype(np.int64, copy=False)
        )
        edge_weight[b, :take_edges] = torch.from_numpy(pair_scores[top_edge_order])
        edge_mask[b, :take_edges] = True

    batch = RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_type=None,
        slot_id=None,
    )
    if debug:
        validate_retrieved_batch_shapes(batch, batch_size=batch_size)
    return batch


__all__ = ["build_retrieved_graph_batch_from_ann"]
