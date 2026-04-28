"""Encode retrieved sparse subgraphs into weighted, normalized bag vectors."""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn

from schemas.graph_builder_retrieval import (
    BUILDER_EMBEDDING_DIM,
    RetrievedGraphBatch,
    validate_retrieved_batch_shapes,
)

logger = logging.getLogger(__name__)

BAG_ENCODER_DEFAULT_DIM: Final[int] = 128
_NEAR_ZERO_NORM_EPS = 1e-6


class BagEncoder(nn.Module):
    """Pool each retrieved graph row into a weighted, L2-normalized bag vector.

    Padded nodes are ignored. Sparse edges contribute incident weights to both
    endpoints; optional message passing performs a single parameter-free
    neighbor aggregation round before weighted pooling.
    """

    def __init__(
        self,
        embedding_dim: int = BAG_ENCODER_DEFAULT_DIM,
        message_passing: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.message_passing = message_passing

    @staticmethod
    def _valid_edges_for_row(
        *,
        edge_index_row: torch.Tensor,
        edge_weight_row: torch.Tensor,
        edge_mask_row: torch.Tensor,
        node_mask_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = int(node_mask_row.shape[0])
        valid = edge_mask_row.bool()
        if not bool(valid.any()):
            empty_idx = torch.zeros(0, dtype=torch.long, device=edge_index_row.device)
            empty_weight = torch.zeros(0, dtype=torch.float32, device=edge_weight_row.device)
            return empty_idx, empty_idx, empty_weight

        src_all = edge_index_row[0]
        dst_all = edge_index_row[1]
        in_bounds = (src_all >= 0) & (src_all < node_count) & (dst_all >= 0) & (dst_all < node_count)
        valid_idx = (valid & in_bounds).nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            empty_idx = torch.zeros(0, dtype=torch.long, device=edge_index_row.device)
            empty_weight = torch.zeros(0, dtype=torch.float32, device=edge_weight_row.device)
            return empty_idx, empty_idx, empty_weight

        src = src_all[valid_idx]
        dst = dst_all[valid_idx]
        active = node_mask_row[src] & node_mask_row[dst]
        if not bool(active.any()):
            empty_idx = torch.zeros(0, dtype=torch.long, device=edge_index_row.device)
            empty_weight = torch.zeros(0, dtype=torch.float32, device=edge_weight_row.device)
            return empty_idx, empty_idx, empty_weight

        src = src[active]
        dst = dst[active]
        weight = edge_weight_row[valid_idx][active].to(dtype=torch.float32)
        return src, dst, weight

    @staticmethod
    def _node_weights_for_row(
        *,
        active_mask: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.zeros(active_mask.shape[0], dtype=torch.float32, device=active_mask.device)
        if src.numel() > 0:
            weights.scatter_add_(0, src, edge_weight)
            weights.scatter_add_(0, dst, edge_weight)

        active_count = int(active_mask.sum().item())
        if active_count == 0:
            return weights

        total = weights[active_mask].sum()
        if float(total.item()) <= 0.0:
            weights[active_mask] = 1.0 / float(active_count)
            return weights

        weights[active_mask] = weights[active_mask] / total
        return weights

    @staticmethod
    def _message_pass_row(
        *,
        node_feat_row: torch.Tensor,
        active_mask: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        updated = node_feat_row.clone()
        if src.numel() == 0:
            updated[~active_mask] = 0.0
            return updated

        agg = torch.zeros_like(node_feat_row)
        degree = torch.zeros(node_feat_row.shape[0], dtype=torch.float32, device=node_feat_row.device)
        weight_col = edge_weight.unsqueeze(-1)

        agg.index_add_(0, src, node_feat_row[dst] * weight_col)
        agg.index_add_(0, dst, node_feat_row[src] * weight_col)
        degree.index_add_(0, src, edge_weight)
        degree.index_add_(0, dst, edge_weight)

        has_neighbors = active_mask & (degree > 0)
        if bool(has_neighbors.any()):
            updated[has_neighbors] = updated[has_neighbors] + (
                agg[has_neighbors] / degree[has_neighbors].unsqueeze(-1)
            )
        updated[~active_mask] = 0.0
        return updated

    def forward(
        self,
        batch: RetrievedGraphBatch,
        message_passing: bool | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch.node_feat.shape[0])
        validate_retrieved_batch_shapes(batch, batch_size=batch_size)

        x = batch.node_feat.to(dtype=torch.float32)
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"node_feat last dim must match embedding_dim ({self.embedding_dim}), "
                f"got {x.shape[-1]}",
            )

        use_message_passing = self.message_passing if message_passing is None else message_passing
        node_mask = batch.node_mask.bool()
        bag_rows: list[torch.Tensor] = []

        for row in range(batch_size):
            active_mask = node_mask[row]
            node_feat_row = x[row] * active_mask.to(dtype=torch.float32).unsqueeze(-1)
            src, dst, edge_weight = self._valid_edges_for_row(
                edge_index_row=batch.edge_index[row],
                edge_weight_row=batch.edge_weight[row],
                edge_mask_row=batch.edge_mask[row],
                node_mask_row=active_mask,
            )
            weights = self._node_weights_for_row(
                active_mask=active_mask,
                src=src,
                dst=dst,
                edge_weight=edge_weight,
            )
            if use_message_passing:
                node_feat_row = self._message_pass_row(
                    node_feat_row=node_feat_row,
                    active_mask=active_mask,
                    src=src,
                    dst=dst,
                    edge_weight=edge_weight,
                )
            bag_rows.append((node_feat_row * weights.unsqueeze(-1)).sum(dim=0))

        bag = torch.stack(bag_rows, dim=0)
        norms = torch.linalg.vector_norm(bag, ord=2, dim=1)
        out = torch.zeros_like(bag)
        ok = norms >= _NEAR_ZERO_NORM_EPS
        if bool(ok.any()):
            out[ok] = bag[ok] / norms[ok].unsqueeze(-1)
        near_zero_count = int((~ok).sum().item())
        if near_zero_count > 0:
            logger.warning(
                "BagEncoder produced near-zero output for %d batch rows; returning zero vectors",
                near_zero_count,
            )
        return out


__all__ = ["BAG_ENCODER_DEFAULT_DIM", "BagEncoder", "BUILDER_EMBEDDING_DIM"]
