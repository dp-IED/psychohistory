"""Mean-pool retrieved subgraph node features into a fixed-width bag vector."""

from __future__ import annotations

import torch
import torch.nn as nn

from schemas.graph_builder_retrieval import BUILDER_EMBEDDING_DIM, RetrievedGraphBatch


class BagEncoder(nn.Module):
    """Mean-pool valid nodes per batch row into ``[B, BUILDER_EMBEDDING_DIM]``.

    For each row ``b``, ``out[b] = mean(node_feat[b, node_mask[b]], dim=0)``.
    Rows where ``node_mask[b]`` is entirely false produce a zero vector (same
    as mean over an empty set, implemented via a clamped denominator).
    """

    def forward(self, batch: RetrievedGraphBatch) -> torch.Tensor:
        x = batch.node_feat
        if x.shape[-1] != BUILDER_EMBEDDING_DIM:
            raise ValueError(
                f"node_feat last dim must be BUILDER_EMBEDDING_DIM ({BUILDER_EMBEDDING_DIM}), "
                f"got {x.shape[-1]}",
            )

        mask = batch.node_mask.bool()
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        num = (x * w).sum(dim=1)
        denom = w.sum(dim=1).clamp(min=1.0)
        return num / denom
