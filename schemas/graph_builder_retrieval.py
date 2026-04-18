"""Tensor batch contract for graph-builder retrieval feeding a bag encoder.

Stage 1 (pretraining): this batch is consumed for contrastive / self-supervised
objectives without downstream utility labels.
Stage 2 (weak supervision): the same batch is paired with labels for weak
utility training when those targets are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MAX_RETRIEVED_NODES = 50
MAX_RETRIEVED_EDGES = 200
BUILDER_EMBEDDING_DIM = 128


@dataclass
class RetrievedGraphBatch:
    """Padded batch of retrieved subgraphs for a bag encoder / GNN stack.

    **Feature width F:** ``node_feat[..., F]`` is the per-node feature dimension.
    When nodes are warehouse-aligned embedding vectors, set ``F ==
    BUILDER_EMBEDDING_DIM`` (128) so tensors match embedding tables. Raw or
    mixed numeric features may use a different ``F``; the encoder projection
    must then map to the model's internal width.

    **Masks:** ``node_mask`` and ``edge_mask`` are ``True`` or ``1`` for valid
    positions and ``False`` or ``0`` for padding. Padded ``edge_index`` entries
    should point only within valid nodes or use a sentinel (e.g. self-loop on
    pad index 0) as long as ``edge_mask`` is ``0`` for those edges so they are
    ignored in message passing.

    **Optional typing / slots:** ``node_type`` uses ``0`` for padding nodes.
    ``slot_id`` is reserved for tier or template slot indices aligned with the
    probe plan; padding positions should use ``0``.

    **Fields (tensor shapes):**
    ``node_feat`` — ``[B, MAX_RETRIEVED_NODES, F]`` float; ``edge_index`` —
    ``[B, 2, MAX_RETRIEVED_EDGES]`` long; ``edge_weight`` — ``[B,
    MAX_RETRIEVED_EDGES]`` float; ``node_mask`` / ``edge_mask`` — ``[B,
    MAX_RETRIEVED_NODES]`` / ``[B, MAX_RETRIEVED_EDGES]`` bool or float;
    optional ``node_type`` / ``slot_id`` — ``[B, MAX_RETRIEVED_NODES]`` long.
    """

    node_feat: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    node_type: torch.Tensor | None = None
    slot_id: torch.Tensor | None = None


def validate_retrieved_batch_shapes(batch: RetrievedGraphBatch, *, batch_size: int) -> None:
    """Raise ``ValueError`` if any required tensor has wrong rank or bounds vs ``batch_size``."""

    B = batch_size
    N, E = MAX_RETRIEVED_NODES, MAX_RETRIEVED_EDGES

    def _fail(name: str, detail: str) -> None:
        raise ValueError(f"RetrievedGraphBatch.{name}: {detail}")

    nf = batch.node_feat
    if nf.dim() != 3:
        _fail("node_feat", f"expected rank 3 [B, {N}, F], got rank {nf.dim()}")
    if nf.shape[0] != B or nf.shape[1] != N:
        _fail(
            "node_feat",
            f"expected shape ({B}, {N}, F), got {tuple(nf.shape)}",
        )
    if nf.shape[2] < 1:
        _fail("node_feat", f"feature dim F must be >= 1, got {nf.shape[2]}")

    ei = batch.edge_index
    if ei.dim() != 3:
        _fail("edge_index", f"expected rank 3 [B, 2, {E}], got rank {ei.dim()}")
    if ei.shape != (B, 2, E):
        _fail("edge_index", f"expected shape ({B}, 2, {E}), got {tuple(ei.shape)}")

    ew = batch.edge_weight
    if ew.dim() != 2:
        _fail("edge_weight", f"expected rank 2 [B, {E}], got rank {ew.dim()}")
    if ew.shape != (B, E):
        _fail("edge_weight", f"expected shape ({B}, {E}), got {tuple(ew.shape)}")

    nm = batch.node_mask
    if nm.dim() != 2:
        _fail("node_mask", f"expected rank 2 [B, {N}], got rank {nm.dim()}")
    if nm.shape != (B, N):
        _fail("node_mask", f"expected shape ({B}, {N}), got {tuple(nm.shape)}")

    em = batch.edge_mask
    if em.dim() != 2:
        _fail("edge_mask", f"expected rank 2 [B, {E}], got rank {em.dim()}")
    if em.shape != (B, E):
        _fail("edge_mask", f"expected shape ({B}, {E}), got {tuple(em.shape)}")

    if batch.node_type is not None:
        nt = batch.node_type
        if nt.dim() != 2 or nt.shape != (B, N):
            _fail(
                "node_type",
                f"expected shape ({B}, {N}) or None, got {tuple(nt.shape) if nt.dim() else 'scalar'}",
            )

    if batch.slot_id is not None:
        sid = batch.slot_id
        if sid.dim() != 2 or sid.shape != (B, N):
            _fail(
                "slot_id",
                f"expected shape ({B}, {N}) or None, got {tuple(sid.shape) if sid.dim() else 'scalar'}",
            )


__all__ = [
    "BUILDER_EMBEDDING_DIM",
    "MAX_RETRIEVED_EDGES",
    "MAX_RETRIEVED_NODES",
    "RetrievedGraphBatch",
    "validate_retrieved_batch_shapes",
]
