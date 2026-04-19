"""Query encoder for mmap-backed node warehouse probes (128-d unit sphere)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from schemas.graph_builder_probe import ActorStateQuery
from schemas.graph_builder_warehouse import (
    NODE_WAREHOUSE_EMBEDDING_DIM_V1,
    NodeWarehouseManifest,
    NodeWarehouseRowMeta,
)

ENTITY_HINT_KEYS = "entity_hint_keys"

_GEO_BUCKETS = 32
_ACTOR_BUCKETS = 32
_FLAG_BUCKETS = 32
_HASH_FEATURE_DIM = _GEO_BUCKETS + _ACTOR_BUCKETS + _FLAG_BUCKETS
_LINEAR_IN_DIM = _HASH_FEATURE_DIM + NODE_WAREHOUSE_EMBEDDING_DIM_V1


def normalize_hint(s: str) -> str:
    return s.strip().casefold()


def build_hint_index(rows: Sequence[NodeWarehouseRowMeta]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        raw = row.extensions.get(ENTITY_HINT_KEYS) if row.extensions else None
        aliases: list[str] = raw if isinstance(raw, list) else []
        for k in aliases:
            if not isinstance(k, str):
                continue
            nk = normalize_hint(k)
            if nk in out and out[nk] != row.node_id:
                raise ValueError(
                    f"hint alias conflict for {nk!r}: node_ids {out[nk]!r} and {row.node_id!r}",
                )
            out[nk] = row.node_id
    return out


def build_id_to_row_index(rows: list[NodeWarehouseRowMeta], row_count: int) -> dict[str, int]:
    if len(rows) != row_count:
        raise ValueError(f"rows length ({len(rows)}) must equal row_count ({row_count})")
    return {rows[i].node_id: i for i in range(len(rows))}


def _stable_bucket(s: str, n_buckets: int, salt: bytes) -> int:
    h = hashlib.sha256(salt + s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_buckets


def _multi_hot_bucket(
    strings: Sequence[str],
    n_buckets: int,
    salt: bytes,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    v = torch.zeros(n_buckets, device=device, dtype=dtype)
    for s in strings:
        idx = _stable_bucket(s, n_buckets, salt)
        v[idx] += 1.0
    total = float(v.sum().item())
    if total > 0.0:
        v = v / total
    return v


@dataclass
class WarehouseMmapContext:
    manifest: NodeWarehouseManifest
    mmap: np.ndarray
    id_to_row: dict[str, int]
    row_meta_by_id: dict[str, NodeWarehouseRowMeta]


def warehouse_context_from_manifest(manifest: NodeWarehouseManifest, mmap: np.ndarray) -> WarehouseMmapContext:
    rows = manifest.rows
    if rows is None:
        raise ValueError("manifest.rows must be present for warehouse_context_from_manifest")
    if len(rows) != manifest.row_count:
        raise ValueError(
            f"manifest rows length ({len(rows)}) must equal row_count ({manifest.row_count})",
        )
    if mmap.shape != (manifest.row_count, NODE_WAREHOUSE_EMBEDDING_DIM_V1):
        raise ValueError(
            f"mmap shape {mmap.shape} must be ({manifest.row_count}, {NODE_WAREHOUSE_EMBEDDING_DIM_V1})",
        )
    id_to_row = build_id_to_row_index(rows, manifest.row_count)
    row_meta_by_id = {r.node_id: r for r in rows}
    return WarehouseMmapContext(
        manifest=manifest,
        mmap=mmap,
        id_to_row=id_to_row,
        row_meta_by_id=row_meta_by_id,
    )


class QueryEncoder(nn.Module):
    """Fuses hashed categorical features and hint embeddings into one 128-d unit vector."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.unk_embedding = nn.Parameter(torch.randn(NODE_WAREHOUSE_EMBEDDING_DIM_V1))
        torch.manual_seed(43)
        self.fuse = nn.Linear(_LINEAR_IN_DIM, NODE_WAREHOUSE_EMBEDDING_DIM_V1, bias=True)

    def encode_features(
        self,
        geography: Sequence[str],
        actor_type: Sequence[str],
        state_flags: Sequence[str],
        hint_embeddings: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        device = self.unk_embedding.device
        dtype = self.unk_embedding.dtype
        geo = _multi_hot_bucket(geography, _GEO_BUCKETS, b"geo", device=device, dtype=dtype)
        act = _multi_hot_bucket(actor_type, _ACTOR_BUCKETS, b"act", device=device, dtype=dtype)
        flg = _multi_hot_bucket(state_flags, _FLAG_BUCKETS, b"flg", device=device, dtype=dtype)
        hfeat = torch.cat([geo, act, flg], dim=0)
        if hint_embeddings:
            stacked = torch.stack([h.to(device=device, dtype=dtype) for h in hint_embeddings], dim=0)
            hint_mean = stacked.mean(dim=0)
        else:
            hint_mean = torch.zeros(
                NODE_WAREHOUSE_EMBEDDING_DIM_V1,
                device=device,
                dtype=dtype,
            )
        fused_in = torch.cat([hfeat, hint_mean], dim=0)
        out = self.fuse(fused_in)
        return F.normalize(out, dim=-1)

    def unk_unit(self) -> torch.Tensor:
        return F.normalize(self.unk_embedding, dim=-1)


def encode_actor_state_query(
    *,
    actor_state: ActorStateQuery,
    probe_id: str,
    slice_ctx: WarehouseMmapContext,
    full_ctx: WarehouseMmapContext,
    encoder: QueryEncoder,
    hint_index_override: dict[str, str] | None = None,
) -> torch.Tensor:
    """Encode an actor-state probe into the warehouse embedding space.

    hint_index_override is a test-only hook: when set, it replaces the hint index
    built from full_ctx.manifest.rows (used to simulate manifest integrity cases).

    After a hint resolves to a row, if that row has ``first_seen`` set and
    ``first_seen > actor_state.as_of``, encoding raises (evidence after the probe
    date). Same calendar day is allowed (``first_seen == as_of``).
    """
    if hint_index_override is not None:
        hint_index: dict[str, str] = dict(hint_index_override)
    else:
        hint_index = build_hint_index(full_ctx.manifest.rows or [])

    device = encoder.unk_embedding.device
    dtype = encoder.unk_embedding.dtype
    hint_vecs: list[torch.Tensor] = []
    log = logging.getLogger(__name__)

    for hint in actor_state.entity_hints:
        key = normalize_hint(hint)
        if key not in hint_index:
            hint_vecs.append(encoder.unk_unit())
            continue
        node_id = hint_index[key]
        meta = full_ctx.row_meta_by_id.get(node_id)
        if meta is not None and meta.first_seen is not None and meta.first_seen > actor_state.as_of:
            raise ValueError(
                "temporal out-of-bounds: entity hint evidence is after probe as_of "
                f"(probe_id={probe_id!r}, hint={hint!r}, node_id={node_id!r}, "
                f"first_seen={meta.first_seen.isoformat()}, as_of={actor_state.as_of.isoformat()})",
            )

        emb_np: np.ndarray | None = None
        if node_id in slice_ctx.id_to_row:
            idx = slice_ctx.id_to_row[node_id]
            emb_np = slice_ctx.mmap[idx]
        elif node_id in full_ctx.id_to_row:
            idx = full_ctx.id_to_row[node_id]
            log.info(
                "cross-slice node embedding lookup (hint resolved in full warehouse, not in slice mmap)",
                extra={
                    "probe_id": probe_id,
                    "hint": hint,
                    "node_id": node_id,
                },
            )
            emb_np = full_ctx.mmap[idx]
        else:
            raise ValueError(
                "manifest integrity: mmap row missing for hint-resolved node_id "
                f"(probe_id={probe_id!r}, hint={hint!r}, node_id={node_id!r})",
            )

        hint_vecs.append(torch.tensor(emb_np, device=device, dtype=dtype))

    return encoder.encode_features(
        actor_state.geography,
        actor_state.actor_type,
        actor_state.state_flags,
        hint_vecs,
    )


__all__ = [
    "ENTITY_HINT_KEYS",
    "QueryEncoder",
    "WarehouseMmapContext",
    "build_hint_index",
    "build_id_to_row_index",
    "encode_actor_state_query",
    "normalize_hint",
    "warehouse_context_from_manifest",
]
