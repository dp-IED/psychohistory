"""Brute-force ANN top-k and ANN index metadata (Phase 2 tiny-N / CI, no FAISS)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from schemas.graph_builder_warehouse import NodeWarehouseManifest


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row of ``x``; output dtype float32 (``axis=1``)."""
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, np.float32(1e-12))
    return (arr / norms).astype(np.float32)


def brute_topk(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-``k`` neighbors by cosine similarity via L2-normalized dot product.

    ``query`` has shape ``(d,)``; ``corpus`` is ``(N, d)`` with the same ``d``
    (e.g. 128). Rows and the query are L2-normalized; scores are cosine
    similarity in ``[-1, 1]`` (typically near 1 for near-duplicates).

    Returns ``(indices, scores)`` both length ``k``, sorted by descending
    score. If ``N < k``, valid neighbors fill the prefix; remaining slots use
    index ``-1`` and score ``-inf`` (no neighbor). If ``N == 0``, all slots are
    padded that way.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    c = np.asarray(corpus, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != q.shape[0]:
        raise ValueError(
            f"corpus must be (N, d) with d={q.shape[0]}, got shape {c.shape}",
        )
    d = int(q.shape[0])
    qn = l2_normalize_rows(q.reshape(1, d)).reshape(d)
    cn = l2_normalize_rows(c)
    n_rows = int(cn.shape[0])
    indices = np.full(k, -1, dtype=np.int64)
    scores = np.full(k, -np.inf, dtype=np.float32)
    if n_rows == 0 or k == 0:
        return indices, scores
    sims = (cn @ qn).astype(np.float32)
    take = min(k, n_rows)
    order = np.argsort(-sims, kind="stable")[:take]
    indices[:take] = order.astype(np.int64, copy=False)
    scores[:take] = sims[order]
    return indices, scores


class AnnIndexMetadata(BaseModel):
    """JSON-serialized metadata for a tiny-N brute ANN sidecar."""

    model_config = ConfigDict(frozen=True)

    embedding_version: str
    row_count: int
    embedding_dim: int = 128
    index_kind: Literal["brute_v0"] = "brute_v0"


def write_ann_metadata(path: Path, meta: AnnIndexMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = meta.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_ann_metadata(path: Path) -> AnnIndexMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnnIndexMetadata.model_validate(data)


def assert_manifest_matches_ann_metadata(
    manifest: NodeWarehouseManifest,
    meta: AnnIndexMetadata,
) -> None:
    if manifest.embedding_version != meta.embedding_version:
        raise ValueError(
            "manifest.embedding_version does not match ann metadata: "
            f"{manifest.embedding_version!r} != {meta.embedding_version!r}",
        )
    if manifest.row_count != meta.row_count:
        raise ValueError(
            "manifest.row_count does not match ann metadata: "
            f"{manifest.row_count} != {meta.row_count}",
        )
    if manifest.embedding_dim != meta.embedding_dim:
        raise ValueError(
            "manifest.embedding_dim does not match ann metadata: "
            f"{manifest.embedding_dim} != {meta.embedding_dim}",
        )


__all__ = [
    "AnnIndexMetadata",
    "assert_manifest_matches_ann_metadata",
    "brute_topk",
    "l2_normalize_rows",
    "read_ann_metadata",
    "write_ann_metadata",
]
