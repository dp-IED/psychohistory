"""Binary float32 matrix I/O and JSON manifest persistence for node warehouses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from schemas.graph_builder_warehouse import NodeWarehouseManifest


def write_float32_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write a contiguous ``(N, 128)`` float32 matrix to ``path`` (raw bytes).

    Rejects degenerate rows (near-zero L2 norm) so NaNs/Infs from bad upstream
    normalisation cannot enter the mmap and silently corrupt ANN search.
    """
    assert matrix.ndim == 2
    arr = np.asarray(matrix, dtype=np.float32, order="C")
    assert arr.dtype == np.float32
    assert arr.shape[1] == 128
    if arr.shape[0] > 0:
        row_norms = np.linalg.norm(arr, axis=1)
        if not bool(np.all(np.isfinite(row_norms))) or float(np.min(row_norms)) < 1e-6:
            raise ValueError(
                "mmap write: every row must have finite L2 norm >= 1e-6 "
                "(reject zero-norm / NaN rows before they corrupt the ANN index)",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def read_float32_matrix(
    mmap_path: Path,
    *,
    row_count: int,
    embedding_dim: int = 128,
) -> np.memmap:
    """Memory-map a float32 matrix as read-only ``(row_count, embedding_dim)``."""
    return np.memmap(
        mmap_path,
        dtype=np.float32,
        mode="r",
        shape=(row_count, embedding_dim),
    )


def write_manifest(path: Path, manifest: NodeWarehouseManifest) -> None:
    """Serialize ``manifest`` to JSON (``model_dump(mode='json')``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "read_float32_matrix",
    "write_float32_matrix",
    "write_manifest",
]
