"""Typed manifest for mmap-backed node embedding tables (ANN / FAISS-ready v0).

Warehouse rows are stored as a contiguous float32 matrix on disk; metadata in
the manifest is optional for large corpora.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NODE_WAREHOUSE_EMBEDDING_DIM_V1: Literal[128] = 128


class EmbeddingSpaceSpec(BaseModel):
    """Contract for vectors stored in the node warehouse and used at query time.

    Rows are **128-dimensional** and **L2-normalised**. Query vectors from the
    query encoder live in the **same** space as node rows; there is **no**
    post-hoc linear map or projection between query encoder output and node
    embeddings. **Stage 1** joint contrastive pretraining aligns the query and
    node encoders so both emit comparable directions on the unit sphere.
    """

    model_config = ConfigDict(frozen=True)

    dim: Literal[128] = NODE_WAREHOUSE_EMBEDDING_DIM_V1


class NodeWarehouseRowMeta(BaseModel):
    """Per-node metadata aligned with one row of the mmap embedding matrix."""

    node_id: str
    slice_id: str | None = None
    source_refs: dict[str, str] | None = None
    admin1_code: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)


class NodeWarehouseManifest(BaseModel):
    """Versioned index for a float32 embedding matrix and optional row metadata."""

    manifest_version: str
    embedding_dim: int = Field(default=NODE_WAREHOUSE_EMBEDDING_DIM_V1, ge=1)
    mmap_path: str
    row_count: int = Field(ge=0)
    rows: list[NodeWarehouseRowMeta] | None = None

    @field_validator("embedding_dim")
    @classmethod
    def _embedding_dim_is_v1(cls, v: int) -> int:
        if v != NODE_WAREHOUSE_EMBEDDING_DIM_V1:
            raise ValueError(
                f"embedding_dim must be {NODE_WAREHOUSE_EMBEDDING_DIM_V1} for v1 manifests, got {v}",
            )
        return v

    @model_validator(mode="after")
    def _rows_len_matches_row_count_when_present(self) -> NodeWarehouseManifest:
        rows = self.rows
        if rows is not None and len(rows) > 0 and len(rows) != self.row_count:
            raise ValueError(
                f"rows length ({len(rows)}) must equal row_count ({self.row_count}) when rows is non-empty",
            )
        return self


__all__ = [
    "EmbeddingSpaceSpec",
    "NODE_WAREHOUSE_EMBEDDING_DIM_V1",
    "NodeWarehouseManifest",
    "NodeWarehouseRowMeta",
]
