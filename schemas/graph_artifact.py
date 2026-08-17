"""Minimal graph artifact types used by Polymarket branch graphs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

GRAPH_ARTIFACT_FORMAT = "graph_artifact_v1"


class ArtifactTimeSpan(BaseModel):
    start: str | None = None
    end: str | None = None
    granularity: str | None = None


class ArtifactProvenance(BaseModel):
    sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArtifactNode(BaseModel):
    id: str
    type: str
    layer: str | None = None
    label: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    time: ArtifactTimeSpan = Field(default_factory=ArtifactTimeSpan)
    slice_ids: list[str] = Field(default_factory=list)
    train_eval_split: str | None = None
    provenance: ArtifactProvenance = Field(default_factory=ArtifactProvenance)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactEdge(BaseModel):
    source: str
    target: str
    type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: ArtifactProvenance = Field(default_factory=ArtifactProvenance)
    time: ArtifactTimeSpan = Field(default_factory=ArtifactTimeSpan)
    slice_ids: list[str] = Field(default_factory=list)
    train_eval_split: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactTaskLabel(BaseModel):
    task_id: str
    label: str
    node_ids: list[str] = Field(default_factory=list)
    edge_indices: list[int] = Field(default_factory=list)
    split: str | None = None


class ArtifactTargetRecord(BaseModel):
    target_id: str
    name: str
    value: float | int | str | bool | None = None
    split: str | None = None
    slice_id: str | None = None
    node_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphArtifactV1(BaseModel):
    artifact_format: str = GRAPH_ARTIFACT_FORMAT
    probe_id: str
    schema_version: str | None = None
    nodes: list[ArtifactNode]
    edges: list[ArtifactEdge]
    task_labels: list[ArtifactTaskLabel] = Field(default_factory=list)
    target_table: list[ArtifactTargetRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _references_existing_nodes(self) -> "GraphArtifactV1":
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"edge source not found in nodes: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"edge target not found in nodes: {edge.target}")
        return self
