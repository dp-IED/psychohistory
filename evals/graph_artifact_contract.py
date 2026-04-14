"""Versioned graph artifact contract for GNN candidate datasets."""

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


def _coerce_timespan(raw: Any) -> ArtifactTimeSpan:
    if isinstance(raw, dict):
        return ArtifactTimeSpan(
            start=str(raw.get("start") or raw.get("start_time") or raw.get("start_year"))
            if raw.get("start") or raw.get("start_time") or raw.get("start_year") is not None
            else None,
            end=str(raw.get("end") or raw.get("end_time") or raw.get("end_year"))
            if raw.get("end") or raw.get("end_time") or raw.get("end_year") is not None
            else None,
            granularity=raw.get("granularity"),
        )
    return ArtifactTimeSpan()


def _coerce_provenance(raw: Any) -> ArtifactProvenance:
    if not isinstance(raw, dict):
        return ArtifactProvenance()
    sources = raw.get("sources") or raw.get("source") or []
    if isinstance(sources, str):
        sources = [sources]
    notes = raw.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    return ArtifactProvenance(
        sources=[str(item) for item in sources if str(item).strip()],
        notes=[str(item) for item in notes if str(item).strip()],
    )


def normalize_graph_artifact(
    *,
    probe_id: str,
    schema_version: str | None,
    graph: dict[str, Any],
    default_split: str = "eval",
) -> GraphArtifactV1:
    nodes: list[ArtifactNode] = []
    for raw_node in graph.get("nodes") or []:
        if not isinstance(raw_node, dict):
            continue
        attrs = {
            key: value
            for key, value in raw_node.items()
            if key
            not in {
                "id",
                "type",
                "layer",
                "label",
                "external_ids",
                "time",
                "slice_ids",
                "train_eval_split",
                "provenance",
            }
        }
        nodes.append(
            ArtifactNode(
                id=str(raw_node.get("id")),
                type=str(raw_node.get("type")),
                layer=raw_node.get("layer"),
                label=raw_node.get("label"),
                external_ids={
                    str(k): str(v)
                    for k, v in (raw_node.get("external_ids") or {}).items()
                    if isinstance(raw_node.get("external_ids"), dict)
                },
                time=_coerce_timespan(raw_node.get("time")),
                slice_ids=[str(item) for item in raw_node.get("slice_ids", [])]
                if isinstance(raw_node.get("slice_ids"), list)
                else [],
                train_eval_split=str(raw_node.get("train_eval_split") or default_split),
                provenance=_coerce_provenance(raw_node.get("provenance")),
                attributes=attrs,
            )
        )

    edges: list[ArtifactEdge] = []
    for raw_edge in graph.get("edges") or []:
        if not isinstance(raw_edge, dict):
            continue
        attrs = {
            key: value
            for key, value in raw_edge.items()
            if key
            not in {
                "source",
                "target",
                "type",
                "confidence",
                "provenance",
                "time",
                "slice_ids",
                "train_eval_split",
                "task_ids",
            }
        }
        edges.append(
            ArtifactEdge(
                source=str(raw_edge.get("source")),
                target=str(raw_edge.get("target")),
                type=str(raw_edge.get("type")),
                confidence=float(raw_edge.get("confidence", 1.0)),
                provenance=_coerce_provenance(raw_edge.get("provenance")),
                time=_coerce_timespan(raw_edge.get("time")),
                slice_ids=[str(item) for item in raw_edge.get("slice_ids", [])]
                if isinstance(raw_edge.get("slice_ids"), list)
                else [],
                train_eval_split=str(raw_edge.get("train_eval_split") or default_split),
                task_ids=[str(item) for item in raw_edge.get("task_ids", [])]
                if isinstance(raw_edge.get("task_ids"), list)
                else [],
                attributes=attrs,
            )
        )

    return GraphArtifactV1(
        probe_id=probe_id,
        schema_version=schema_version,
        nodes=nodes,
        edges=edges,
        task_labels=[
            ArtifactTaskLabel.model_validate(item)
            for item in graph.get("task_labels", [])
            if isinstance(item, dict)
        ],
        target_table=[
            ArtifactTargetRecord.model_validate(item)
            for item in graph.get("target_table", [])
            if isinstance(item, dict)
        ],
        metadata=graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {},
    )
