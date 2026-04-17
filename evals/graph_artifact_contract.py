"""Versioned graph artifact contract for GNN candidate datasets."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field, model_validator


GRAPH_ARTIFACT_FORMAT = "graph_artifact_v1"
UTC = dt.timezone.utc


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


def assert_point_in_time_target_table(
    artifact: GraphArtifactV1,
    *,
    forecast_origin: dt.date,
) -> None:
    """
    Fail closed when a supervised target row claims the label was only observable **after**
    ``forecast_origin`` (simulates delayed resolutions / retroactive labels leaking into PIT training).

    Convention: ``ArtifactTargetRecord.metadata["observable_no_earlier_than"]`` is an ISO date
    string for the earliest wall-clock time the label could be known. If set, it must be
    ``<= forecast_origin`` for rows used as inputs at ``forecast_origin``.
    """

    origin_s = forecast_origin.isoformat()
    for record in artifact.target_table:
        when = record.metadata.get("observable_no_earlier_than")
        if when is None:
            continue
        if str(when) > origin_s:
            raise ValueError(
                f"PIT violation: target {record.target_id!r} observable_no_earlier_than={when!r} "
                f"is after forecast_origin {forecast_origin}"
            )


def _parse_metadata_forecast_origin(metadata: dict[str, Any]) -> dt.datetime | None:
    raw = metadata.get("forecast_origin")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _span_start_date(span: ArtifactTimeSpan) -> dt.date | None:
    if not span.start:
        return None
    head = span.start.strip()[:10]
    if len(head) != 10 or head[4] != "-" or head[7] != "-":
        return None
    try:
        return dt.date.fromisoformat(head)
    except ValueError:
        return None


def validate_graph_artifact_point_in_time(artifact: GraphArtifactV1) -> None:
    """
    When ``metadata["forecast_origin"]`` is set (ISO-8601, ``Z`` allowed), fail closed on
    obvious point-in-time leaks in the **input graph**: label-window events, reports that
    were not yet available at ``forecast_origin``, and target rows that declare a future
    observability bound (see :func:`assert_point_in_time_target_table`).
    """

    origin_dt = _parse_metadata_forecast_origin(artifact.metadata)
    if origin_dt is None:
        return
    origin_date = origin_dt.date()
    assert_point_in_time_target_table(artifact, forecast_origin=origin_date)

    for node in artifact.nodes:
        if node.type != "Event":
            continue
        event_day = _span_start_date(node.time)
        if event_day is not None and event_day >= origin_date:
            raise ValueError(
                f"PIT violation: Event {node.id!r} has time.start on or after forecast_origin "
                f"date {origin_date.isoformat()} (label-window or future event in feature graph)"
            )
        raw_avail = node.attributes.get("source_available_at")
        if isinstance(raw_avail, str) and raw_avail.strip():
            text = raw_avail.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            avail = dt.datetime.fromisoformat(text)
            if avail.tzinfo is None:
                avail = avail.replace(tzinfo=UTC)
            avail = avail.astimezone(UTC)
            if avail >= origin_dt:
                raise ValueError(
                    f"PIT violation: Event {node.id!r} has source_available_at {raw_avail!r} "
                    "on or after metadata.forecast_origin (post-t report availability leak)"
                )

    for edge in artifact.edges:
        if edge.type not in {"occurs_in", "reports", "participates_in"}:
            continue
        edge_day = _span_start_date(edge.time)
        if edge_day is None:
            continue
        if edge_day >= origin_date:
            raise ValueError(
                f"PIT violation: edge {edge.type!r} {edge.source!r}->{edge.target!r} has "
                f"time.start on or after forecast_origin date {origin_date.isoformat()}"
            )
