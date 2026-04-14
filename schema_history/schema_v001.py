"""
Psychohistory graph IR (Phase 1 autoresearch mutates this file only).

Edge-based truth is primary: use EdgeType.INSTANTIATES from an EVENTIVE source to a
CONJUNCTURAL or STRUCTURAL target for upward linkage. Node.instantiates / phase_of
are optional mirrors for convenience and must stay consistent when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


FEATURE_DIM: int = 32


class NodeType(str, Enum):
    AGENT = "Agent"
    BELIEF = "Belief"
    RESOURCE = "Resource"
    INSTITUTION = "Institution"
    EVENT = "Event"
    CLAIM = "Claim"
    CONSTRAINT = "Constraint"


class EdgeType(str, Enum):
    CAUSES = "causes"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    PRESUPPOSES = "presupposes"
    TRANSFORMS = "transforms"
    CONSTRAINS = "constrains"
    INSTANTIATES = "INSTANTIATES"
    PHASE_TRANSITION = "PHASE_TRANSITION"
    REPRODUCES = "REPRODUCES"
    RUPTURES = "RUPTURES"
    ACCELERATES = "ACCELERATES"
    DECELERATES = "DECELERATES"


class TemporalScale(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    CONJUNCTURAL = "CONJUNCTURAL"
    EVENTIVE = "EVENTIVE"


class TemporalPrecision(str, Enum):
    UNKNOWN = "unknown"
    CENTURY = "century"
    DECADE = "decade"
    YEAR = "year"
    DAY = "day"
    INSTANT = "instant"


@dataclass
class TemporalSpan:
    """JSON-stable span; use ISO-8601 strings where possible."""

    start: str | None = None
    end: str | None = None
    precision: TemporalPrecision = TemporalPrecision.UNKNOWN


@dataclass
class Node:
    id: str
    type: NodeType
    label: str
    features: list[float]
    confidence: float
    contention: float
    provenance: list[str]
    presupposes: list[str]
    chunk_refs: list[str]
    temporal_scale: TemporalScale
    temporal_span: TemporalSpan | None = None
    instantiates: list[str] = field(default_factory=list)
    phase_of: list[str] = field(default_factory=list)
    layer: str | None = None
    civilisation: str | None = None
    region: str | None = None
    legitimacy_proxy: float | None = None
    seshat_codes: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    sign: int
    weight: float
    conditioned_on: list[str]
    confidence: float


@dataclass
class EpistemicFrame:
    anchor: str
    centrality: dict[str, float]


@dataclass
class GraphBundle:
    """A named graph instance used by fixtures and evaluation."""

    probe_id: str
    parsimony_exception: bool
    nodes: list[Node]
    edges: list[Edge]
    max_nodes_parsimony: int | None = 30

    @property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}


def zero_features() -> list[float]:
    return [0.0] * FEATURE_DIM


def node_from_dict(d: dict[str, Any]) -> Node:
    ts = d.get("temporal_span")
    span = None
    if isinstance(ts, dict):
        prec_raw = ts.get("precision", "unknown")
        prec = (
            TemporalPrecision(prec_raw)
            if prec_raw in TemporalPrecision._value2member_map_
            else TemporalPrecision.UNKNOWN
        )
        span = TemporalSpan(
            start=ts.get("start"),
            end=ts.get("end"),
            precision=prec,
        )
    elif ts is None:
        span = None
    else:
        span = None

    feats = d.get("features")
    if feats is None:
        f = zero_features()
    else:
        f = [float(x) for x in feats]
        if len(f) != FEATURE_DIM:
            raise ValueError(f"features must have length {FEATURE_DIM}, got {len(f)}")

    return Node(
        id=d["id"],
        type=NodeType(d["type"]),
        label=d.get("label", ""),
        features=f,
        confidence=float(d.get("confidence", 0.5)),
        contention=float(d.get("contention", 0.0)),
        provenance=list(d.get("provenance", [])),
        presupposes=list(d.get("presupposes", [])),
        chunk_refs=list(d.get("chunk_refs", [])),
        temporal_scale=TemporalScale(d["temporal_scale"]),
        temporal_span=span,
        instantiates=list(d.get("instantiates", [])),
        phase_of=list(d.get("phase_of", [])),
        layer=d.get("layer"),
        civilisation=d.get("civilisation"),
        region=d.get("region"),
        legitimacy_proxy=(
            float(d["legitimacy_proxy"]) if d.get("legitimacy_proxy") is not None else None
        ),
        seshat_codes=list(d.get("seshat_codes", [])),
    )


def edge_from_dict(d: dict[str, Any]) -> Edge:
    return Edge(
        source=d["source"],
        target=d["target"],
        type=EdgeType(d["type"]),
        sign=int(d["sign"]),
        weight=float(d.get("weight", 1.0)),
        conditioned_on=list(d.get("conditioned_on", [])),
        confidence=float(d.get("confidence", 0.5)),
    )


def graph_from_dict(d: dict[str, Any]) -> GraphBundle:
    nodes = [node_from_dict(x) for x in d["nodes"]]
    edges = [edge_from_dict(x) for x in d["edges"]]
    return GraphBundle(
        probe_id=d["probe_id"],
        parsimony_exception=bool(d.get("parsimony_exception", False)),
        nodes=nodes,
        edges=edges,
        max_nodes_parsimony=d.get("max_nodes_parsimony", 30),
    )
