"""
Pydantic models for an extensible graph IR schema specification.

Probe-specific types are not hardcoded here: they appear as registered strings
on NodeSpec/EdgeSpec. The seed schema in base_schema.py may include common
benchmark types; autoresearch mutates that file or candidate copies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Layer(str, Enum):
    """Semantic layers aligned with benchmark probes (axes + epistemic framing)."""

    MATERIAL = "material"
    INSTITUTIONAL = "institutional"
    IDEOLOGICAL = "ideological"
    COUNTERWEIGHT = "counterweight"
    EPISTEMIC = "epistemic"
    PERSISTENCE = "persistence"


class EpistemicSpec(BaseModel):
    """How nodes/edges record confidence, disagreement, and viewpoint."""

    tracks_confidence: bool = True
    tracks_contention: bool = True
    tracks_provenance: bool = True
    perspective_aware: bool = True
    min_distinct_perspectives_for_contested: int = Field(
        default=2,
        ge=1,
        description="Anti-objective: avoid collapsing contested claims into one cluster.",
    )


class ProjectionSpec(BaseModel):
    """Fixed-dimensional constraints for downstream GNN projection."""

    dim: int = Field(ge=1)
    required: bool = True
    per_type_optional: dict[str, int] = Field(default_factory=dict)
    # TODO: bind to learned adapters once a training stack exists


class TemporalSpec(BaseModel):
    """Time-scoping for nodes and edges."""

    node_time_scoped_default: bool = True
    edge_time_scoped_default: bool = True
    supports_slice_ids: bool = True
    iso8601_span: bool = True


class NodeSpec(BaseModel):
    """Declaration of a node type the IR may instantiate."""

    name: str
    description: str = ""
    primary_layer: Layer | None = None
    epistemic: EpistemicSpec = Field(default_factory=EpistemicSpec)
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Open vocabulary: attr name -> type hint string (e.g. 'float', 'list[str]').",
    )
    extensions: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    """Declaration of an edge type (relation label)."""

    name: str
    description: str = ""
    allowed_source_layers: list[Layer] | None = None
    allowed_target_layers: list[Layer] | None = None
    epistemic: EpistemicSpec = Field(default_factory=EpistemicSpec)
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    extensions: dict[str, Any] = Field(default_factory=dict)


class GraphSchema(BaseModel):
    """Full IR schema: extensible registry of node/edge kinds + global constraints."""

    name: str = "psychohistory_graph_ir"
    version: str = "0.1.0"
    benchmark_pack: str = "v0"
    ontology_compatibility: str = "v1-patched"
    node_types: dict[str, NodeSpec] = Field(default_factory=dict)
    edge_types: dict[str, EdgeSpec] = Field(default_factory=dict)
    projection: ProjectionSpec = Field(default_factory=lambda: ProjectionSpec(dim=32, required=True))
    persistence_hooks_enabled: bool = False
    layers_declared: list[Layer] = Field(
        default_factory=lambda: [
            Layer.MATERIAL,
            Layer.INSTITUTIONAL,
            Layer.IDEOLOGICAL,
            Layer.COUNTERWEIGHT,
            Layer.EPISTEMIC,
        ]
    )
    global_epistemic: EpistemicSpec = Field(default_factory=EpistemicSpec)
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch for research metadata without schema churn.",
    )

    def node_type_names(self) -> set[str]:
        return set(self.node_types.keys())

    def edge_type_names(self) -> set[str]:
        return set(self.edge_types.keys())
