"""Pydantic models for the graph-builder probe / query interface (q_struct v0)."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AssumptionEmphasis(str, Enum):
    """Soft-gate emphasis aligned with the five named assumption types in the graph-builder contract."""

    PERSISTENCE = "Persistence"
    PROPAGATION = "Propagation"
    PRECURSOR = "Precursor"
    SUPPRESSION = "Suppression"
    COORDINATION = "Coordination"


class ActorStateQuery(BaseModel):
    """JSON `actor_state`: geography and actor typing with optional hints and an as-of slice."""

    geography: list[str] = Field(..., min_length=1, description="At least one region or place token.")
    actor_type: list[str] = Field(..., min_length=1, description="At least one actor kind (person, group, institution, etc.).")
    entity_hints: list[str] = Field(default_factory=list, description="Optional string hints to ground retrieval.")
    state_flags: list[str] = Field(default_factory=list, description="Optional behavioural-state tags.")
    as_of: date = Field(description="Evidence slice date for the actor-state query.")


class TrendThreadV0(BaseModel):
    """Minimal v0 placeholder for trend_thread in q_struct."""

    label: str | None = Field(default=None, description="Reserved: named slow-moving dynamic.")
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Forward-compatible bag for future trend-thread fields.",
    )


class HistoricalAnalogueV0(BaseModel):
    """Minimal v0 placeholder for historical_analogue in q_struct."""

    template_key: str | None = Field(default=None, description="Reserved: analogue template identifier.")
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Forward-compatible bag for future analogue fields.",
    )


class QStructV0(BaseModel):
    """Versioned query structure passed into the graph builder."""

    schema_version: Literal["q_struct_v0"] = "q_struct_v0"
    actor_state: ActorStateQuery
    trend_thread: TrendThreadV0 | None = None
    historical_analogue: HistoricalAnalogueV0 | None = None


class LensParamsV0(BaseModel):
    """Forecaster / template lens parameters aligned with v0 prompt slots."""

    horizon_days: int | None = Field(
        default=None,
        ge=1,
        description="Optional forecast horizon in whole days; omit when not constrained.",
    )
    context_snippet: str | None = Field(
        default=None,
        description="Optional short free-text context carried into templates or embeddings.",
    )


class GenerationMeta(BaseModel):
    """Provenance for synthetic or templated probe rows."""

    template_id: str = Field(description="Template or recipe identifier used to render the probe.")
    generator_version: str = Field(description="Version string of the generator that produced this row.")
    seed: int | None = Field(default=None, description="Optional RNG seed for reproducibility.")
    notes: str | None = Field(default=None, description="Optional short human or pipeline notes.")
    assumption_gate_coverage: AssumptionEmphasis | None = Field(
        default=None,
        description=(
            "Explicit assumption-gate label for corpus coverage audits (JSONL / logs). "
            "When set, must match ProbeRecord.assumption_emphasis. "
            "France harness and similar corpora should set this on every row so Stage 1 "
            "can verify no gate is starved without re-deriving from other fields."
        ),
    )


class ProbeRecord(BaseModel):
    """A single training or evaluation probe with structured query and lens."""

    probe_id: str
    origin: date = Field(description="Training-row evidence cutoff / weekly origin date.")
    nl_text: str
    q_struct: QStructV0
    lens_params: LensParamsV0
    assumption_emphasis: AssumptionEmphasis = Field(
        description="Which assumption gate is emphasized for coverage and modulation.",
    )
    subject_id: str | None = None
    generation_meta: GenerationMeta

    @model_validator(mode="after")
    def actor_as_of_not_after_origin(self) -> ProbeRecord:
        as_of = self.q_struct.actor_state.as_of
        if as_of > self.origin:
            raise ValueError(
                f"q_struct.actor_state.as_of ({as_of}) must be on or before origin ({self.origin}); "
                "same calendar day is allowed.",
            )
        cov = self.generation_meta.assumption_gate_coverage
        if cov is not None and cov != self.assumption_emphasis:
            raise ValueError(
                "generation_meta.assumption_gate_coverage must match assumption_emphasis when set; "
                f"got coverage={cov!r} vs emphasis={self.assumption_emphasis!r}",
            )
        return self


class CompileTrace(BaseModel):
    """Compile-time diagnostics for probe materialization or validation."""

    input_hash: str
    validation_errors: list[str] = Field(default_factory=list)
    normalized_embedding_text: str | None = Field(
        default=None,
        description="Optional canonical text used for embedding after normalization.",
    )


__all__ = [
    "ActorStateQuery",
    "AssumptionEmphasis",
    "CompileTrace",
    "GenerationMeta",
    "HistoricalAnalogueV0",
    "LensParamsV0",
    "ProbeRecord",
    "QStructV0",
    "TrendThreadV0",
]
