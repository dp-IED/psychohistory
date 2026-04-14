"""Machine-readable eval reports + placeholder external metric hooks."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from schemas.schema_types import GraphSchema

from evals.probe_functional import FunctionalTaskResult
from evals.schema_constraints import ConstraintResult
from evals.structural_validity import StructuralResult


class StubScores(BaseModel):
    """Placeholder integrations (no API keys in v0)."""

    gdelt_coverage: float = 0.0
    synthetic_traversal_qa: float = 0.0
    polymarket_brier: float = 0.0
    persistence_layer_ablation: float = 0.0
    schema_quality: float = 0.0
    adversarial_agent: float = 0.0
    notes: str = "TODO: wire GDELT / traversal QA / Polymarket / ablation runners"


class EvalReport(BaseModel):
    """JSON-serializable composite report."""

    schema_name: str
    schema_version: str
    generated_at_utc: str
    structural_ok: bool
    functional_ok: bool
    constraints_ok: bool
    structural: list[dict[str, Any]] = Field(default_factory=list)
    functional: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    stub_scores: StubScores = Field(default_factory=StubScores)
    weights: dict[str, float] = Field(default_factory=dict)
    composite_score: float = 0.0
    traversal_score: float = 0.0
    discipline_score: float = 0.0
    ablation_gain_score: float = 0.0
    failure_class: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


def _ser_structural(results: list[StructuralResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def _ser_functional(results: list[FunctionalTaskResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


def classify_failure(
    *,
    eval_crashed: bool = False,
    structural_ok: bool = True,
    functional_ok: bool = True,
    constraints_ok: bool = True,
    incompatible_projection: bool = False,
    objective_ok: bool = True,
    score_regression: bool = False,
) -> str | None:
    if eval_crashed:
        return "eval_crash"
    if incompatible_projection:
        return "incompatible_projection"
    if not structural_ok:
        return "schema_invalid"
    if not constraints_ok:
        return "schema_invalid"
    if not objective_ok:
        return "objective_invalid"
    if not functional_ok:
        return "retrieval_regression"
    if score_regression:
        return "score_regression"
    return None


def build_report(
    schema: GraphSchema,
    structural_ok: bool,
    structural: list[StructuralResult],
    functional_ok: bool,
    functional: list[FunctionalTaskResult],
    constraints: ConstraintResult,
    stub: StubScores | None = None,
    weights: dict[str, float] | None = None,
    composite_score: float = 0.0,
    traversal_score: float = 0.0,
    discipline_score: float = 0.0,
    ablation_gain_score: float = 0.0,
    failure_class: str | None = None,
    meta: dict[str, Any] | None = None,
) -> EvalReport:
    now = datetime.now(timezone.utc).isoformat()
    return EvalReport(
        schema_name=schema.name,
        schema_version=schema.version,
        generated_at_utc=now,
        structural_ok=structural_ok,
        functional_ok=functional_ok,
        constraints_ok=constraints.ok,
        structural=_ser_structural(structural),
        functional=_ser_functional(functional),
        constraints={"issues": constraints.issues},
        stub_scores=stub or StubScores(),
        weights=weights or {},
        composite_score=composite_score,
        traversal_score=traversal_score,
        discipline_score=discipline_score,
        ablation_gain_score=ablation_gain_score,
        failure_class=failure_class,
        meta=meta or {},
    )


def report_to_jsonable(report: EvalReport) -> dict[str, Any]:
    return report.model_dump()
