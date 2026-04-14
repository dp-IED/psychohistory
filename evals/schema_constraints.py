"""
Schema-level constraints: projection, epistemic anti-collapse, optional persistence hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas.schema_types import GraphSchema, Layer

from evals.structural_validity import iter_probe_files, load_probe_yaml


@dataclass
class ConstraintResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


def check_projection(schema: GraphSchema) -> ConstraintResult:
    issues: list[str] = []
    if schema.projection.required and schema.projection.dim < 1:
        issues.append("projection.dim must be >= 1 when required")
    return ConstraintResult(ok=not issues, issues=issues)


def check_epistemic_anti_collapse(schema: GraphSchema) -> ConstraintResult:
    """Penalize schemas that remove multi-perspective support for contested claims."""

    issues: list[str] = []
    m = schema.global_epistemic.min_distinct_perspectives_for_contested
    if m < 2:
        issues.append("global_epistemic.min_distinct_perspectives_for_contested should be >= 2 (anti-objective)")
    if "Claim" in schema.node_types:
        c = schema.node_types["Claim"].epistemic.min_distinct_perspectives_for_contested
        if c < 2:
            issues.append("Claim node epistemic min_distinct_perspectives_for_contested should be >= 2")
    return ConstraintResult(ok=not issues, issues=issues)


def _probe_wants_persistence(data: dict[str, Any]) -> bool:
    tier = str(data.get("compatibility_tier") or "")
    ref = data.get("ontology_delta_ref")
    if ref:
        return True
    if "v1" in tier or "patched" in tier:
        return True
    may = (
        ((data.get("schema_requirements") or {}).get("may_represent") or {}).get("node_types") or []
    )
    return bool(may)


def check_persistence_hooks(
    probe_dir: Path, schema: GraphSchema, probe_ids: set[str] | None = None
) -> ConstraintResult:
    """
    If any probe references ontology deltas / may_represent persistence types,
    ensure optional persistence node/edge kinds exist when hooks are enabled.
    """

    issues: list[str] = []
    need_persist = False
    for p in iter_probe_files(probe_dir, probe_ids=probe_ids):
        data = load_probe_yaml(p)
        if _probe_wants_persistence(data):
            need_persist = True
            break

    if not need_persist:
        return ConstraintResult(ok=True, issues=[])

    if not schema.persistence_hooks_enabled:
        issues.append("persistence_hooks_enabled is false but benchmark references persistence delta / may_represent")
    persist_names = {"LineageClaim", "TransmissionChannel", "LegacyAggregate"}
    missing = [n for n in persist_names if n not in schema.node_types]
    if missing:
        issues.append(f"optional persistence node types missing from schema: {missing}")

    if Layer.PERSISTENCE not in schema.layers_declared:
        issues.append("Layer.PERSISTENCE should appear in layers_declared when persistence is in play")

    return ConstraintResult(ok=not issues, issues=issues)


def run_schema_constraints(
    probe_dir: Path, schema: GraphSchema, probe_ids: set[str] | None = None
) -> ConstraintResult:
    acc: list[str] = []
    for fn in (check_projection, check_epistemic_anti_collapse):
        r = fn(schema)
        acc.extend(r.issues)
    r2 = check_persistence_hooks(probe_dir, schema, probe_ids=probe_ids)
    acc.extend(r2.issues)
    return ConstraintResult(ok=not acc, issues=acc)
