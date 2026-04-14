"""Probe objective spec parsing for traversal/discipline/ablation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoldenTaskSpec:
    id: str
    start_types: list[str] = field(default_factory=list)
    required_edges: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    min_hops: int = 1
    must_cross_layers: list[str] = field(default_factory=list)
    expected_answer_ids: list[str] = field(default_factory=list)


@dataclass
class AblationSpec:
    name: str
    for_tasks: list[str] = field(default_factory=list)


@dataclass
class ComplexityBudget:
    max_node_types_used: int | None = None
    max_edge_types_used: int | None = None
    max_mean_supporting_path_len: float | None = None


@dataclass
class ObjectiveProbeSpec:
    probe_id: str
    golden_tasks: list[GoldenTaskSpec] = field(default_factory=list)
    designated_ablations: list[AblationSpec] = field(default_factory=list)
    complexity_budget: ComplexityBudget = field(default_factory=ComplexityBudget)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
    return []


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_objective_spec(data: dict[str, Any], probe_stem: str) -> ObjectiveProbeSpec:
    probe_id = str(data.get("probe_id") or data.get("probeid") or probe_stem)
    out = ObjectiveProbeSpec(probe_id=probe_id)

    raw_tasks = data.get("golden_tasks") or data.get("goldentasks") or []
    if raw_tasks and not isinstance(raw_tasks, list):
        out.errors.append("golden_tasks must be a list")
        raw_tasks = []
    for idx, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            out.errors.append(f"golden_tasks[{idx}] must be a mapping")
            continue
        tid = str(raw.get("id") or f"{probe_id}_task_{idx}")
        start_types = _str_list(raw.get("start_types") or raw.get("starttypes"))
        required_edges = _str_list(
            raw.get("required_edges") or raw.get("required_edge_sequence") or raw.get("requirededges")
        )
        target_types = _str_list(
            raw.get("target_types") or raw.get("allowed_end_types") or raw.get("targettypes")
        )
        min_hops = _as_int(raw.get("min_hops") or raw.get("minhops") or 1, default=1)
        layers = _str_list(raw.get("must_cross_layers") or raw.get("required_layers") or [])
        answers = _str_list(raw.get("expected_answer_ids") or raw.get("expectedanswers") or [])
        if not start_types:
            out.errors.append(f"golden_tasks[{idx}] missing start_types")
        if not target_types:
            out.errors.append(f"golden_tasks[{idx}] missing target_types/allowed_end_types")
        if not required_edges:
            out.errors.append(f"golden_tasks[{idx}] missing required_edges/required_edge_sequence")
        out.golden_tasks.append(
            GoldenTaskSpec(
                id=tid,
                start_types=start_types,
                required_edges=required_edges,
                target_types=target_types,
                min_hops=max(1, min_hops),
                must_cross_layers=layers,
                expected_answer_ids=answers,
            )
        )

    raw_ablations = data.get("designated_ablations") or data.get("designatedablations") or []
    if raw_ablations and not isinstance(raw_ablations, list):
        out.errors.append("designated_ablations must be a list")
        raw_ablations = []
    for idx, raw in enumerate(raw_ablations):
        if not isinstance(raw, dict):
            out.errors.append(f"designated_ablations[{idx}] must be a mapping")
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            out.errors.append(f"designated_ablations[{idx}] missing name")
            continue
        out.designated_ablations.append(
            AblationSpec(name=name, for_tasks=_str_list(raw.get("for_tasks") or raw.get("fortasks")))
        )

    raw_budget = data.get("complexity_budget") or data.get("complexitybudget") or {}
    if raw_budget and not isinstance(raw_budget, dict):
        out.errors.append("complexity_budget must be a mapping")
        raw_budget = {}
    out.complexity_budget = ComplexityBudget(
        max_node_types_used=_as_int(raw_budget.get("max_node_types_used"), default=0)
        if "max_node_types_used" in raw_budget
        else None,
        max_edge_types_used=_as_int(raw_budget.get("max_edge_types_used"), default=0)
        if "max_edge_types_used" in raw_budget
        else None,
        max_mean_supporting_path_len=_as_float_or_none(raw_budget.get("max_mean_supporting_path_len")),
    )

    task_ids = {task.id for task in out.golden_tasks}
    for ab in out.designated_ablations:
        missing = [task_id for task_id in ab.for_tasks if task_id not in task_ids]
        if missing:
            out.errors.append(f"designated_ablations.{ab.name} references unknown tasks: {missing}")

    return out
