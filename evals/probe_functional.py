"""
Functional retrieval-style checks derived from probe_tasks (YAML-driven).

Full multi-hop retrieval requires a populated graph + embedder: we only verify
that declared answer shapes reference types present in the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas.schema_types import GraphSchema

from evals.structural_validity import load_probe_yaml


_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class FunctionalTaskResult:
    probe_id: str
    task_id: str
    ok: bool
    score: float = 0.0
    missing_types: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _expected_types_from_shape(items: list[Any]) -> list[str]:
    types: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        s = it.strip()
        if s.lower().startswith("edge path") or "semantics" in s.lower():
            types.append("__edge_path__")
            continue
        if s.lower().startswith("time-scoped"):
            types.append("__time_subgraph__")
            continue
        if _TOKEN.match(s):
            types.append(s)
        else:
            types.append("__narrative__")
    return types


def _has_edge_path_capability(schema: GraphSchema) -> bool:
    edges = list(schema.edge_types.values())
    if not edges:
        return False
    with_layer_hints = sum(
        1 for e in edges if (e.allowed_source_layers and e.allowed_target_layers)
    )
    return with_layer_hints / len(edges) >= 0.05


def _has_time_subgraph_capability(schema: GraphSchema) -> bool:
    if schema.extensions.get("edge_temporal_binding"):
        return True
    return any(bool(n.extensions.get("temporal_profile")) for n in schema.node_types.values())


def _has_narrative_capability(schema: GraphSchema) -> bool:
    for name in ("Narrative", "MediaNarrative", "Norm"):
        n = schema.node_types.get(name)
        if not n:
            continue
        if n.extensions.get("ir_role") or n.extensions.get("retrieval_axes"):
            return True
    return False


def check_probe_functional(probe_path: Path, schema: GraphSchema) -> list[FunctionalTaskResult]:
    data = load_probe_yaml(probe_path)
    probe_id = str(data.get("probe_id") or data.get("probeid") or probe_path.stem)
    tasks_block = data.get("probe_tasks") or data.get("probetasks") or {}
    retrieval = tasks_block.get("retrieval_tasks") or tasks_block.get("retrievaltasks") or []

    have = schema.node_type_names()
    out: list[FunctionalTaskResult] = []

    for t in retrieval:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", "unknown"))
        shape = t.get("expected_answer_shape") or []
        if not isinstance(shape, list):
            continue

        missing: list[str] = []
        missing_caps: list[str] = []
        notes: list[str] = []
        checks = 0
        passed = 0
        for typ in _expected_types_from_shape(shape):
            if typ == "__edge_path__":
                checks += 1
                if _has_edge_path_capability(schema):
                    passed += 1
                else:
                    missing_caps.append("edge_path_capability")
                continue
            if typ == "__time_subgraph__":
                checks += 1
                if _has_time_subgraph_capability(schema):
                    passed += 1
                else:
                    missing_caps.append("time_subgraph_capability")
                continue
            if typ == "__narrative__":
                checks += 1
                if _has_narrative_capability(schema):
                    passed += 1
                else:
                    missing_caps.append("narrative_capability")
                continue
            checks += 1
            if typ not in have:
                missing.append(typ)
            else:
                passed += 1
        score = 1.0 if checks == 0 else passed / checks
        if missing_caps:
            notes.append(f"missing capability tokens: {sorted(set(missing_caps))}")

        out.append(
            FunctionalTaskResult(
                probe_id=probe_id,
                task_id=tid,
                ok=(not missing) and score >= 0.8,
                score=score,
                missing_types=missing,
                missing_capabilities=sorted(set(missing_caps)),
                notes=notes,
            )
        )

    return out


def run_probe_functional(probe_dir: Path, schema: GraphSchema) -> tuple[bool, list[FunctionalTaskResult]]:
    from evals.structural_validity import iter_probe_files

    all_ok = True
    acc: list[FunctionalTaskResult] = []
    for path in iter_probe_files(probe_dir):
        for r in check_probe_functional(path, schema):
            acc.append(r)
            if not r.ok:
                all_ok = False
    return all_ok, acc
