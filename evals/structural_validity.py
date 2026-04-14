"""YAML probe structural checks: must_represent types ⊆ schema registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import yaml

from schemas.schema_types import GraphSchema


@dataclass
class StructuralResult:
    probe_id: str
    ok: bool
    missing_nodes: list[str] = field(default_factory=list)
    missing_edges: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _coerce_type_dict(d: dict[str, Any]) -> dict[str, list[str]]:
    nodes = d.get("node_types") or d.get("nodetypes") or []
    edges = d.get("edge_types") or d.get("edgetypes") or []
    if isinstance(nodes, dict) and "required" in nodes:
        nodes = nodes["required"]
    if isinstance(edges, dict) and "required" in edges:
        edges = edges["required"]
    return {"node_types": list(nodes or []), "edge_types": list(edges or [])}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _canonical_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _pull_types_from_block(block: dict[str, Any]) -> dict[str, list[str]] | None:
    """Handle must_represent dict, mustrepresent list + sibling nodetypes, required.*, etc."""

    flat_nodes = (
        block.get("must_represent_node_types")
        or block.get("mustrepresentnodetypes")
        or block.get("must_represent_nodes")
    )
    flat_edges = (
        block.get("must_represent_edges")
        or block.get("mustrepresentedgetypes")
        or block.get("must_represent_relations")
    )
    if isinstance(flat_nodes, list) or isinstance(flat_edges, list):
        return {"node_types": _as_str_list(flat_nodes), "edge_types": _as_str_list(flat_edges)}

    if "must_represent" in block:
        mr = block["must_represent"]
        if isinstance(mr, dict) and (
            "node_types" in mr
            or "nodetypes" in mr
            or "edge_types" in mr
            or "edgetypes" in mr
        ):
            return _coerce_type_dict(mr)

    if "mustrepresent" in block:
        mr = block["mustrepresent"]
        if isinstance(mr, dict) and (
            "node_types" in mr
            or "nodetypes" in mr
            or "edge_types" in mr
            or "edgetypes" in mr
        ):
            return _coerce_type_dict(mr)

    nt = block.get("nodetypes") or block.get("node_types")
    et = block.get("edgetypes") or block.get("edge_types")
    if isinstance(nt, list) and nt and isinstance(nt[0], str):
        el = et if isinstance(et, list) else []
        return {"node_types": list(nt), "edge_types": list(el)}

    if isinstance(nt, dict) and "required" in nt:
        req_e = et.get("required") if isinstance(et, dict) else []
        return {"node_types": list(nt["required"]), "edge_types": list(req_e or [])}

    mr = block.get("mustrepresent") or block.get("must_represent")
    if isinstance(mr, dict):
        return _coerce_type_dict(mr)
    return None


def extract_must_represent(data: dict[str, Any]) -> dict[str, list[str]] | None:
    """Support probe_spec_v2, schema_requirements.mustrepresent (6f), and schemarequirements (6a–6e)."""

    for key in ("schema_requirements", "schemarequirements", "schemaRequirements"):
        block = data.get(key)
        if isinstance(block, dict):
            got = _pull_types_from_block(block)
            if got and (got["node_types"] or got["edge_types"]):
                return got
    return None


def load_probe_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Probe {path} root must be a mapping")
    return data


def probe_identifiers(probe_path: Path, data: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            data.get("probe_id"),
            data.get("probeid"),
            data.get("id"),
            probe_path.stem,
        )
        if value
    }


def iter_probe_files(probe_dir: Path, probe_ids: set[str] | None = None) -> list[Path]:
    """Benchmark probe YAML files (excludes ontology deltas and CSV)."""

    out: list[Path] = []
    for p in sorted(probe_dir.glob("*.yaml")):
        if p.name.startswith("."):
            continue
        if probe_ids:
            data = load_probe_yaml(p)
            if probe_identifiers(p, data).isdisjoint(probe_ids):
                continue
        out.append(p)
    return out


def check_probe_structural(probe_path: Path, schema: GraphSchema) -> StructuralResult:
    data = load_probe_yaml(probe_path)
    probe_id = str(data.get("probe_id") or data.get("probeid") or probe_path.stem)
    req = extract_must_represent(data)
    if req is None:
        return StructuralResult(
            probe_id=probe_id,
            ok=False,
            errors=["missing extractable must_represent types"],
        )

    nodes = list((req.get("node_types") or []) if isinstance(req, dict) else [])
    edges = list((req.get("edge_types") or []) if isinstance(req, dict) else [])

    have_n = schema.node_type_names()
    have_e = schema.edge_type_names()
    have_e_canonical = {_canonical_token(e) for e in have_e}

    missing_nodes = [n for n in nodes if n not in have_n]
    missing_edges = [
        e
        for e in edges
        if e not in have_e and _canonical_token(e) not in have_e_canonical
    ]

    ok = not missing_nodes and not missing_edges
    errors: list[str] = []
    if missing_nodes:
        errors.append(f"missing node types: {missing_nodes}")
    if missing_edges:
        errors.append(f"missing edge types: {missing_edges}")

    return StructuralResult(
        probe_id=probe_id,
        ok=ok,
        missing_nodes=missing_nodes,
        missing_edges=missing_edges,
        errors=errors,
    )


def run_structural_validity(
    probe_dir: Path, schema: GraphSchema, probe_ids: set[str] | None = None
) -> tuple[bool, list[StructuralResult]]:
    results: list[StructuralResult] = []
    all_ok = True
    for path in iter_probe_files(probe_dir, probe_ids=probe_ids):
        r = check_probe_structural(path, schema)
        results.append(r)
        if not r.ok:
            all_ok = False
    return all_ok, results
