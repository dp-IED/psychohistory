"""Graph-backed objective evaluation: traversal, discipline, and ablation lift."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.objective_specs import AblationSpec, GoldenTaskSpec, parse_objective_spec
from evals.probe_graph_precompute import get_artifact_status, load_graph_artifact
from evals.structural_validity import _canonical_token, iter_probe_files, load_probe_yaml
from schemas.schema_types import GraphSchema, Layer


@dataclass
class TraversalTaskResult:
    probe_id: str
    task_id: str
    ok: bool
    score: float
    path_found: bool
    matched_paths: int
    best_path_len: int | None
    missing_layers: list[str] = field(default_factory=list)
    missing_answers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ObjectiveProbeResult:
    probe_id: str
    traversal_score: float
    discipline_score: float
    ablation_gain_score: float
    task_results: list[TraversalTaskResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class ObjectiveLayerResult:
    ok: bool
    applied: bool
    traversal_score: float
    discipline_score: float
    ablation_gain_score: float
    probes: list[ObjectiveProbeResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _node_layer(node: dict[str, Any], schema: GraphSchema) -> str:
    layer = node.get("layer")
    if isinstance(layer, str) and layer:
        return layer.lower()
    typ = str(node.get("type") or "")
    if typ and typ in schema.node_types:
        primary = schema.node_types[typ].primary_layer
        if primary is not None:
            return str(primary.value).lower()
    return "unscoped"


def _build_indices(
    graph: dict[str, Any], schema: GraphSchema
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("id") or "")
        if not nid:
            continue
        node_by_id[nid] = node
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source") or "")
        dst = str(edge.get("target") or "")
        et = str(edge.get("type") or "")
        if not src or not dst or not et:
            continue
        if src not in node_by_id or dst not in node_by_id:
            continue
        edge_norm = {
            "source": src,
            "target": dst,
            "type": et,
            "type_token": _canonical_token(et),
            "source_layer": _node_layer(node_by_id[src], schema),
            "target_layer": _node_layer(node_by_id[dst], schema),
        }
        outgoing[src].append(edge_norm)
    return node_by_id, outgoing


def _find_paths(
    *,
    task: GoldenTaskSpec,
    node_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    schema: GraphSchema,
) -> list[dict[str, Any]]:
    required_tokens = [_canonical_token(edge.rstrip("?")) for edge in task.required_edges]
    start_ids = [nid for nid, node in node_by_id.items() if str(node.get("type")) in set(task.start_types)]
    target_types = set(task.target_types)
    max_depth = max(task.min_hops + 4, len(required_tokens) + 1)
    matches: list[dict[str, Any]] = []

    def dfs(
        node_id: str,
        depth: int,
        req_idx: int,
        path_nodes: list[str],
        path_edges: list[str],
        layers_seen: set[str],
    ) -> None:
        if depth >= max_depth:
            return
        for edge in outgoing.get(node_id, []):
            edge_token = str(edge["type_token"])
            if req_idx < len(required_tokens) and edge_token != required_tokens[req_idx]:
                continue
            next_idx = req_idx + 1
            next_node = str(edge["target"])
            next_node_type = str(node_by_id[next_node].get("type") or "")
            next_layers = set(layers_seen)
            next_layers.add(_node_layer(node_by_id[next_node], schema))
            next_path_nodes = path_nodes + [next_node]
            next_path_edges = path_edges + [str(edge["type"])]
            hops = len(next_path_edges)
            if (
                next_node_type in target_types
                and next_idx >= len(required_tokens)
                and hops >= task.min_hops
            ):
                matches.append(
                    {
                        "path_nodes": next_path_nodes,
                        "path_edges": next_path_edges,
                        "layers_seen": sorted(next_layers),
                        "answer_id": next_node,
                        "answer_type": next_node_type,
                        "hops": hops,
                    }
                )
            dfs(
                next_node,
                depth=depth + 1,
                req_idx=next_idx,
                path_nodes=next_path_nodes,
                path_edges=next_path_edges,
                layers_seen=next_layers,
            )

    for start_id in start_ids:
        layers = {_node_layer(node_by_id[start_id], schema)}
        dfs(
            start_id,
            depth=0,
            req_idx=0,
            path_nodes=[start_id],
            path_edges=[],
            layers_seen=layers,
        )
    return matches


def _score_task(task: GoldenTaskSpec, matched_paths: list[dict[str, Any]]) -> TraversalTaskResult:
    missing_layers: list[str] = []
    missing_answers: list[str] = []
    if not matched_paths:
        return TraversalTaskResult(
            probe_id="",
            task_id=task.id,
            ok=False,
            score=0.0,
            path_found=False,
            matched_paths=0,
            best_path_len=None,
            missing_layers=list(task.must_cross_layers),
            missing_answers=list(task.expected_answer_ids),
            notes=["no valid typed path found"],
        )

    required_layers = {layer.lower() for layer in task.must_cross_layers}
    answer_ids = {str(path["answer_id"]) for path in matched_paths}
    layer_hits = 0
    for path in matched_paths:
        path_layers = {str(layer).lower() for layer in path.get("layers_seen", [])}
        if required_layers.issubset(path_layers):
            layer_hits += 1
    layer_ratio = layer_hits / len(matched_paths) if matched_paths else 0.0
    if required_layers and layer_ratio < 1.0:
        seen_union = set()
        for path in matched_paths:
            seen_union.update({str(layer).lower() for layer in path.get("layers_seen", [])})
        missing_layers = sorted(required_layers - seen_union)

    if task.expected_answer_ids:
        expected = set(task.expected_answer_ids)
        answer_ratio = len(expected.intersection(answer_ids)) / len(expected)
        missing_answers = sorted(expected - answer_ids)
        semantic_match = 0.5 * layer_ratio + 0.5 * answer_ratio
    else:
        semantic_match = layer_ratio if required_layers else 1.0

    avg_hops = sum(int(path["hops"]) for path in matched_paths) / len(matched_paths)
    minimality = _clip01(float(task.min_hops) / max(avg_hops, 1.0))
    reach = 1.0
    score = _clip01((0.5 * reach) + (0.3 * semantic_match) + (0.2 * minimality))
    ok = score >= 0.7
    best_path_len = min(int(path["hops"]) for path in matched_paths)
    notes: list[str] = []
    if missing_layers:
        notes.append(f"missing required layers across matches: {missing_layers}")
    if missing_answers:
        notes.append(f"missing expected answers: {missing_answers}")
    return TraversalTaskResult(
        probe_id="",
        task_id=task.id,
        ok=ok,
        score=score,
        path_found=True,
        matched_paths=len(matched_paths),
        best_path_len=best_path_len,
        missing_layers=missing_layers,
        missing_answers=missing_answers,
        notes=notes,
    )


def _alias_redundancy(schema: GraphSchema, used_edge_types: set[str]) -> float:
    families: dict[str, set[str]] = defaultdict(set)
    for edge_name, edge_spec in schema.edge_types.items():
        family = edge_spec.extensions.get("normalization_family")
        if isinstance(family, str) and family:
            families[family].add(edge_name)
    if not families:
        return 0.0
    penalties: list[float] = []
    for names in families.values():
        if len(names) <= 1:
            continue
        used = len(names.intersection(used_edge_types))
        penalties.append((len(names) - used) / len(names))
    if not penalties:
        return 0.0
    return _clip01(sum(penalties) / len(penalties))


def _apply_ablation(graph: dict[str, Any], schema: GraphSchema, ablation: AblationSpec) -> dict[str, Any]:
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [dict(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)]
    name = ablation.name.strip().lower()
    if name == "persistence_off":
        persistence_nodes = {
            node_name
            for node_name, spec in schema.node_types.items()
            if spec.primary_layer == Layer.PERSISTENCE
        }
        removed_node_ids = {
            str(node.get("id"))
            for node in nodes
            if str(node.get("type") or "") in persistence_nodes
        }
        edges_to_drop = {
            _canonical_token("inherits_from"),
            _canonical_token("transmits_through"),
            _canonical_token("leaves_legacy_via"),
        }
        nodes = [node for node in nodes if str(node.get("id")) not in removed_node_ids]
        edges = [
            edge
            for edge in edges
            if str(edge.get("source")) not in removed_node_ids
            and str(edge.get("target")) not in removed_node_ids
            and _canonical_token(str(edge.get("type") or "")) not in edges_to_drop
        ]
    elif name == "epistemic_off":
        edges_to_drop = {
            _canonical_token("presupposes"),
            _canonical_token("contests"),
            _canonical_token("delegitimizes"),
            _canonical_token("polarizes"),
        }
        edges = [
            edge
            for edge in edges
            if _canonical_token(str(edge.get("type") or "")) not in edges_to_drop
        ]
    return {"nodes": nodes, "edges": edges}


def evaluate_objective_layer(
    *,
    probe_dir: Path,
    schema: GraphSchema,
    artifact_dir: Path,
    builder_version: str,
    probe_ids: set[str] | None = None,
) -> ObjectiveLayerResult:
    probe_results: list[ObjectiveProbeResult] = []
    global_errors: list[str] = []
    traversal_scores: list[float] = []
    discipline_scores: list[float] = []
    ablation_scores: list[float] = []
    benchmark_probe_count = 0
    probe_files = iter_probe_files(probe_dir, probe_ids=probe_ids)
    if not probe_files:
        global_errors.append("strict objective mode requires at least one probe yaml")

    for probe_file in probe_files:
        probe_data = load_probe_yaml(probe_file)
        probe_id = str(probe_data.get("probe_id") or probe_data.get("probeid") or probe_file.stem)
        spec = parse_objective_spec(probe_data, probe_file.stem)
        errors = list(spec.errors)
        required_keys = ("golden_tasks", "designated_ablations", "complexity_budget")
        for key in required_keys:
            if key not in probe_data:
                errors.append(f"strict benchmark mode requires '{key}'")

        status = get_artifact_status(
            artifact_dir=artifact_dir,
            probe_file=probe_file,
            probe_data=probe_data,
            schema=schema,
            builder_version=builder_version,
        )
        if not status.fresh:
            errors.append(f"graph artifact not fresh for probe {status.probe_id}: {status.reason}")

        if errors:
            probe_results.append(
                ObjectiveProbeResult(
                    probe_id=probe_id,
                    traversal_score=0.0,
                    discipline_score=0.0,
                    ablation_gain_score=0.0,
                    errors=errors,
                )
            )
            global_errors.extend(errors)
            continue

        if not status.artifact_path.exists():
            continue
        graph = load_graph_artifact(status.artifact_path)
        node_by_id, outgoing = _build_indices(graph, schema)
        task_results: list[TraversalTaskResult] = []
        task_score_by_id: dict[str, float] = {}
        task_path_map: dict[str, dict[str, Any]] = {}
        for task in spec.golden_tasks:
            matches = _find_paths(task=task, node_by_id=node_by_id, outgoing=outgoing, schema=schema)
            t_result = _score_task(task, matches)
            t_result.probe_id = probe_id
            task_results.append(t_result)
            task_score_by_id[task.id] = t_result.score
            if matches:
                task_path_map[task.id] = matches[0]

        traversal = sum(task_score_by_id.values()) / len(task_score_by_id) if task_score_by_id else 0.0

        used_node_types: set[str] = set()
        used_edge_types: set[str] = set()
        for path in task_path_map.values():
            for node_id in path.get("path_nodes", []):
                node = node_by_id.get(str(node_id))
                if node:
                    used_node_types.add(str(node.get("type") or ""))
            for edge_type in path.get("path_edges", []):
                used_edge_types.add(str(edge_type))

        total_node_types = max(1, len(schema.node_types))
        total_edge_types = max(1, len(schema.edge_types))
        unused_node_rate = _clip01(1.0 - (len(used_node_types) / total_node_types))
        unused_edge_rate = _clip01(1.0 - (len(used_edge_types) / total_edge_types))
        alias_redundancy = _alias_redundancy(schema, used_edge_types)
        sprawl_terms = [
            max(0.0, float(path.get("hops", 0)) - float(task.min_hops))
            for task in spec.golden_tasks
            for path in [task_path_map.get(task.id)]
            if path is not None
        ]
        mean_path_sprawl = (sum(sprawl_terms) / len(sprawl_terms)) if sprawl_terms else 1.0
        mean_path_sprawl_norm = _clip01(mean_path_sprawl / 4.0)
        discipline = _clip01(
            traversal
            - (0.25 * unused_node_rate)
            - (0.25 * unused_edge_rate)
            - (0.20 * alias_redundancy)
            - (0.30 * mean_path_sprawl_norm)
        )

        deltas: list[float] = []
        for ablation in spec.designated_ablations:
            if not ablation.for_tasks:
                continue
            ablated = _apply_ablation(graph, schema, ablation)
            a_nodes, a_outgoing = _build_indices(ablated, schema)
            for task_id in ablation.for_tasks:
                task = next((item for item in spec.golden_tasks if item.id == task_id), None)
                if task is None:
                    continue
                full_score = task_score_by_id.get(task_id, 0.0)
                ablated_paths = _find_paths(task=task, node_by_id=a_nodes, outgoing=a_outgoing, schema=schema)
                ablated_score = _score_task(task, ablated_paths).score
                deltas.append(max(0.0, full_score - ablated_score))
        ablation_gain = _clip01(sum(deltas) / len(deltas)) if deltas else 0.0

        probe_results.append(
            ObjectiveProbeResult(
                probe_id=probe_id,
                traversal_score=traversal,
                discipline_score=discipline,
                ablation_gain_score=ablation_gain,
                task_results=task_results,
                diagnostics={
                    "unused_node_rate": unused_node_rate,
                    "unused_edge_rate": unused_edge_rate,
                    "alias_redundancy": alias_redundancy,
                    "mean_path_sprawl": mean_path_sprawl,
                    "artifact_path": str(status.artifact_path),
                },
                errors=errors,
            )
        )
        benchmark_probe_count += 1
        traversal_scores.append(traversal)
        discipline_scores.append(discipline)
        ablation_scores.append(ablation_gain)

    applied = benchmark_probe_count > 0
    if not applied:
        global_errors.append("strict objective mode requires at least one probe with golden_tasks")
    return ObjectiveLayerResult(
        ok=not global_errors,
        applied=applied,
        traversal_score=(sum(traversal_scores) / len(traversal_scores)) if traversal_scores else 0.0,
        discipline_score=(sum(discipline_scores) / len(discipline_scores)) if discipline_scores else 0.0,
        ablation_gain_score=(sum(ablation_scores) / len(ablation_scores)) if ablation_scores else 0.0,
        probes=probe_results,
        errors=global_errors,
    )
