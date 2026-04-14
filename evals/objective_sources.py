"""External-source objective buckets for probe graph artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.probe_graph_precompute import get_artifact_status, load_graph_artifact
from evals.structural_validity import iter_probe_files, load_probe_yaml
from evals.wikidata_linking import (
    extract_seed_entity_labels,
    node_labels,
    normalize_entity_label,
    wikidata_qid_for_node,
)
from schemas.schema_types import GraphSchema
from evals.objective_eval import _build_indices, _find_paths
from evals.objective_specs import parse_objective_spec


_GROUNDABLE_TYPES = {
    "Actor",
    "Institution",
    "Office",
    "Province",
    "MilitaryFormation",
    "Event",
    "Policy",
    "Empire",
    "FrontierPeople",
    "AdministrativeRegion",
    "LocalElite",
    "Place",
    "Organization",
    "Person",
}


@dataclass
class SourceMetric:
    """A scored external-source objective component."""

    score: float
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class SourceProbeResult:
    probe_id: str
    v: SourceMetric
    g: SourceMetric
    i: SourceMetric
    f: SourceMetric
    c: SourceMetric
    errors: list[str] = field(default_factory=list)


@dataclass
class SourceObjectiveResult:
    score: float
    buckets: dict[str, SourceMetric]
    probes: list[SourceProbeResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "buckets": {name: asdict(metric) for name, metric in self.buckets.items()},
            "probes": [
                {
                    "probe_id": probe.probe_id,
                    "V": asdict(probe.v),
                    "G": asdict(probe.g),
                    "I": asdict(probe.i),
                    "F": asdict(probe.f),
                    "C": asdict(probe.c),
                    "errors": probe.errors,
                }
                for probe in self.probes
            ],
            "errors": self.errors,
            "weights": {"V": 0.15, "G": 0.20, "I": 0.20, "F": 0.30, "C": 0.15},
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _groundable_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    typed = [
        node
        for node in nodes
        if str(node.get("type") or "") in _GROUNDABLE_TYPES
        or any(key in node for key in ("label", "name", "title", "external_ids", "wikidata_id"))
    ]
    return typed or nodes


def evaluate_wikidata_grounding(probe_data: dict[str, Any], graph: dict[str, Any]) -> SourceMetric:
    nodes = _groundable_nodes(graph)
    grounded_nodes: list[dict[str, Any]] = []
    qids: set[str] = set()
    for node in nodes:
        qid = wikidata_qid_for_node(node)
        if qid:
            grounded_nodes.append(node)
            qids.add(qid)

    node_link_rate = (len(grounded_nodes) / len(nodes)) if nodes else 0.0
    seed_labels = extract_seed_entity_labels(probe_data)
    grounded_label_index: set[str] = set()
    for node in grounded_nodes:
        grounded_label_index.update(node_labels(node))
    matched_seed_labels = [
        label for label in seed_labels if normalize_entity_label(label) in grounded_label_index
    ]
    seed_link_rate = (len(matched_seed_labels) / len(seed_labels)) if seed_labels else None

    if seed_link_rate is None:
        score = node_link_rate
    else:
        score = (0.65 * node_link_rate) + (0.35 * seed_link_rate)

    notes: list[str] = []
    if not nodes:
        notes.append("graph artifact has no nodes")
    if seed_labels and not matched_seed_labels:
        notes.append("no seed_entities labels were matched to Wikidata-grounded graph nodes")
    return SourceMetric(
        score=_clip01(score),
        status="ok",
        metrics={
            "source": "Wikidata",
            "node_link_rate": node_link_rate,
            "seed_entity_link_rate": seed_link_rate,
            "grounded_nodes": len(grounded_nodes),
            "groundable_nodes": len(nodes),
            "seed_entities": len(seed_labels),
            "matched_seed_entities": len(matched_seed_labels),
            "unique_qids": len(qids),
        },
        notes=notes,
    )


def _task_path_grounding_rate(
    probe_data: dict[str, Any],
    graph: dict[str, Any],
    schema: GraphSchema,
    probe_stem: str,
) -> float | None:
    spec = parse_objective_spec(probe_data, probe_stem)
    if not spec.golden_tasks:
        return None
    node_by_id, outgoing = _build_indices(graph, schema)
    path_node_ids: set[str] = set()
    for task in spec.golden_tasks:
        matches = _find_paths(task=task, node_by_id=node_by_id, outgoing=outgoing, schema=schema)
        if not matches:
            continue
        for node_id in matches[0].get("path_nodes", []):
            path_node_ids.add(str(node_id))
    if not path_node_ids:
        return None
    nodes = [node_by_id[node_id] for node_id in path_node_ids if node_id in node_by_id]
    groundable = _groundable_nodes({"nodes": nodes, "edges": []})
    if not groundable:
        return None
    grounded = sum(1 for node in groundable if wikidata_qid_for_node(node))
    return grounded / len(groundable)


def _stub_metric(name: str) -> SourceMetric:
    return SourceMetric(
        score=0.0,
        status="unavailable",
        notes=[f"{name} adapter is not implemented yet"],
    )


def evaluate_source_objectives(
    *,
    probe_dir: Path,
    schema: GraphSchema,
    artifact_dir: Path,
    builder_version: str,
    probe_ids: set[str] | None = None,
) -> SourceObjectiveResult:
    probe_results: list[SourceProbeResult] = []
    errors: list[str] = []
    g_scores: list[float] = []

    for probe_file in iter_probe_files(probe_dir, probe_ids=probe_ids):
        probe_data = load_probe_yaml(probe_file)
        probe_id = str(probe_data.get("probe_id") or probe_data.get("probeid") or probe_file.stem)
        probe_errors: list[str] = []
        status = get_artifact_status(
            artifact_dir=artifact_dir,
            probe_file=probe_file,
            probe_data=probe_data,
            schema=schema,
            builder_version=builder_version,
        )
        if not status.fresh:
            msg = f"graph artifact not fresh for Wikidata grounding: {status.reason}"
            probe_errors.append(msg)
            errors.append(f"{probe_id}: {msg}")
            g_metric = SourceMetric(score=0.0, status="unavailable", notes=[msg])
        else:
            graph = load_graph_artifact(status.artifact_path)
            g_metric = evaluate_wikidata_grounding(probe_data, graph)
            path_rate = _task_path_grounding_rate(probe_data, graph, schema, probe_file.stem)
            if path_rate is not None:
                g_metric.metrics["task_path_grounding_rate"] = path_rate
                base_score = g_metric.score
                g_metric.score = _clip01((0.8 * base_score) + (0.2 * path_rate))
            g_scores.append(g_metric.score)

        probe_results.append(
            SourceProbeResult(
                probe_id=probe_id,
                v=_stub_metric("V temporal validity"),
                g=g_metric,
                i=_stub_metric("I ideology and narrative retrieval"),
                f=_stub_metric("F forecast skill"),
                c=_stub_metric("C counterweight sensitivity"),
                errors=probe_errors,
            )
        )

    aggregate_g = SourceMetric(
        score=(sum(g_scores) / len(g_scores)) if g_scores else 0.0,
        status="ok" if g_scores else "unavailable",
        metrics={"source": "Wikidata", "probes_scored": len(g_scores)},
        notes=[] if g_scores else ["no fresh graph artifacts available for Wikidata grounding"],
    )
    buckets = {
        "V": _stub_metric("V temporal validity"),
        "G": aggregate_g,
        "I": _stub_metric("I ideology and narrative retrieval"),
        "F": _stub_metric("F forecast skill"),
        "C": _stub_metric("C counterweight sensitivity"),
    }
    composite = 0.20 * aggregate_g.score
    return SourceObjectiveResult(score=_clip01(composite), buckets=buckets, probes=probe_results, errors=errors)
