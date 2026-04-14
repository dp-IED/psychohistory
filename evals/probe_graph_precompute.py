"""Precompute and validate per-probe graph artifacts used by objective evals."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import typer

from evals.objective_specs import parse_objective_spec
from evals.graph_artifact_contract import GRAPH_ARTIFACT_FORMAT, GraphArtifactV1, normalize_graph_artifact
from evals.structural_validity import iter_probe_files, load_probe_yaml
from evals.wikidata_linking import (
    DEFAULT_WIKIDATA_CACHE_PATH,
    enrich_graph_with_wikidata,
    iter_seed_entity_items,
    normalize_entity_label,
    seed_entity_node_type,
)
from schemas.schema_registry import load_graph_schema
from schemas.schema_types import GraphSchema


@dataclass
class ArtifactStatus:
    probe_id: str
    artifact_path: Path
    manifest_path: Path
    fresh: bool
    reason: str = ""


def _sha_jsonable(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(packed.encode("utf-8")).hexdigest()


def schema_fingerprint(schema: GraphSchema) -> str:
    return _sha_jsonable(schema.model_dump(mode="json"))


def probe_fingerprint(probe_data: dict[str, Any]) -> str:
    return _sha_jsonable(probe_data)


def _probe_id(probe_file: Path, probe_data: dict[str, Any]) -> str:
    return str(probe_data.get("probe_id") or probe_data.get("probeid") or probe_file.stem)


def _paths_for_probe(artifact_dir: Path, probe_id: str) -> tuple[Path, Path]:
    artifact_path = artifact_dir / f"{probe_id}.graph.json"
    manifest_path = artifact_dir / f"{probe_id}.manifest.json"
    return artifact_path, manifest_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _is_artifact_fresh(
    manifest: dict[str, Any],
    expected_schema_fp: str,
    expected_probe_fp: str,
    expected_builder_version: str,
) -> tuple[bool, str]:
    if manifest.get("schema_fingerprint") != expected_schema_fp:
        return False, "schema_fingerprint mismatch"
    if manifest.get("probe_fingerprint") != expected_probe_fp:
        return False, "probe_fingerprint mismatch"
    if manifest.get("builder_version") != expected_builder_version:
        return False, "builder_version mismatch"
    return True, ""


def get_artifact_status(
    *,
    artifact_dir: Path,
    probe_file: Path,
    probe_data: dict[str, Any],
    schema: GraphSchema,
    builder_version: str,
) -> ArtifactStatus:
    probe_id = _probe_id(probe_file, probe_data)
    artifact_path, manifest_path = _paths_for_probe(artifact_dir, probe_id)
    if not artifact_path.exists() or not manifest_path.exists():
        return ArtifactStatus(
            probe_id=probe_id,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            fresh=False,
            reason="artifact or manifest missing",
        )
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return ArtifactStatus(
            probe_id=probe_id,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            fresh=False,
            reason=f"manifest unreadable: {exc}",
        )
    ok, reason = _is_artifact_fresh(
        manifest,
        expected_schema_fp=schema_fingerprint(schema),
        expected_probe_fp=probe_fingerprint(probe_data),
        expected_builder_version=builder_version,
    )
    return ArtifactStatus(
        probe_id=probe_id,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        fresh=ok,
        reason=reason,
    )


def load_graph_artifact(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"{path} must include 'nodes' list and 'edges' list")
    if payload.get("artifact_format") == GRAPH_ARTIFACT_FORMAT:
        GraphArtifactV1.model_validate(payload)
    return payload


def _ensure_agent_command_permissions(command: str) -> str:
    stripped = command.lstrip()
    if not stripped.startswith("agent "):
        return command
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "agent":
        return command

    existing = set(tokens)
    extras: list[str] = []
    if "--trust" not in existing:
        extras.append("--trust")
    if "--force" not in existing and "-f" not in existing and "--yolo" not in existing:
        extras.append("--force")
    if "-p" not in existing and "--print" not in existing:
        extras.append("--print")

    if not extras:
        return command
    normalized = [tokens[0], *extras, *tokens[1:]]
    return shlex.join(normalized)


def _synthesize_graph_for_probe(*, probe_file: Path, schema: GraphSchema) -> dict[str, Any]:
    """Deterministic minimal graph satisfying golden_tasks edge sequences when possible."""

    probe_data = load_probe_yaml(probe_file)
    spec = parse_objective_spec(probe_data, probe_file.stem)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    task_labels: list[dict[str, Any]] = []
    target_table: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    seed_ids_by_type: dict[str, list[str]] = {}
    scope = probe_data.get("scope") if isinstance(probe_data.get("scope"), dict) else {}
    temporal = probe_data.get("temporal") if isinstance(probe_data.get("temporal"), dict) else {}
    temporal_window = scope.get("temporal_window") if isinstance(scope.get("temporal_window"), dict) else {}
    time_span = {
        "start": temporal.get("start") or temporal_window.get("start_year"),
        "end": temporal.get("end") or temporal_window.get("end_year"),
        "granularity": temporal.get("slice_granularity"),
    }

    def add_node(node_id: str, node_type: str, **extra: Any) -> None:
        if node_id in node_ids:
            return
        layer = None
        if node_type in schema.node_types:
            primary = schema.node_types[node_type].primary_layer
            if primary is not None:
                layer = str(primary.value)
        payload: dict[str, Any] = {
            "id": node_id,
            "type": node_type,
            "time": time_span,
            "slice_ids": [probe_data.get("probe_id") or probe_data.get("id") or probe_file.stem],
            "train_eval_split": "eval",
        }
        if layer:
            payload["layer"] = layer
        payload.update(extra)
        nodes.append(payload)
        node_ids.add(node_id)

    def add_edge(source: str, target: str, edge_type: str, task_id: str | None = None) -> None:
        payload: dict[str, Any] = {
            "source": source,
            "target": target,
            "type": edge_type,
            "confidence": 1.0,
            "time": time_span,
            "slice_ids": [probe_data.get("probe_id") or probe_data.get("id") or probe_file.stem],
            "train_eval_split": "eval",
        }
        if task_id:
            payload["task_ids"] = [task_id]
        edges.append(payload)

    def intermediate_type_for_edge(edge_type: str, fallback: str) -> str:
        normalized = edge_type.rstrip("?")
        if normalized in {"transmits_through", "transmitsthrough"}:
            return "TransmissionChannel"
        if normalized in {"leaves_legacy_via", "leaveslegacyvia"}:
            return "LegacyAggregate"
        if normalized in {"inherits_from", "inheritsfrom"}:
            return "LineageClaim"
        if normalized in {"contests", "presupposes", "delegitimizes"}:
            return "Claim"
        return fallback

    def seed_node_id(label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", normalize_entity_label(label)).strip("_")
        return f"seed_entity__{slug or 'entity'}"

    for group, label in iter_seed_entity_items(probe_data):
        node_type = seed_entity_node_type(group)
        node_id = seed_node_id(label)
        add_node(
            node_id,
            node_type,
            label=label,
            seed_entity=True,
            seed_entity_group=group,
        )
        seed_ids_by_type.setdefault(node_type, []).append(node_id)

    for task in spec.golden_tasks:
        if not task.start_types or not task.required_edges or not task.target_types:
            continue
        start_type = task.start_types[0]
        target_type = task.target_types[0]
        start_id = next(iter(seed_ids_by_type.get(start_type, [])), f"{task.id}__start")
        target_candidates = seed_ids_by_type.get(target_type, [])
        target_id = (
            target_candidates[-1]
            if target_candidates and target_candidates[-1] != start_id
            else f"{task.id}__end"
        )
        add_node(start_id, start_type)
        add_node(target_id, target_type)
        cursor = start_id
        for idx, edge_type in enumerate(task.required_edges):
            optional = edge_type.endswith("?")
            et = edge_type[:-1] if optional else edge_type
            if idx == len(task.required_edges) - 1:
                add_edge(cursor, target_id, et, task.id)
                cursor = target_id
            else:
                mid_id = f"{task.id}__mid_{idx}"
                mid_type = intermediate_type_for_edge(et, start_type)
                add_node(mid_id, mid_type)
                add_edge(cursor, mid_id, et, task.id)
                cursor = mid_id
        task_edge_indices = [idx for idx, edge in enumerate(edges) if task.id in edge.get("task_ids", [])]
        task_node_ids = sorted(
            {
                str(edge["source"])
                for idx, edge in enumerate(edges)
                if idx in task_edge_indices
            }
            | {
                str(edge["target"])
                for idx, edge in enumerate(edges)
                if idx in task_edge_indices
            }
        )
        task_labels.append(
            {
                "task_id": task.id,
                "label": "golden_task_path",
                "node_ids": task_node_ids,
                "edge_indices": task_edge_indices,
                "split": "eval",
            }
        )
        target_table.append(
            {
                "target_id": f"{task.id}__path_present",
                "name": "path_present",
                "value": True,
                "split": "eval",
                "slice_id": probe_data.get("probe_id") or probe_data.get("id") or probe_file.stem,
                "node_ids": task_node_ids,
                "metadata": {"task_id": task.id},
            }
        )

    graph = {
        "artifact_format": GRAPH_ARTIFACT_FORMAT,
        "probe_id": probe_data.get("probe_id") or probe_data.get("id") or probe_file.stem,
        "schema_version": schema.version,
        "nodes": nodes,
        "edges": edges,
        "task_labels": task_labels,
        "target_table": target_table,
        "metadata": {"builder": "__builtin__", "intended_use": "gnn_candidate_eval"},
    }
    return normalize_graph_artifact(
        probe_id=str(graph["probe_id"]),
        schema_version=schema.version,
        graph=graph,
    ).model_dump(mode="json")


def _build_graph_with_command(
    *,
    command_template: str,
    repo_root: Path,
    probe_file: Path,
    probe_id: str,
    artifact_path: Path,
    schema: GraphSchema,
    show_output: bool,
) -> None:
    values = {
        "probe_file": str(probe_file),
        "probe_id": probe_id,
        "artifact_file": str(artifact_path),
        "schema_json": json.dumps(schema.model_dump(mode="json"), ensure_ascii=False),
    }
    if command_template.strip() == "__builtin__":
        graph = _synthesize_graph_for_probe(probe_file=probe_file, schema=schema)
        artifact_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    command = command_template
    for key, value in values.items():
        command = command.replace(f"{{{key}}}", value)
    command = _ensure_agent_command_permissions(command)
    proc = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if show_output and proc.stdout.strip():
        typer.echo(proc.stdout.strip())
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"graph precompute command failed for {probe_id}: {msg}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"graph precompute command did not create {artifact_path}")


def precompute_graph_artifacts(
    *,
    probe_dir: Path,
    schema: GraphSchema,
    artifact_dir: Path,
    command_template: str,
    builder_version: str,
    probe_ids: set[str] | None = None,
    force: bool = False,
    show_output: bool = False,
    repo_root: Path | None = None,
    wikidata_enrich: bool = False,
    wikidata_cache_path: Path | None = None,
) -> list[ArtifactStatus]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root = repo_root or Path.cwd()
    statuses: list[ArtifactStatus] = []
    schema_fp = schema_fingerprint(schema)
    resolved_wikidata_cache_path = wikidata_cache_path or (root / DEFAULT_WIKIDATA_CACHE_PATH)
    for probe_file in iter_probe_files(probe_dir, probe_ids=probe_ids):
        probe_data = load_probe_yaml(probe_file)
        probe_id = _probe_id(probe_file, probe_data)
        artifact_path, manifest_path = _paths_for_probe(artifact_dir, probe_id)
        status = get_artifact_status(
            artifact_dir=artifact_dir,
            probe_file=probe_file,
            probe_data=probe_data,
            schema=schema,
            builder_version=builder_version,
        )
        should_build = force or (not status.fresh)
        if wikidata_enrich and status.fresh:
            try:
                manifest = _read_json(manifest_path)
                should_build = "wikidata_enrichment" not in manifest
            except Exception:
                should_build = True
        if should_build:
            _build_graph_with_command(
                command_template=command_template,
                repo_root=root,
                probe_file=probe_file,
                probe_id=probe_id,
                artifact_path=artifact_path,
                schema=schema,
                show_output=show_output,
            )
            wikidata_meta: dict[str, Any] | None = None
            if wikidata_enrich:
                graph = load_graph_artifact(artifact_path)
                graph = enrich_graph_with_wikidata(
                    probe_data,
                    graph,
                    cache_path=resolved_wikidata_cache_path,
                )
                artifact_path.write_text(
                    json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                enrichment = graph.get("wikidata_enrichment")
                if isinstance(enrichment, dict):
                    wikidata_meta = enrichment
            manifest = {
                "probe_id": probe_id,
                "probe_file": str(probe_file),
                "artifact_file": str(artifact_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "schema_fingerprint": schema_fp,
                "probe_fingerprint": probe_fingerprint(probe_data),
                "builder_version": builder_version,
            }
            if wikidata_meta is not None:
                manifest["wikidata_enrichment"] = wikidata_meta
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            status = ArtifactStatus(
                probe_id=probe_id,
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                fresh=True,
                reason="rebuilt",
            )
        statuses.append(status)
    return statuses


def ensure_probe_graph_artifacts(
    *,
    probe_dir: Path,
    schema: GraphSchema,
    artifact_dir: Path,
    command_template: str,
    builder_version: str,
    probe_ids: set[str] | None = None,
    show_output: bool = False,
    repo_root: Path | None = None,
    wikidata_enrich: bool = False,
    wikidata_cache_path: Path | None = None,
) -> list[ArtifactStatus]:
    return precompute_graph_artifacts(
        probe_dir=probe_dir,
        schema=schema,
        artifact_dir=artifact_dir,
        command_template=command_template,
        builder_version=builder_version,
        probe_ids=probe_ids,
        force=False,
        show_output=show_output,
        repo_root=repo_root,
        wikidata_enrich=wikidata_enrich,
        wikidata_cache_path=wikidata_cache_path,
    )


def main(
    schema_spec: str = typer.Option("schemas.base_schema", "--schema"),
    probe_dir: Path = typer.Option(Path("probes"), "--probe-dir", exists=True, file_okay=False),
    artifact_dir: Path = typer.Option(
        Path("autoresearch/experiments/graph_cache"), "--artifact-dir", file_okay=False
    ),
    command_template: str = typer.Option(
        ...,
        "--graph-cmd",
        help=(
            "Command that writes probe graph JSON. Supports {probe_file}, {probe_id}, {artifact_file}, {schema_json}."
        ),
    ),
    builder_version: str = typer.Option("v1", "--builder-version"),
    force: bool = typer.Option(False, "--force", help="Rebuild all artifacts regardless of freshness."),
    show_output: bool = typer.Option(False, "--show-output"),
    wikidata_enrich: bool = typer.Option(
        False,
        "--wikidata-enrich",
        help="Resolve probe seed_entities through Wikidata wbsearchentities and write QIDs into graph artifacts.",
    ),
    wikidata_cache_path: Path = typer.Option(
        DEFAULT_WIKIDATA_CACHE_PATH,
        "--wikidata-cache-path",
        help="Shared local JSON cache for Wikidata search results.",
    ),
) -> None:
    schema = load_graph_schema(schema_spec)
    statuses = precompute_graph_artifacts(
        probe_dir=probe_dir,
        schema=schema,
        artifact_dir=artifact_dir,
        command_template=command_template,
        builder_version=builder_version,
        force=force,
        show_output=show_output,
        repo_root=Path.cwd(),
        wikidata_enrich=wikidata_enrich,
        wikidata_cache_path=wikidata_cache_path,
    )
    payload = [
        {
            "probe_id": s.probe_id,
            "artifact_path": str(s.artifact_path),
            "manifest_path": str(s.manifest_path),
            "fresh": s.fresh,
            "reason": s.reason,
        }
        for s in statuses
    ]
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
