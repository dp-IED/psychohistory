"""CLI and library entrypoint for probe-suite evaluation."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path

import typer

from schemas.schema_registry import load_graph_schema
from schemas.schema_types import GraphSchema

from autoresearch.scoring import compose_score

from evals.objective_eval import evaluate_objective_layer
from evals.probe_graph_precompute import ensure_probe_graph_artifacts
from evals.probe_functional import run_probe_functional
from evals.reporting import (
    EvalReport,
    StubScores,
    build_report,
    classify_failure,
    report_to_jsonable,
)
from evals.schema_constraints import check_projection, run_schema_constraints
from evals.structural_validity import run_structural_validity


def _rate(results: list[object], is_ok: Callable[[object], bool]) -> float:
    if not results:
        return 1.0
    passed = sum(1 for r in results if is_ok(r))
    return passed / len(results)


def _schema_quality_score(schema: GraphSchema) -> float:
    nodes = list(schema.node_types.values())
    edges = list(schema.edge_types.values())
    if not nodes or not edges:
        return 0.0
    node_rich = sum(1 for n in nodes if n.attributes or n.extensions) / len(nodes)
    edge_rich = sum(
        1
        for e in edges
        if e.extensions or (e.allowed_source_layers and e.allowed_target_layers)
    ) / len(edges)
    projection_rich = min(1.0, len(schema.projection.per_type_optional) / 4.0)
    temporal_rich = sum(1 for n in nodes if n.extensions.get("temporal_profile")) / len(nodes)
    return (0.35 * node_rich) + (0.35 * edge_rich) + (0.2 * projection_rich) + (0.1 * temporal_rich)


def evaluate_schema(
    schema: GraphSchema,
    probe_dir: Path,
    weights: dict[str, float] | None = None,
    graph_artifact_dir: Path = Path("autoresearch/experiments/graph_cache"),
    objective_strict_mode: bool = False,
    graph_builder_version: str = "v1",
    traversal_promotion_threshold: float | None = None,
    graph_precompute_cmd: str | None = None,
    auto_refresh_graph_artifacts: bool = False,
    graph_precompute_show_output: bool = False,
    graph_precompute_repo_root: Path | None = None,
) -> EvalReport:
    if objective_strict_mode and auto_refresh_graph_artifacts:
        if not graph_precompute_cmd:
            raise ValueError(
                "objective_strict_mode requires graph_precompute_cmd (or disable auto_refresh_graph_artifacts)"
            )
        ensure_probe_graph_artifacts(
            probe_dir=probe_dir,
            schema=schema,
            artifact_dir=graph_artifact_dir,
            command_template=graph_precompute_cmd,
            builder_version=graph_builder_version,
            show_output=graph_precompute_show_output,
            repo_root=graph_precompute_repo_root or Path.cwd(),
        )
    s_ok, s_res = run_structural_validity(probe_dir, schema)
    f_ok, f_res = run_probe_functional(probe_dir, schema)
    c_res = run_schema_constraints(probe_dir, schema)
    proj_ok = check_projection(schema).ok

    structural_rate = _rate(s_res, lambda r: bool(getattr(r, "ok", False)))
    functional_rate = (
        sum(float(getattr(r, "score", 0.0)) for r in f_res) / len(f_res) if f_res else 1.0
    )
    constraints_rate = 1.0 / (1.0 + len(c_res.issues))
    objective = evaluate_objective_layer(
        probe_dir=probe_dir,
        schema=schema,
        artifact_dir=graph_artifact_dir,
        builder_version=graph_builder_version,
        strict_mode=objective_strict_mode,
    )
    objective_scores = {
        "traversal": objective.traversal_score,
        "discipline": objective.discipline_score,
        "ablation_gain": objective.ablation_gain_score,
    }

    stub = StubScores(schema_quality=_schema_quality_score(schema))
    stub_dict = {k: float(v) for k, v in stub.model_dump(exclude={"notes"}).items()}
    objective_weight_defaults = (
        {"traversal": 0.30, "discipline": 0.15, "ablation_gain": 0.20}
        if objective.applied and any(objective_scores.values())
        else {}
    )
    merged_weights = {**objective_weight_defaults, **(weights or {})}
    composite, parts = compose_score(
        structural_rate,
        functional_rate,
        constraints_rate,
        stub_dict,
        merged_weights,
        objective_scores=objective_scores,
        traversal_promotion_threshold=traversal_promotion_threshold,
    )
    objective_ok = objective.ok if objective_strict_mode else True

    fc = classify_failure(
        eval_crashed=False,
        structural_ok=s_ok,
        functional_ok=f_ok,
        constraints_ok=c_res.ok,
        incompatible_projection=not proj_ok,
        objective_ok=objective_ok,
        score_regression=False,
    )

    return build_report(
        schema=schema,
        structural_ok=s_ok,
        structural=s_res,
        functional_ok=f_ok,
        functional=f_res,
        constraints=c_res,
        stub=stub,
        weights=merged_weights,
        composite_score=composite,
        traversal_score=objective.traversal_score,
        discipline_score=objective.discipline_score,
        ablation_gain_score=objective.ablation_gain_score,
        failure_class=fc,
        meta={
            "score_parts": parts,
            "gate_rates": {
                "structural": structural_rate,
                "functional": functional_rate,
                "constraints": constraints_rate,
                "traversal": objective.traversal_score,
                "discipline": objective.discipline_score,
                "ablation_gain": objective.ablation_gain_score,
            },
            "objective_v1": {
                "strict_mode": objective_strict_mode,
                "graph_artifact_dir": str(graph_artifact_dir),
                "graph_builder_version": graph_builder_version,
                "applied": objective.applied,
                "ok": objective.ok,
                "errors": objective.errors,
                "probes": [
                    {
                        "probe_id": p.probe_id,
                        "traversal_score": p.traversal_score,
                        "discipline_score": p.discipline_score,
                        "ablation_gain_score": p.ablation_gain_score,
                        "errors": p.errors,
                        "diagnostics": p.diagnostics,
                        "tasks": [
                            {
                                "task_id": t.task_id,
                                "ok": t.ok,
                                "score": t.score,
                                "path_found": t.path_found,
                                "matched_paths": t.matched_paths,
                                "best_path_len": t.best_path_len,
                                "missing_layers": t.missing_layers,
                                "missing_answers": t.missing_answers,
                                "notes": t.notes,
                            }
                            for t in p.task_results
                        ],
                    }
                    for p in objective.probes
                ],
            },
        },
    )


def run_eval_cli(
    schema_spec: str,
    probe_dir: Path,
    weights: dict[str, float] | None = None,
    out: Path | None = None,
    graph_artifact_dir: Path = Path("autoresearch/experiments/graph_cache"),
    objective_strict_mode: bool = True,
    graph_builder_version: str = "v1",
    traversal_promotion_threshold: float | None = None,
    graph_precompute_cmd: str | None = "__builtin__",
    auto_refresh_graph_artifacts: bool = True,
    graph_precompute_show_output: bool = False,
    graph_precompute_repo_root: Path | None = None,
) -> EvalReport:
    schema = load_graph_schema(schema_spec)
    report = evaluate_schema(
        schema,
        probe_dir,
        weights=weights,
        graph_artifact_dir=graph_artifact_dir,
        objective_strict_mode=objective_strict_mode,
        graph_builder_version=graph_builder_version,
        traversal_promotion_threshold=traversal_promotion_threshold,
        graph_precompute_cmd=graph_precompute_cmd,
        auto_refresh_graph_artifacts=auto_refresh_graph_artifacts,
        graph_precompute_show_output=graph_precompute_show_output,
        graph_precompute_repo_root=graph_precompute_repo_root,
    )
    payload = report_to_jsonable(report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(
    schema_spec: str = typer.Option(
        "schemas.base_schema",
        "--schema",
        help="Python module or path to schema file exporting GRAPH_SCHEMA or get_seed_graph_schema()",
    ),
    probe_dir: Path = typer.Option(Path("probes"), "--probe-dir", exists=True, file_okay=False, dir_okay=True),
    weights: str | None = typer.Option(None, "--weights", help="JSON object of score weights"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON report to this path"),
    graph_artifact_dir: Path = typer.Option(
        Path("autoresearch/experiments/graph_cache"),
        "--graph-artifact-dir",
        help="Directory with precomputed per-probe graph artifacts and manifests.",
    ),
    objective_strict_mode: bool = typer.Option(
        True,
        "--objective-strict-mode/--objective-lenient-mode",
        help="Fail eval when objective fields/artifacts are missing or stale.",
    ),
    graph_builder_version: str = typer.Option(
        "v1",
        "--graph-builder-version",
        help="Expected graph builder version in artifact manifests.",
    ),
    traversal_promotion_threshold: float | None = typer.Option(
        None,
        "--traversal-promotion-threshold",
        help="If set, external stub contribution is blocked unless traversal reaches this threshold.",
    ),
    graph_precompute_cmd: str | None = typer.Option(
        "__builtin__",
        "--graph-precompute-cmd",
        help=(
            "Command to refresh stale graph artifacts before eval. "
            "Use '__builtin__' for deterministic synthesis, or a shell command with "
            "{probe_file}, {probe_id}, {artifact_file}, {schema_json}."
        ),
    ),
    auto_refresh_graph_artifacts: bool = typer.Option(
        True,
        "--auto-refresh-graph-artifacts/--no-auto-refresh-graph-artifacts",
        help="Rebuild graph artifacts when fingerprints disagree with the current schema/probe YAML.",
    ),
    graph_precompute_show_output: bool = typer.Option(
        False,
        "--graph-precompute-show-output",
        help="Print stdout from graph precompute commands.",
    ),
) -> None:
    w: dict[str, float] | None = None
    if weights:
        parsed = json.loads(weights)
        w = {str(k): float(v) for k, v in parsed.items()}

    try:
        report = run_eval_cli(
            schema_spec,
            probe_dir,
            weights=w,
            out=out,
            graph_artifact_dir=graph_artifact_dir,
            objective_strict_mode=objective_strict_mode,
            graph_builder_version=graph_builder_version,
            traversal_promotion_threshold=traversal_promotion_threshold,
            graph_precompute_cmd=graph_precompute_cmd,
            auto_refresh_graph_artifacts=auto_refresh_graph_artifacts,
            graph_precompute_show_output=graph_precompute_show_output,
        )
    except Exception:
        print(traceback.format_exc())
        raise typer.Exit(code=1)

    typer.echo(json.dumps(report_to_jsonable(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
