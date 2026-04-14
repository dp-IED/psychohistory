from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.logging import ExperimentRecord, append_experiment
from evals.objective_eval import evaluate_objective_layer
from evals.objective_specs import parse_objective_spec
from evals.probe_graph_precompute import probe_fingerprint, schema_fingerprint
from autoresearch.runner import (
    _apply_noop_penalty,
    _candidate_change_info,
    _apply_adversarial_result,
    _ensure_agent_command_permissions,
    _format_commit_message,
    _parse_json_from_stdout,
    _resolve_schema_source_root,
    _run_adversarial_command,
    _run_proposal_command,
    _score_delta,
)
from autoresearch.config import HarnessPaths
from autoresearch.candidate_manager import prepare_candidate_tree
from autoresearch.scoring import compose_from_eval_report, compose_score
from evals.structural_validity import (
    check_probe_structural,
    extract_must_represent,
    iter_probe_files,
    load_probe_yaml,
)
from schemas.base_schema import get_seed_graph_schema
from schemas.schema_registry import load_graph_schema
from schemas.schema_types import EdgeSpec, GraphSchema, Layer, NodeSpec, ProjectionSpec
from evals.run_eval import evaluate_schema, run_eval_cli


def test_load_one_probe(repo_root: Path):
    probes = iter_probe_files(repo_root / "probes")
    assert probes, "expected at least one probe-*.yaml"
    data = load_probe_yaml(probes[0])
    assert isinstance(data, dict)


def test_extract_must_represent_supports_flat_schema_requirement_keys(repo_root: Path):
    probe = load_probe_yaml(repo_root / "probes" / "probe-10-panafricanism.yaml")
    req = extract_must_represent(probe)
    assert req is not None
    assert "Narrative" in req["node_types"]
    assert "transmitsthrough" in req["edge_types"]


def test_structural_check_accepts_edge_alias_spelling_variants(repo_root: Path):
    schema = load_graph_schema("schemas.base_schema")
    for probe_name in (
        "probe-8-protestant-reformation.yaml",
        "probe-9-liberation-theology.yaml",
        "probe-10-panafricanism.yaml",
        "probe-11-hindutva.yaml",
        "probe-12-ukraine.yaml",
        "probe-13-sudan.yaml",
    ):
        result = check_probe_structural(repo_root / "probes" / probe_name, schema)
        assert result.ok, f"{probe_name}: {result.errors}"


def test_instantiate_base_schema():
    s = get_seed_graph_schema()
    assert s.projection.dim >= 1
    assert s.node_types


def test_schema_registry_module():
    s = load_graph_schema("schemas.base_schema")
    assert s.name


def test_noop_eval(repo_root: Path):
    s = load_graph_schema("schemas.base_schema")
    r = evaluate_schema(s, repo_root / "probes", objective_strict_mode=False)
    payload = r.model_dump()
    assert "composite_score" in payload
    assert payload["structural_ok"] in (True, False)
    assert "gate_rates" in payload.get("meta", {})


def test_objective_spec_parser_extracts_core_fields():
    raw = {
        "probe_id": "probe-x",
        "golden_tasks": [
            {
                "id": "t1",
                "start_types": ["Claim"],
                "required_edges": ["presupposes"],
                "target_types": ["Claim"],
                "min_hops": 1,
                "must_cross_layers": ["epistemic"],
            }
        ],
        "designated_ablations": [{"name": "epistemic_off", "for_tasks": ["t1"]}],
        "complexity_budget": {"max_node_types_used": 3},
    }
    spec = parse_objective_spec(raw, "probe-x")
    assert spec.is_valid
    assert spec.golden_tasks[0].id == "t1"
    assert spec.designated_ablations[0].name == "epistemic_off"


def _write_probe_and_artifact(
    tmp_path: Path, probe_data: dict, schema: GraphSchema, graph_payload: dict, builder_version: str = "v1"
) -> tuple[Path, Path]:
    probe_dir = tmp_path / "probes"
    artifact_dir = tmp_path / "graph_cache"
    probe_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    probe_id = str(probe_data.get("probe_id"))
    probe_file = probe_dir / f"{probe_id}.yaml"
    full_probe_data = {
        "probe_id": probe_id,
        "golden_tasks": [
            {
                "id": "chain",
                "start_types": ["Claim"],
                "required_edges": ["presupposes"],
                "target_types": ["Claim"],
                "min_hops": 1,
                "must_cross_layers": ["epistemic"],
            }
        ],
        "designated_ablations": [{"name": "epistemic_off", "for_tasks": ["chain"]}],
        "complexity_budget": {"max_node_types_used": 3, "max_edge_types_used": 3},
    }
    probe_file.write_text(json.dumps(full_probe_data, indent=2) + "\n", encoding="utf-8")

    artifact_path = artifact_dir / f"{probe_id}.graph.json"
    artifact_path.write_text(json.dumps(graph_payload, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "probe_id": probe_id,
        "probe_file": str(probe_file),
        "artifact_file": str(artifact_path),
        "schema_fingerprint": schema_fingerprint(schema),
        "probe_fingerprint": probe_fingerprint(full_probe_data),
        "builder_version": builder_version,
    }
    (artifact_dir / f"{probe_id}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return probe_dir, artifact_dir


def _mini_schema(extra_edges: int = 0) -> GraphSchema:
    edges = {"presupposes": EdgeSpec(name="presupposes")}
    for idx in range(extra_edges):
        edges[f"unused_edge_{idx}"] = EdgeSpec(name=f"unused_edge_{idx}")
    return GraphSchema(
        name="mini",
        version="0.0.1",
        projection=ProjectionSpec(dim=8, required=True),
        node_types={
            "Claim": NodeSpec(name="Claim", primary_layer=Layer.EPISTEMIC),
            **{f"UnusedType{idx}": NodeSpec(name=f"UnusedType{idx}", primary_layer=Layer.MATERIAL) for idx in range(extra_edges)},
        },
        edge_types=edges,
    )


def test_objective_eval_penalizes_registry_breadth_without_task_lift(tmp_path: Path):
    probe_data = {"probe_id": "probe-mini"}
    graph_payload = {
        "nodes": [
            {"id": "c1", "type": "Claim", "layer": "epistemic"},
            {"id": "c2", "type": "Claim", "layer": "epistemic"},
        ],
        "edges": [{"source": "c1", "target": "c2", "type": "presupposes"}],
    }
    baseline = _mini_schema(extra_edges=0)
    broad = _mini_schema(extra_edges=6)
    probe_dir, artifact_dir = _write_probe_and_artifact(tmp_path, probe_data, baseline, graph_payload)

    baseline_result = evaluate_objective_layer(
        probe_dir=probe_dir,
        schema=baseline,
        artifact_dir=artifact_dir,
        builder_version="v1",
        strict_mode=True,
    )
    assert baseline_result.ok
    assert baseline_result.discipline_score > 0.0

    # Re-write manifest to match broadened schema while graph stays identical.
    manifest_path = artifact_dir / "probe-mini.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_fingerprint"] = schema_fingerprint(broad)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    broad_result = evaluate_objective_layer(
        probe_dir=probe_dir,
        schema=broad,
        artifact_dir=artifact_dir,
        builder_version="v1",
        strict_mode=True,
    )
    assert broad_result.ok
    assert broad_result.discipline_score < baseline_result.discipline_score


def test_run_eval_cli_strict_mode_refreshes_graph_artifacts(tmp_path: Path):
    probe_dir = tmp_path / "probes"
    artifact_dir = tmp_path / "graph_cache"
    probe_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    probe_payload = {
        "probe_id": "probe-mini-cli",
        "schema_requirements": {
            "must_represent_node_types": ["Claim", "Narrative", "Institution"],
            "must_represent_edges": ["contests", "supports"],
        },
        "golden_tasks": [
            {
                "id": "chain",
                "start_types": ["Claim"],
                "required_edges": ["presupposes"],
                "target_types": ["Claim"],
                "min_hops": 1,
                "must_cross_layers": ["epistemic"],
            }
        ],
        "designated_ablations": [{"name": "epistemic_off", "for_tasks": ["chain"]}],
        "complexity_budget": {"max_node_types_used": 3, "max_edge_types_used": 3},
    }
    (probe_dir / "probe-mini-cli.yaml").write_text(json.dumps(probe_payload, indent=2) + "\n", encoding="utf-8")

    report = run_eval_cli(
        "schemas.base_schema",
        probe_dir,
        weights=None,
        out=None,
        graph_artifact_dir=artifact_dir,
        objective_strict_mode=True,
        graph_builder_version="v1",
        traversal_promotion_threshold=None,
        graph_precompute_cmd="__builtin__",
        auto_refresh_graph_artifacts=True,
        graph_precompute_show_output=False,
    )
    assert report.failure_class is None
    assert (artifact_dir / "probe-mini-cli.graph.json").exists()
    assert (artifact_dir / "probe-mini-cli.manifest.json").exists()


def test_experiment_log_append(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    rec = ExperimentRecord(
        iteration=0,
        candidate_id="test_cand",
        timestamp_utc="",
        composite_score=0.42,
        proposal={"rationale": "test"},
    )
    append_experiment(log, rec)
    line = log.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["composite_score"] == 0.42


def test_prepare_candidate_tree_uses_custom_schema_source(tmp_path: Path):
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "seed_source"
    (repo_root / "schemas").mkdir(parents=True, exist_ok=True)
    (source_root / "schemas").mkdir(parents=True, exist_ok=True)
    (repo_root / "schemas" / "base_schema.py").write_text("REPO_SCHEMA = True\n", encoding="utf-8")
    (source_root / "schemas" / "base_schema.py").write_text("CUSTOM_SCHEMA = True\n", encoding="utf-8")

    candidate_root = prepare_candidate_tree(
        root=repo_root,
        candidate_id="cand_test_seed",
        seed_schema_spec="schemas.base_schema",
        schema_source_root=source_root,
    )

    copied = (candidate_root / "schemas" / "base_schema.py").read_text(encoding="utf-8")
    assert "CUSTOM_SCHEMA = True" in copied
    assert "REPO_SCHEMA = True" not in copied


def test_resolve_schema_source_root_prefers_best_latest(tmp_path: Path):
    root = tmp_path / "repo"
    best_latest = root / "autoresearch" / "experiments" / "best" / "latest" / "schemas"
    best_latest.mkdir(parents=True, exist_ok=True)
    paths = HarnessPaths.default(root)

    chosen = _resolve_schema_source_root(paths, seed_from_best=True)
    assert chosen == root / "autoresearch" / "experiments" / "best" / "latest"


def test_resolve_schema_source_root_falls_back_to_repo(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    paths = HarnessPaths.default(root)

    chosen = _resolve_schema_source_root(paths, seed_from_best=True)
    assert chosen == root


def test_compose_score_allows_partial_gate_progress():
    strong, _ = compose_score(1.0, 1.0, 1.0, {}, None)
    mixed, parts = compose_score(0.5, 1.0, 1.0, {}, None)
    assert mixed < strong
    assert parts["structural"] == 0.5


def test_run_proposal_command_writes_proposal(tmp_path: Path):
    proposal_path = tmp_path / "proposal.json"
    command = (
        "python -c \"import pathlib; pathlib.Path(r'{proposal_file}').write_text("
        "'{\\\"rationale\\\":\\\"cmd\\\",\\\"file_patch_plan\\\":[]}', encoding='utf-8')\""
    )
    _run_proposal_command(
        command_template=command,
        proposal_file=proposal_path,
        iteration=3,
        prior_best=0.5,
        candidate_id="cand_test",
        root=tmp_path,
        show_output=False,
    )
    assert proposal_path.exists()


def test_ensure_agent_command_permissions_injects_force_and_trust():
    cmd = "agent -p 'hello world'"
    out = _ensure_agent_command_permissions(cmd)
    assert "--trust" in out
    assert "--force" in out


def test_ensure_agent_command_permissions_keeps_non_agent_command():
    cmd = "python -m something"
    out = _ensure_agent_command_permissions(cmd)
    assert out == cmd


def test_score_delta_handles_first_iteration():
    assert _score_delta(float("-inf"), 0.84) == 0.84


def test_format_commit_message_contains_delta_and_score():
    msg = _format_commit_message("cand_abc", 0.80, 0.85)
    assert "cand_abc" in msg
    assert "0.8500" in msg
    assert "+0.0500" in msg


def test_parse_json_from_stdout_uses_last_json_line():
    payload = _parse_json_from_stdout("noise line\n{\"adversarial_score\": 0.7}")
    assert payload["adversarial_score"] == 0.7


def test_run_adversarial_command_parses_output(tmp_path: Path):
    cmd = "python -c \"print('{\\\"adversarial_score\\\": 0.9, \\\"summary\\\": \\\"ok\\\"}')\""
    payload = _run_adversarial_command(
        command_template=cmd,
        iteration=0,
        prior_best=0.5,
        candidate_id="cand_test",
        root=tmp_path,
        candidate_root=tmp_path,
        schema_spec="schemas.base_schema",
        probe_dir=tmp_path,
        show_output=False,
    )
    assert payload["adversarial_score"] == 0.9


def test_run_adversarial_command_rejects_invalid_payload(tmp_path: Path):
    cmd = "python -c \"print('{\\\"summary\\\": \\\"missing score\\\"}')\""
    with pytest.raises(ValueError):
        _run_adversarial_command(
            command_template=cmd,
            iteration=0,
            prior_best=0.5,
            candidate_id="cand_test",
            root=tmp_path,
            candidate_root=tmp_path,
            schema_spec="schemas.base_schema",
            probe_dir=tmp_path,
            show_output=False,
        )


def test_apply_adversarial_result_updates_composite():
    report = {
        "structural_ok": True,
        "functional_ok": True,
        "constraints_ok": True,
        "stub_scores": {
            "gdelt_coverage": 0.0,
            "synthetic_traversal_qa": 0.0,
            "polymarket_brier": 0.0,
            "persistence_layer_ablation": 0.0,
        },
        "meta": {"gate_rates": {"structural": 0.6, "functional": 1.0, "constraints": 1.0}},
    }
    updated, composite = _apply_adversarial_result(report, {"adversarial_score": 1.0, "summary": "stress pass"})
    assert updated["stub_scores"]["adversarial_agent"] == 1.0
    assert updated["meta"]["adversarial_eval"]["summary"] == "stress pass"
    recomputed, parts = compose_from_eval_report(updated)
    assert composite == recomputed
    assert parts["structural"] == 0.6


def test_compose_from_eval_report_ignores_non_numeric_stub_fields():
    report = {
        "structural_ok": True,
        "functional_ok": True,
        "constraints_ok": True,
        "stub_scores": {"gdelt_coverage": 0.0, "notes": "non-numeric"},
        "meta": {"gate_rates": {"structural": 1.0, "functional": 1.0, "constraints": 1.0}},
    }
    composite, parts = compose_from_eval_report(report)
    assert composite == 0.85
    assert parts["external_stub"] == 0.0


def test_apply_noop_penalty_reduces_score_when_unchanged():
    report = {"composite_score": 0.85, "meta": {}}
    updated, score = _apply_noop_penalty(report, 0.85, changed=False, penalty=0.01)
    assert score == 0.84
    assert updated["meta"]["noop_penalty"]["applied"] is True


def test_candidate_change_info_detects_change(tmp_path: Path):
    root = tmp_path / "cand"
    (root / "schemas_before").mkdir(parents=True)
    (root / "schemas").mkdir(parents=True)
    (root / "schemas_before" / "base_schema.py").write_text("A\n", encoding="utf-8")
    (root / "schemas" / "base_schema.py").write_text("B\n", encoding="utf-8")
    info = _candidate_change_info(root)
    assert info.changed is True


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
