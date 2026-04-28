from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.graph_builder_contract6 import PipelineOutput, _gate_variation_check, run_contract6
from baselines.stage1_probe_corpus import Stage1ProbeCorpus


def test_pipeline_output_schema_ranges() -> None:
    out = PipelineOutput(
        probe_id="p1",
        query_as_of="2011-01-01",
        retrieved_node_ids=["n1"],
        retrieved_node_scores=[0.5],
        tier_counts={"actor_state": 1, "trend_thread": 0, "historical_analogue": 0},
        gate_persistence=0.1,
        gate_propagation=0.2,
        gate_precursor=0.3,
        gate_suppression=0.4,
        gate_coordination=0.5,
        forecast_probability=0.6,
        forecast_label_description="desc",
        hint_resolution_rate=1.0,
        bag_embedding_norm=1.0,
    )
    assert 0.0 <= out.gate_persistence <= 1.0
    assert 0.0 <= out.gate_propagation <= 1.0
    assert 0.0 <= out.gate_precursor <= 1.0
    assert 0.0 <= out.gate_suppression <= 1.0
    assert 0.0 <= out.gate_coordination <= 1.0
    assert 0.0 <= out.forecast_probability <= 1.0
    assert 0.0 <= out.hint_resolution_rate <= 1.0
    assert out.bag_embedding_norm >= 0.0


def test_gate_variation_check_monotonic_pass() -> None:
    all_probes = Stage1ProbeCorpus.arab_spring_default().probes
    p_persistence = next(p for p in all_probes if p.assumption_emphasis.value == "Persistence")
    p_precursor = next(p for p in all_probes if p.assumption_emphasis.value == "Precursor")
    p_coord = next(p for p in all_probes if p.assumption_emphasis.value == "Coordination")
    p_supp = next(p for p in all_probes if p.assumption_emphasis.value == "Suppression")
    probes = [p_persistence, p_precursor, p_coord, p_supp]

    outputs = [
        PipelineOutput(probe_id=probes[0].probe_id, query_as_of="2011-01-01", retrieved_node_ids=[], retrieved_node_scores=[], tier_counts={"actor_state": 0, "trend_thread": 0, "historical_analogue": 0}, gate_persistence=0.9, gate_propagation=0.0, gate_precursor=0.0, gate_suppression=0.0, gate_coordination=0.1, forecast_probability=0.5, forecast_label_description="x", hint_resolution_rate=1.0, bag_embedding_norm=1.0),
        PipelineOutput(probe_id=probes[1].probe_id, query_as_of="2011-01-01", retrieved_node_ids=[], retrieved_node_scores=[], tier_counts={"actor_state": 0, "trend_thread": 0, "historical_analogue": 0}, gate_persistence=0.1, gate_propagation=0.0, gate_precursor=0.0, gate_suppression=0.0, gate_coordination=0.1, forecast_probability=0.5, forecast_label_description="x", hint_resolution_rate=1.0, bag_embedding_norm=1.0),
        PipelineOutput(probe_id=probes[2].probe_id, query_as_of="2011-01-01", retrieved_node_ids=[], retrieved_node_scores=[], tier_counts={"actor_state": 0, "trend_thread": 0, "historical_analogue": 0}, gate_persistence=0.1, gate_propagation=0.0, gate_precursor=0.0, gate_suppression=0.0, gate_coordination=0.9, forecast_probability=0.5, forecast_label_description="x", hint_resolution_rate=1.0, bag_embedding_norm=1.0),
        PipelineOutput(probe_id=probes[3].probe_id, query_as_of="2011-01-01", retrieved_node_ids=[], retrieved_node_scores=[], tier_counts={"actor_state": 0, "trend_thread": 0, "historical_analogue": 0}, gate_persistence=0.1, gate_propagation=0.0, gate_precursor=0.0, gate_suppression=0.0, gate_coordination=0.1, forecast_probability=0.5, forecast_label_description="x", hint_resolution_rate=1.0, bag_embedding_norm=1.0),
    ]
    assert _gate_variation_check(outputs, probes) == "pass"


@pytest.mark.integration
def test_contract6_integration_smoke_real_artifacts(tmp_path: Path) -> None:
    manifest = Path('/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/node_warehouse_v1_fixed_manifest.json')
    mmap = Path('/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/node_warehouse_v1_fixed.mmap')
    ckpt = Path('/Users/darenpalmer/conductor/shared-data/psychohistory-v2/arab_spring/checkpoints/stage1_predictive_rankchange_phase3_hotfix_smoke/query_encoder_epoch_000.pt')
    warehouse = Path('/Users/darenpalmer/conductor/shared-data/psychohistory-v2/warehouse/events.duckdb')

    if not (manifest.exists() and mmap.exists() and ckpt.exists() and warehouse.exists()):
        pytest.skip('real artifacts unavailable')

    outdir = tmp_path / 'contract6_smoke'
    summary = run_contract6(
        manifest_path=manifest,
        mmap_path=mmap,
        encoder_checkpoint=ckpt,
        output_dir=outdir,
        warehouse_duckdb=warehouse,
        max_probes=5,
        run_id='contract6_smoke_test',
    )

    assert (outdir / 'contract6_summary.json').exists()
    assert (outdir / 'contract6_outputs.jsonl').exists()
    assert summary['completed'] == 5
    assert summary['failed'] == 0
    assert summary['brier_score'] is None or isinstance(summary['brier_score'], float)

    s = json.loads((outdir / 'contract6_summary.json').read_text(encoding='utf-8'))
    assert s['run_id'] == 'contract6_smoke_test'
