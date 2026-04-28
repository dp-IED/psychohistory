from __future__ import annotations

from datetime import date
import json

import numpy as np
import pytest

from baselines.stage1_predictive_targets import build_predictive_targets
from baselines.stage1_predictive_targets import load_predictive_targets
from baselines.stage1_predictive_tuples import build_predictive_training_tuples
from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta


def _manifest() -> NodeWarehouseManifest:
    rows = [
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|US|2011-01", admin1_code="US", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_d|SY|2011-03", admin1_code="SY", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_e|EG|2011-02", admin1_code="EG", first_seen=date(2011, 2, 10)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_f|US|2011-06", admin1_code="US", first_seen=date(2011, 6, 1)),
    ]
    return NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=len(rows),
        rows=rows,
    )


def _probe() -> ProbeRecord:
    return ProbeRecord(
        probe_id="p-eg-1",
        origin=date(2011, 3, 5),
        nl_text="Egypt unrest",
        q_struct=QStructV0(
            actor_state=ActorStateQuery(
                geography=["egypt-cairo"],
                actor_type=["group"],
                as_of=date(2011, 3, 5),
            )
        ),
        lens_params=LensParamsV0(horizon_days=30),
        assumption_emphasis=AssumptionEmphasis.PRECURSOR,
        generation_meta=GenerationMeta(template_id="t", generator_version="v", assumption_gate_coverage=AssumptionEmphasis.PRECURSOR),
    )


def _write_custom_targets(manifest: NodeWarehouseManifest, out_dir, *, h30: np.ndarray, h60: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rankchange_h30d.npy", np.asarray(h30, dtype=np.float32))
    np.save(out_dir / "rankchange_h60d.npy", np.asarray(h60, dtype=np.float32))
    meta = {
        "manifest_version": "stage1_predictive_targets_v1",
        "objective_version": "predictive_coding_rankchange_v1",
        "embedding_version": manifest.embedding_version,
        "source_manifest_path": "/tmp/manifest.json",
        "row_count": manifest.row_count,
        "horizons_days": [30, 60],
        "horizon_weights": [1.0, 0.5],
        "target_files": {"30": "rankchange_h30d.npy", "60": "rankchange_h60d.npy"},
        "actor_missing_rows": 0,
        "rows_without_first_seen": 0,
    }
    meta_path = out_dir / "stage1_predictive_targets.rankchange_v1.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def test_predictive_tuples_deterministic(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30, 60],
        horizon_weights=[1.0, 0.5],
    )
    probes = [_probe()]

    tuples_a, meta_a = build_predictive_training_tuples(probes, manifest, meta_path, seed=123, max_neg_per_stratum=4)
    tuples_b, meta_b = build_predictive_training_tuples(probes, manifest, meta_path, seed=123, max_neg_per_stratum=4)

    assert tuples_a == tuples_b
    assert meta_a == meta_b


def test_predictive_tuples_include_required_negative_strata(tmp_path):
    manifest = _manifest()
    # EG indices: 0,1,2,3,6. Make idx 6 wrong-horizon at h=30 (low at 30, high at 60).
    meta_path = _write_custom_targets(
        manifest,
        tmp_path,
        h30=np.array([0.8, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        h60=np.array([0.2, 0.1, 0.0, 0.4, 0.0, 0.0, 0.9, 0.0], dtype=np.float32),
    )
    tuples, _meta = build_predictive_training_tuples([_probe()], manifest, meta_path, seed=7, max_neg_per_stratum=8)

    assert any(int(t["strata_counts"]["same_time_wrong_domain"]) > 0 for t in tuples)
    assert any(int(t["strata_counts"]["same_domain_wrong_horizon"]) > 0 for t in tuples)
    assert any(int(t["strata_counts"]["same_region_non_precursor"]) > 0 for t in tuples)


def test_wrong_horizon_negatives_are_positive_in_other_horizon(tmp_path):
    manifest = _manifest()
    meta_path = _write_custom_targets(
        manifest,
        tmp_path,
        h30=np.array([0.8, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        h60=np.array([0.2, 0.1, 0.0, 0.4, 0.0, 0.0, 0.9, 0.0], dtype=np.float32),
    )
    meta, targets = load_predictive_targets(meta_path, manifest)
    tuples, _meta = build_predictive_training_tuples([_probe()], manifest, meta_path, seed=7, max_neg_per_stratum=8)

    seen = 0
    for t in tuples:
        h = int(t["horizon_days"])
        wrong_horizon = t["strata_indices"]["same_domain_wrong_horizon"]
        for i in wrong_horizon:
            seen += 1
            assert float(targets[h][i]) <= 1e-6
            assert any(float(targets[oh][i]) > 1e-6 for oh in meta.horizons_days if oh != h)
    assert seen > 0


def test_strata_disjoint_after_overlap_removal(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30, 60],
        horizon_weights=[1.0, 0.5],
    )
    tuples, _meta = build_predictive_training_tuples([_probe()], manifest, meta_path, seed=11, max_neg_per_stratum=8)
    for t in tuples:
        s = t["strata_indices"]
        wd = set(s["same_time_wrong_domain"])
        wh = set(s["same_domain_wrong_horizon"])
        np0 = set(s["same_region_non_precursor"])
        assert wd.isdisjoint(wh)
        assert wd.isdisjoint(np0)
        assert wh.isdisjoint(np0)


def test_predictive_tuple_metadata_counts_integrity(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30, 60],
        horizon_weights=[1.0, 0.5],
    )
    probes = [_probe(), _probe().model_copy(update={"probe_id": "p-eg-2"})]
    tuples, metadata = build_predictive_training_tuples(probes, manifest, meta_path, seed=5, max_neg_per_stratum=8)

    assert metadata["tuple_count"] == len(tuples)
    assert sum(metadata["per_horizon_tuple_counts"].values()) == len(tuples)

    computed = {
        "same_time_wrong_domain": 0,
        "same_domain_wrong_horizon": 0,
        "same_region_non_precursor": 0,
    }
    for t in tuples:
        strata = t["strata_counts"]
        for k in computed:
            computed[k] += int(strata[k])
    assert computed == metadata["per_stratum_negative_counts"]

    empty_count = sum(1 for t in tuples if (len(t["pos_global_indices"]) == 0 or len(t["neg_global_indices"]) == 0))
    assert empty_count == metadata["empty_tuple_count"]
    assert metadata["empty_tuple_rate"] == (float(empty_count) / float(len(tuples)))


def test_same_time_wrong_domain_enforces_first_seen_le_as_of(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30, 60],
        horizon_weights=[1.0, 0.5],
    )
    tuples, _ = build_predictive_training_tuples([_probe()], manifest, meta_path, seed=9, max_neg_per_stratum=16)
    as_of = _probe().q_struct.actor_state.as_of
    for t in tuples:
        for i in t["strata_indices"]["same_time_wrong_domain"]:
            assert manifest.rows[i].first_seen is not None
            assert manifest.rows[i].first_seen <= as_of


def test_invalid_caps_raise_value_error(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30, 60],
        horizon_weights=[1.0, 0.5],
    )
    with pytest.raises(ValueError, match="max_pos_per_tuple"):
        build_predictive_training_tuples([_probe()], manifest, meta_path, max_pos_per_tuple=0)
    with pytest.raises(ValueError, match="max_neg_per_stratum"):
        build_predictive_training_tuples([_probe()], manifest, meta_path, max_neg_per_stratum=0)
