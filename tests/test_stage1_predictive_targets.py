from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from baselines.stage1_predictive_targets import build_predictive_targets, load_predictive_targets
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta


def _manifest() -> NodeWarehouseManifest:
    rows = [
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
    ]
    return NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=len(rows),
        rows=rows,
    )


def _centrality_manifest() -> NodeWarehouseManifest:
    rows = [
        # t0: actors a,b,c,d all present in EG; complete co-activation means equal
        # centrality and lexical tie-break ranks a=1,b=2,c=3,d=4.
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_d|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        # t1: only b,c,d remain in EG; all equal centrality, ranks b=1,c=2,d=3.
        # actor_d improves 4->3 so positive normalized gain 1/4.
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_d|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 5)),
        # Another admin1 to ensure slicing is admin1-local.
        NodeWarehouseRowMeta(node_id="ar_v1|actor_x|US|2011-01", admin1_code="US", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_y|US|2011-03", admin1_code="US", first_seen=date(2011, 3, 5)),
    ]
    return NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=len(rows),
        rows=rows,
    )


def _nearest_future_manifest() -> NodeWarehouseManifest:
    rows = [
        # t0
        NodeWarehouseRowMeta(node_id="ar_v1|actor_a|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|EG|2011-01", admin1_code="EG", first_seen=date(2011, 1, 1)),
        # t1 candidate (nearest >= t0+30): actor_a disappears, c improves 3->2.
        NodeWarehouseRowMeta(node_id="ar_v1|actor_b|EG|2011-02", admin1_code="EG", first_seen=date(2011, 2, 5)),
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|EG|2011-02", admin1_code="EG", first_seen=date(2011, 2, 5)),
        # t2 farther candidate: not the nearest, should not be used for h=30 from t0.
        NodeWarehouseRowMeta(node_id="ar_v1|actor_c|EG|2011-03", admin1_code="EG", first_seen=date(2011, 3, 10)),
    ]
    return NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=len(rows),
        rows=rows,
    )


def test_build_and_load_predictive_targets(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30],
        horizon_weights=[1.0],
    )

    meta, targets = load_predictive_targets(meta_path, manifest)
    assert meta.embedding_version == manifest.embedding_version
    assert meta.horizons_days == [30]
    assert 30 in targets
    arr = targets[30]
    assert arr.shape == (manifest.row_count,)
    assert arr.dtype == np.float32
    assert float(arr.max()) == 0.0
    assert float(arr.min()) >= 0.0


def test_centrality_rankchange_is_multiplicity_weighted_and_deterministic(tmp_path):
    manifest = _centrality_manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30],
        horizon_weights=[1.0],
    )

    _meta, targets = load_predictive_targets(meta_path, manifest)
    arr = targets[30]

    expected = np.zeros((manifest.row_count,), dtype=np.float32)
    # From t0 -> t1 in EG after actor_a disappears:
    # actor_b: 2 -> 1 => 1/2, actor_c: 3 -> 2 => 1/3, actor_d: 4 -> 3 => 1/4.
    expected[1] = np.float32(0.5)
    expected[2] = np.float32(1.0 / 3.0)
    expected[3] = np.float32(0.25)

    np.testing.assert_allclose(arr, expected)


def test_nearest_date_at_or_after_uses_closest_future_candidate(tmp_path):
    manifest = _nearest_future_manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30],
        horizon_weights=[1.0],
    )

    _meta, targets = load_predictive_targets(meta_path, manifest)
    arr = targets[30]

    # For actor_c at t0, nearest date >= 2011-01-31 is 2011-02-05 (not 2011-03-10):
    # rank improves 3 -> 2 => 1/3.
    assert arr[2] == np.float32(1.0 / 3.0)


def test_invalid_horizons_validation(tmp_path):
    manifest = _manifest()

    with pytest.raises(ValueError, match="horizons_days must be non-empty"):
        build_predictive_targets(
            manifest,
            tmp_path,
            source_manifest_path="/tmp/manifest.json",
            horizons_days=[],
            horizon_weights=[1.0],
        )

    with pytest.raises(ValueError, match="horizon_weights must be non-empty"):
        build_predictive_targets(
            manifest,
            tmp_path,
            source_manifest_path="/tmp/manifest.json",
            horizons_days=[30],
            horizon_weights=[],
        )

    with pytest.raises(ValueError, match="length mismatch"):
        build_predictive_targets(
            manifest,
            tmp_path,
            source_manifest_path="/tmp/manifest.json",
            horizons_days=[30, 60],
            horizon_weights=[1.0],
        )

    with pytest.raises(ValueError, match="positive integers"):
        build_predictive_targets(
            manifest,
            tmp_path,
            source_manifest_path="/tmp/manifest.json",
            horizons_days=[0, 30],
            horizon_weights=[1.0, 0.5],
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        build_predictive_targets(
            manifest,
            tmp_path,
            source_manifest_path="/tmp/manifest.json",
            horizons_days=[30, 30],
            horizon_weights=[1.0, 0.5],
        )


def test_load_predictive_targets_row_count_mismatch_guard(tmp_path):
    manifest = _manifest()
    meta_path = build_predictive_targets(
        manifest,
        tmp_path,
        source_manifest_path="/tmp/manifest.json",
        horizons_days=[30],
        horizon_weights=[1.0],
    )

    mismatched = manifest.model_copy(update={"row_count": manifest.row_count + 1})
    with pytest.raises(ValueError, match="row_count mismatch"):
        load_predictive_targets(meta_path, mismatched)
