from __future__ import annotations

from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from baselines.graph_builder_stage1_train import _per_probe_predictive_rankchange
from schemas.graph_builder_probe import (
    ActorStateQuery,
    AssumptionEmphasis,
    GenerationMeta,
    LensParamsV0,
    ProbeRecord,
    QStructV0,
)
from schemas.graph_builder_warehouse import NodeWarehouseManifest, NodeWarehouseRowMeta


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


def test_predictive_rankchange_accepts_tuple_entry():
    manifest = NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=4,
        rows=[
            NodeWarehouseRowMeta(node_id="n0", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n1", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n2", admin1_code="US", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n3", admin1_code="EG", first_seen=date(2011, 1, 1)),
        ],
    )

    q = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float32)
    global_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    node_mask = torch.tensor([True, True, True, True])
    node_feat = torch.tensor(
        [
            [0.3, -0.2, 0.1],
            [0.1, 0.2, 0.3],
            [-0.2, 0.4, 0.1],
            [0.5, -0.1, -0.2],
        ],
        dtype=torch.float32,
    )
    horizon_targets = {30: np.zeros((4,), dtype=np.float32)}
    horizon_weights = {30: 1.0}

    tuple_entry = {
        "probe_id": "p-eg-1",
        "horizon_days": 30,
        "pos_global_indices": [0, 3],
        "neg_global_indices": [1, 2],
    }

    loss, ok = _per_probe_predictive_rankchange(
        q,
        probe=_probe(),
        global_indices_row=global_indices,
        node_mask_row=node_mask,
        node_feat_row=node_feat,
        manifest=manifest,
        horizon_targets=horizon_targets,
        horizon_weights=horizon_weights,
        temperature=0.1,
        tuple_entry=tuple_entry,
    )

    assert ok is True
    assert loss is not None
    assert torch.isfinite(loss)


def test_tuple_strata_weighting_changes_loss_value():
    manifest = NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=3,
        rows=[
            NodeWarehouseRowMeta(node_id="n0", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n1", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n2", admin1_code="EG", first_seen=date(2011, 1, 1)),
        ],
    )
    q = torch.tensor([1.0, 0.0], dtype=torch.float32)
    global_indices = np.asarray([0, 1, 2], dtype=np.int64)
    node_mask = torch.tensor([True, True, True])
    node_feat = torch.tensor([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    tuple_entry = {
        "probe_id": "p-eg-1",
        "horizon_days": 30,
        "pos_global_indices": [0],
        "neg_global_indices": [1, 2],
        "strata_indices": {
            "same_time_wrong_domain": [1],
            "same_domain_wrong_horizon": [2],
            "same_region_non_precursor": [],
        },
    }
    horizon_targets = {30: np.zeros((3,), dtype=np.float32)}
    horizon_weights = {30: 0.0}

    loss_low, ok_low = _per_probe_predictive_rankchange(
        q,
        probe=_probe(),
        global_indices_row=global_indices,
        node_mask_row=node_mask,
        node_feat_row=node_feat,
        manifest=manifest,
        horizon_targets=horizon_targets,
        horizon_weights=horizon_weights,
        temperature=1.0,
        tuple_entry=tuple_entry,
        predictive_stratum_weight_same_time_wrong_domain=1.0,
    )
    loss_high, ok_high = _per_probe_predictive_rankchange(
        q,
        probe=_probe(),
        global_indices_row=global_indices,
        node_mask_row=node_mask,
        node_feat_row=node_feat,
        manifest=manifest,
        horizon_targets=horizon_targets,
        horizon_weights=horizon_weights,
        temperature=1.0,
        tuple_entry=tuple_entry,
        predictive_stratum_weight_same_time_wrong_domain=8.0,
    )
    assert ok_low and ok_high
    assert loss_low is not None and loss_high is not None
    assert float(loss_high) > float(loss_low)


def test_predictive_tuple_and_fallback_blending_applied():
    manifest = NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=3,
        rows=[
            NodeWarehouseRowMeta(node_id="n0", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n1", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n2", admin1_code="EG", first_seen=date(2011, 1, 1)),
        ],
    )
    q = torch.tensor([1.0, 0.0], dtype=torch.float32)
    global_indices = np.asarray([0, 1, 2], dtype=np.int64)
    node_mask = torch.tensor([True, True, True])
    node_feat = torch.tensor([[2.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    tuple_entry = {
        "probe_id": "p-eg-1",
        "horizon_days": 30,
        "pos_global_indices": [0],
        "neg_global_indices": [1],
        "strata_indices": {
            "same_time_wrong_domain": [1],
            "same_domain_wrong_horizon": [],
            "same_region_non_precursor": [],
        },
    }
    horizon_targets = {30: np.asarray([0.2, 0.8, 0.0], dtype=np.float32)}
    horizon_weights = {30: 1.0}

    tuple_w = 1.5
    fallback_w = 0.5
    loss, ok = _per_probe_predictive_rankchange(
        q,
        probe=_probe(),
        global_indices_row=global_indices,
        node_mask_row=node_mask,
        node_feat_row=node_feat,
        manifest=manifest,
        horizon_targets=horizon_targets,
        horizon_weights=horizon_weights,
        temperature=1.0,
        tuple_entry=tuple_entry,
        predictive_tuple_weight=tuple_w,
        predictive_fallback_weight=fallback_w,
        predictive_stratum_weight_same_time_wrong_domain=1.0,
    )
    assert ok
    assert loss is not None

    tuple_logits = torch.tensor([2.0, -1.0], dtype=torch.float32)
    tuple_y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    tuple_loss = F.binary_cross_entropy_with_logits(tuple_logits, tuple_y)

    fallback_logits = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float32)
    fallback_y = torch.tensor([0.2, 0.8, 0.0], dtype=torch.float32)
    fallback_loss = F.binary_cross_entropy_with_logits(fallback_logits, fallback_y)

    expected = (tuple_w * tuple_loss) + (fallback_w * fallback_loss)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_predictive_rankchange_branch_usage_flags_tuple_and_fallback():
    manifest = NodeWarehouseManifest(
        manifest_version="v1",
        embedding_version="emb_v1",
        mmap_path="dummy.mmap",
        row_count=3,
        rows=[
            NodeWarehouseRowMeta(node_id="n0", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n1", admin1_code="EG", first_seen=date(2011, 1, 1)),
            NodeWarehouseRowMeta(node_id="n2", admin1_code="EG", first_seen=date(2011, 1, 1)),
        ],
    )
    q = torch.tensor([0.2, 0.1], dtype=torch.float32)
    global_indices = np.asarray([0, 1, 2], dtype=np.int64)
    node_mask = torch.tensor([True, True, True])
    node_feat = torch.tensor([[0.5, 0.0], [0.2, 0.1], [-0.3, 0.2]], dtype=torch.float32)
    tuple_entry = {
        "probe_id": "p-eg-1",
        "horizon_days": 30,
        "pos_global_indices": [0],
        "neg_global_indices": [1],
    }
    horizon_targets = {30: np.asarray([1.0, 0.0, 0.0], dtype=np.float32)}
    horizon_weights = {30: 1.0}

    loss, ok, usage = _per_probe_predictive_rankchange(
        q,
        probe=_probe(),
        global_indices_row=global_indices,
        node_mask_row=node_mask,
        node_feat_row=node_feat,
        manifest=manifest,
        horizon_targets=horizon_targets,
        horizon_weights=horizon_weights,
        temperature=1.0,
        tuple_entry=tuple_entry,
        return_branch_usage=True,
    )
    assert ok
    assert loss is not None
    assert usage["tuple_used"] is True
    assert usage["fallback_used"] is True
