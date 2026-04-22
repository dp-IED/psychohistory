from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import torch

from baselines.gnn import HeteroGNNModel, build_graph_from_snapshot
from baselines.graph_builder_bag_encoder import BagEncoder
from baselines.graph_builder_dual_path_ablation import (
    FORECAST_HEAD_IN_DIM,
    forward_path_a,
    forward_path_b,
    make_path_proj,
    run_dual_path_epoch_metrics,
)
from baselines.graph_builder_forecast_stack import ForecastHead, GateMLP
from baselines.features import FEATURE_NAMES
from schemas.graph_builder_retrieval import (
    BUILDER_EMBEDDING_DIM,
    MAX_RETRIEVED_EDGES,
    MAX_RETRIEVED_NODES,
    RetrievedGraphBatch,
)
def _gnn_test_helpers():
    path = Path(__file__).resolve().parent / "test_gnn.py"
    spec = importlib.util.spec_from_file_location("test_gnn_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_gnn_helpers = _gnn_test_helpers()


def _tiny_retrieved_batch(batch_size: int = 2) -> RetrievedGraphBatch:
    node_feat = torch.randn(batch_size, MAX_RETRIEVED_NODES, BUILDER_EMBEDDING_DIM)
    node_mask = torch.zeros(batch_size, MAX_RETRIEVED_NODES, dtype=torch.bool)
    node_mask[:, :4] = True
    edge_index = torch.zeros(batch_size, 2, MAX_RETRIEVED_EDGES, dtype=torch.long)
    edge_weight = torch.zeros(batch_size, MAX_RETRIEVED_EDGES)
    edge_mask = torch.zeros(batch_size, MAX_RETRIEVED_EDGES, dtype=torch.bool)
    return RetrievedGraphBatch(
        node_feat=node_feat,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


def test_dual_path_shapes_shared_head_and_path_b_has_no_gate_mlp() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _gnn_helpers._minimal_snapshot(origin)
    feature_rows = _gnn_helpers._minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    hidden = 32
    gnn = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        hidden_dim=hidden,
    )
    path_proj = make_path_proj(gnn, data)
    assert path_proj.out_features == FORECAST_HEAD_IN_DIM == 133

    shared_head = ForecastHead()
    head_id = id(shared_head)
    gate_mlp = GateMLP()
    bag_encoder = BagEncoder()

    bag_batch = _tiny_retrieved_batch(batch_size=3)
    p_a, _ = forward_path_a(bag_batch, bag_encoder, gate_mlp, shared_head)
    assert p_a.shape == (3, 1)

    p_b, _ = forward_path_b(data, gnn, path_proj, shared_head)
    assert p_b.shape == (1, 1)
    assert id(shared_head) == head_id

    assert path_proj.in_features == hidden
    path_b_modules = {path_proj, shared_head}
    assert gate_mlp not in path_b_modules
    assert not any(isinstance(m, GateMLP) for m in path_b_modules)

    y = torch.zeros(1, 1)
    loss_b = torch.nn.functional.mse_loss(p_b, y)
    loss_b.backward()
    assert all(p is None or p.grad is None for p in gate_mlp.parameters())


def test_run_dual_path_epoch_metrics_stub() -> None:
    d = run_dual_path_epoch_metrics(brier_bag=0.1, brier_legacy=0.2)
    assert d == {"brier_bag": 0.1, "brier_legacy": 0.2}
    assert run_dual_path_epoch_metrics() == {}
