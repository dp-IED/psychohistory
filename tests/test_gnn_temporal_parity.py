from __future__ import annotations

import datetime as dt

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from baselines.features import FEATURE_NAMES, FeatureRow
from baselines.gnn import HeteroGNNModel, build_graph_from_snapshot
from test_gnn import _minimal_feature_rows, _minimal_snapshot


def _legacy_logits(model: HeteroGNNModel, data: HeteroData) -> torch.Tensor:
    loc_x = F.relu(model.loc_proj(data["location"].x))
    evt_x = F.relu(model.evt_proj(data["event"].x))
    if ("event", "occurs_in", "location") in data.edge_types:
        edge_index = data["event", "occurs_in", "location"].edge_index
        loc_x = F.relu(model.conv((evt_x, loc_x), edge_index))
    return model.head(loc_x).squeeze(-1)


def test_hetero_gnn_temporal_none_matches_legacy_forward() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin, n_events=4)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    torch.manual_seed(12345)
    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        hidden_dim=32,
    )
    expected = _legacy_logits(model, data)
    actual = model(data)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)
