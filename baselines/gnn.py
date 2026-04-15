"""Heterogeneous GNN baseline for regional France protest forecasting."""

from __future__ import annotations

import datetime as dt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

from baselines.features import FEATURE_NAMES, FeatureRow
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES

UTC = dt.timezone.utc
EVENT_FEATURE_DIM = 4
EVENT_FEATURE_KEYS = ["goldstein_scale", "avg_tone", "num_mentions", "num_articles"]


class GNNForecastRow(BaseModel):
    forecast_origin: dt.date
    admin1_code: str
    model_name: str
    predicted_count: float
    predicted_occurrence_probability: float
    target_count_next_7d: int = 0
    target_occurs_next_7d: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class HeteroGNNModel(nn.Module):
    def __init__(
        self,
        location_feature_dim: int,
        event_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.loc_proj = nn.Linear(location_feature_dim, hidden_dim)
        self.evt_proj = nn.Linear(event_feature_dim, hidden_dim)
        self.conv = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        loc_x = F.relu(self.loc_proj(data["location"].x))
        evt_x = F.relu(self.evt_proj(data["event"].x))

        if ("event", "occurs_in", "location") in data.edge_types:
            edge_index = data["event", "occurs_in", "location"].edge_index
            loc_x = F.relu(self.conv((evt_x, loc_x), edge_index))

        logits = self.head(loc_x).squeeze(-1)
        return logits


def build_graph_from_snapshot(
    *,
    snapshot: dict[str, Any],
    feature_rows: list[FeatureRow],
) -> HeteroData:
    data = HeteroData()

    location_nodes = [n for n in snapshot["nodes"] if n["type"] == "Location"]
    location_id_to_idx: dict[str, int] = {}
    loc_admin1: list[str] = []
    for i, node in enumerate(location_nodes):
        location_id_to_idx[node["id"]] = i
        loc_admin1.append(node["attributes"]["admin1_code"])

    feature_by_admin1 = {r.admin1_code: r.features for r in feature_rows}
    loc_x = torch.zeros((len(location_nodes), len(FEATURE_NAMES)), dtype=torch.float32)
    for i, code in enumerate(loc_admin1):
        if code in feature_by_admin1:
            loc_x[i] = torch.tensor(
                [feature_by_admin1[code][name] for name in FEATURE_NAMES],
                dtype=torch.float32,
            )
    data["location"].x = loc_x
    data["location"].admin1_codes = loc_admin1

    event_nodes = [n for n in snapshot["nodes"] if n["type"] == "Event"]
    event_id_to_idx: dict[str, int] = {n["id"]: i for i, n in enumerate(event_nodes)}
    evt_x = torch.zeros((len(event_nodes), EVENT_FEATURE_DIM), dtype=torch.float32)
    for i, node in enumerate(event_nodes):
        attrs = node.get("attributes", {})
        for j, key in enumerate(EVENT_FEATURE_KEYS):
            val = attrs.get(key)
            if val is not None:
                evt_x[i, j] = float(val)
    data["event"].x = evt_x

    occurs_in_src, occurs_in_dst = [], []
    for edge in snapshot["edges"]:
        if edge["type"] != "occurs_in":
            continue
        src_id = edge["source"]
        dst_id = edge["target"]
        if src_id in event_id_to_idx and dst_id in location_id_to_idx:
            occurs_in_src.append(event_id_to_idx[src_id])
            occurs_in_dst.append(location_id_to_idx[dst_id])

    if occurs_in_src:
        data["event", "occurs_in", "location"].edge_index = torch.tensor(
            [occurs_in_src, occurs_in_dst], dtype=torch.long
        )
    else:
        data["event", "occurs_in", "location"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    target_counts: dict[str, int] = {}
    target_occurs: dict[str, bool] = {}
    for row in snapshot.get("target_table", []):
        code = row["metadata"]["admin1_code"]
        if code in EXCLUDED_REGIONAL_ADMIN1_CODES:
            continue
        if row["name"] == "target_count_next_7d":
            target_counts[code] = int(row["value"])
        elif row["name"] == "target_occurs_next_7d":
            target_occurs[code] = bool(row["value"])

    y = torch.zeros(len(location_nodes), dtype=torch.float32)
    mask = torch.zeros(len(location_nodes), dtype=torch.bool)
    for i, code in enumerate(loc_admin1):
        if code in EXCLUDED_REGIONAL_ADMIN1_CODES:
            continue
        if code in target_occurs:
            y[i] = float(target_occurs[code])
            mask[i] = True
    data["location"].y = y
    data["location"].mask = mask
    data["location"].target_counts = torch.tensor(
        [target_counts.get(code, 0) for code in loc_admin1], dtype=torch.long
    )

    return data


def train_gnn(
    *,
    graphs: list[HeteroData],
    epochs: int = 30,
    lr: float = 5e-3,
    hidden_dim: int = 64,
) -> HeteroGNNModel:
    location_feature_dim = graphs[0]["location"].x.shape[1]
    event_feature_dim = graphs[0]["event"].x.shape[1] if len(graphs[0]["event"].x) > 0 else EVENT_FEATURE_DIM
    model = HeteroGNNModel(
        location_feature_dim=location_feature_dim,
        event_feature_dim=event_feature_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for data in graphs:
            optimizer.zero_grad()
            logits = model(data)
            mask = data["location"].mask
            if mask.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], data["location"].y[mask]
            )
            loss.backward()
            optimizer.step()
    return model


def predict_gnn(
    *,
    model: HeteroGNNModel,
    graph: HeteroData,
    origin_date: dt.date,
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] | None = None,
) -> list[GNNForecastRow]:
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        probs = torch.sigmoid(logits).cpu().numpy()

    rows: list[GNNForecastRow] = []
    for i, code in enumerate(graph["location"].admin1_codes):
        if code in EXCLUDED_REGIONAL_ADMIN1_CODES:
            continue
        p = float(probs[i])
        target_count, target_occurs = (target_lookup or {}).get((origin_date, code), (0, False))
        rows.append(
            GNNForecastRow(
                forecast_origin=origin_date,
                admin1_code=code,
                model_name="gnn_sage",
                predicted_count=p,
                predicted_occurrence_probability=p,
                target_count_next_7d=target_count,
                target_occurs_next_7d=target_occurs,
            )
        )
    return rows
