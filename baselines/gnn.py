"""Heterogeneous GNN baseline for regional France protest forecasting."""

from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

from baselines.features import FEATURE_NAMES, FeatureRow
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES

UTC = dt.timezone.utc
EVENT_FEATURE_DIM = 4


def _configure_pytorch_threads() -> None:
    """Avoid OpenMP/BLAS oversubscription; some macOS PyTorch wheels segfault otherwise."""
    if os.environ.get("PSYCHOHISTORY_TORCH_CONFIGURE", "1") == "0":
        return
    if sys.platform != "darwin":
        return
    raw = os.environ.get("PYTORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if raw is not None:
        try:
            torch.set_num_threads(max(1, int(raw)))
        except (TypeError, ValueError):
            torch.set_num_threads(1)
    else:
        torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_gnn_runtime_configured = False


def _ensure_gnn_runtime_configured() -> None:
    global _gnn_runtime_configured
    if _gnn_runtime_configured:
        return
    _configure_pytorch_threads()
    _gnn_runtime_configured = True


EVENT_FEATURE_KEYS = ["goldstein_scale", "avg_tone", "num_mentions", "num_articles"]


@dataclass(frozen=True)
class GNNGraphAblation:
    name: str
    use_location_features: bool = True
    use_event_features: bool = True
    use_event_edges: bool = True
    use_source_identity: bool = True
    allowed_source_names: tuple[str, ...] | None = None
    description: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_location_features": self.use_location_features,
            "use_event_features": self.use_event_features,
            "use_event_edges": self.use_event_edges,
            "use_source_identity": self.use_source_identity,
            "allowed_source_names": list(self.allowed_source_names)
            if self.allowed_source_names is not None
            else None,
            "description": self.description,
        }


FULL_GRAPH_ABLATION = GNNGraphAblation(
    name="full_graph",
    description="Location history features plus event attributes passed over occurs_in edges.",
)

GNN_GRAPH_ABLATIONS: tuple[GNNGraphAblation, ...] = (
    FULL_GRAPH_ABLATION,
    GNNGraphAblation(
        name="location_features_only",
        use_event_features=False,
        use_event_edges=False,
        description="Tabular location history features only; event nodes are disconnected.",
    ),
    GNNGraphAblation(
        name="event_layer_only",
        use_location_features=False,
        description="Event attributes passed over occurs_in edges with location features zeroed.",
    ),
    GNNGraphAblation(
        name="no_event_features",
        use_event_features=False,
        description="Location features plus event topology, with event attributes zeroed.",
    ),
    GNNGraphAblation(
        name="no_event_edges",
        use_event_edges=False,
        description="Location features with event nodes present but disconnected.",
    ),
    GNNGraphAblation(
        name="no_location_features",
        use_location_features=False,
        description="Event layer with all tabular location features zeroed.",
    ),
)
GNN_GRAPH_ABLATION_BY_NAME = {ablation.name: ablation for ablation in GNN_GRAPH_ABLATIONS}


def resolve_gnn_graph_ablations(names: list[str] | None = None) -> list[GNNGraphAblation]:
    if names is None:
        return list(GNN_GRAPH_ABLATIONS)

    resolved: list[GNNGraphAblation] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate GNN ablation: {name}")
        try:
            resolved.append(GNN_GRAPH_ABLATION_BY_NAME[name])
        except KeyError as exc:
            available = ", ".join(sorted(GNN_GRAPH_ABLATION_BY_NAME))
            raise ValueError(f"unknown GNN ablation '{name}'. Available: {available}") from exc
        seen.add(name)
    if not resolved:
        raise ValueError("at least one GNN ablation is required")
    return resolved


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
        self.hidden_dim = hidden_dim
        self.loc_proj = nn.Linear(location_feature_dim, hidden_dim)
        self.evt_proj = nn.Linear(event_feature_dim, hidden_dim)
        self.loc_hist_gru = nn.GRU(location_feature_dim, hidden_dim, batch_first=True)
        self.loc_hist_proj = nn.Linear(hidden_dim, hidden_dim)
        self.conv = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_location_temporal(
        self,
        loc_temporal: torch.Tensor,
        loc_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, _ = loc_temporal.shape
        device = loc_temporal.device
        xr = torch.flip(loc_temporal, dims=(1,))
        mr = torch.flip(loc_time_mask, dims=(1,))
        lengths = mr.long().sum(dim=1)
        out = torch.zeros(b, self.hidden_dim, device=device, dtype=loc_temporal.dtype)
        ok = lengths > 0
        if bool(ok.any().item()):
            idx = torch.where(ok)[0]
            x_sub = xr[ok]
            lens = lengths[ok].detach().cpu()
            packed = pack_padded_sequence(x_sub, lens, batch_first=True, enforce_sorted=False)
            _, h_n = self.loc_hist_gru(packed)
            out[idx] = h_n[-1]
        return out

    def forward(
        self,
        data: HeteroData,
        loc_temporal: torch.Tensor | None = None,
        loc_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if loc_temporal is None:
            loc_x = F.relu(self.loc_proj(data["location"].x))
        else:
            if loc_time_mask is None:
                raise ValueError("loc_time_mask is required when loc_temporal is set")
            loc_h = self._encode_location_temporal(loc_temporal, loc_time_mask)
            loc_x = F.relu(self.loc_hist_proj(loc_h))
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
    ablation: GNNGraphAblation | None = None,
) -> HeteroData:
    _ensure_gnn_runtime_configured()
    ablation = ablation or FULL_GRAPH_ABLATION
    data = HeteroData()

    location_nodes = [n for n in snapshot["nodes"] if n["type"] == "Location"]
    location_id_to_idx: dict[str, int] = {}
    loc_admin1: list[str] = []
    for i, node in enumerate(location_nodes):
        location_id_to_idx[node["id"]] = i
        loc_admin1.append(node["attributes"]["admin1_code"])

    feature_by_admin1 = {r.admin1_code: r.features for r in feature_rows}
    loc_x = torch.zeros((len(location_nodes), len(FEATURE_NAMES)), dtype=torch.float32)
    if ablation.use_location_features:
        for i, code in enumerate(loc_admin1):
            if code in feature_by_admin1:
                loc_x[i] = torch.tensor(
                    [feature_by_admin1[code][name] for name in FEATURE_NAMES],
                    dtype=torch.float32,
                )
    data["location"].x = loc_x
    data["location"].admin1_codes = loc_admin1

    event_nodes = [n for n in snapshot["nodes"] if n["type"] == "Event"]
    if ablation.allowed_source_names is not None:
        allowed_source_names = set(ablation.allowed_source_names)
        event_nodes = [
            node
            for node in event_nodes
            if node.get("attributes", {}).get("source_name") in allowed_source_names
        ]
    event_id_to_idx: dict[str, int] = {n["id"]: i for i, n in enumerate(event_nodes)}
    evt_x = torch.zeros((len(event_nodes), EVENT_FEATURE_DIM), dtype=torch.float32)
    if ablation.use_event_features:
        for i, node in enumerate(event_nodes):
            attrs = node.get("attributes", {})
            for j, key in enumerate(EVENT_FEATURE_KEYS):
                val = attrs.get(key)
                if val is not None:
                    evt_x[i, j] = float(val)
    data["event"].x = evt_x

    occurs_in_src, occurs_in_dst = [], []
    if ablation.use_event_edges:
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
    data.graph_ablation = ablation.metadata()

    return data


def train_gnn(
    *,
    graphs: list[HeteroData],
    epochs: int = 30,
    lr: float = 5e-3,
    hidden_dim: int = 64,
) -> HeteroGNNModel:
    location_feature_dim = graphs[0]["location"].x.shape[1]
    event_feature_dim = graphs[0]["event"].x.shape[1]
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
    model_name: str = "gnn_sage",
    ablation: GNNGraphAblation | None = None,
    loc_temporal: torch.Tensor | None = None,
    loc_time_mask: torch.Tensor | None = None,
) -> list[GNNForecastRow]:
    model.eval()
    with torch.no_grad():
        logits = model(graph, loc_temporal, loc_time_mask)
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
                model_name=model_name,
                predicted_count=p,
                predicted_occurrence_probability=p,
                target_count_next_7d=target_count,
                target_occurs_next_7d=target_occurs,
                metadata={"ablation": ablation.metadata()} if ablation is not None else {},
            )
        )
    return rows
