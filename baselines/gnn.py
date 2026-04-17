"""Heterogeneous GNN baseline for regional France protest forecasting."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

from baselines.features import FEATURE_NAMES, FeatureRow
from evals.wikidata_linking import normalize_entity_label, wikidata_qid_for_node
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES

UTC = dt.timezone.utc
EVENT_FEATURE_DIM = 4
ACTOR_SCALAR_FEATURE_DIM = 3
ACTOR_QID_HASH_DIM = 8
ACTOR_FEATURE_DIM = ACTOR_SCALAR_FEATURE_DIM + ACTOR_QID_HASH_DIM
ACTOR_FEATURE_MODES = {"qid", "scalar", "zero"}
ACTOR_MERGE_POLICIES = {"none", "qid"}


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
    use_actor_nodes: bool = True
    use_actor_edges: bool = True
    actor_feature_mode: str = "qid"
    actor_merge_policy: str = "qid"
    use_source_identity: bool = True
    allowed_source_names: tuple[str, ...] | None = None
    description: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_location_features": self.use_location_features,
            "use_event_features": self.use_event_features,
            "use_event_edges": self.use_event_edges,
            "use_actor_nodes": self.use_actor_nodes,
            "use_actor_edges": self.use_actor_edges,
            "actor_feature_mode": self.actor_feature_mode,
            "actor_merge_policy": self.actor_merge_policy,
            "use_source_identity": self.use_source_identity,
            "allowed_source_names": list(self.allowed_source_names)
            if self.allowed_source_names is not None
            else None,
            "description": self.description,
        }


FULL_GRAPH_ABLATION = GNNGraphAblation(
    name="full_graph",
    description=(
        "Location history features, event attributes, and actor identity features passed over "
        "actor participates_in event and event occurs_in location edges."
    ),
)

GNN_GRAPH_ABLATIONS: tuple[GNNGraphAblation, ...] = (
    FULL_GRAPH_ABLATION,
    GNNGraphAblation(
        name="location_features_only",
        use_event_features=False,
        use_event_edges=False,
        use_actor_nodes=False,
        use_actor_edges=False,
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
        use_actor_nodes=False,
        use_actor_edges=False,
        description="Location features with event nodes present but disconnected.",
    ),
    GNNGraphAblation(
        name="no_location_features",
        use_location_features=False,
        description="Event layer with all tabular location features zeroed.",
    ),
    GNNGraphAblation(
        name="no_actor_track",
        use_actor_nodes=False,
        use_actor_edges=False,
        description="Current event-location graph without Actor nodes or participates_in edges.",
    ),
    GNNGraphAblation(
        name="actor_structure_only",
        actor_feature_mode="zero",
        description="Actor nodes and participates_in edges with zero actor features.",
    ),
    GNNGraphAblation(
        name="actor_no_qid_merge",
        actor_merge_policy="none",
        description="Actor track without collapsing duplicate Actor nodes by Wikidata QID.",
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
    """Actor -> Event -> Location GraphSAGE model.

    Expected tensors:
    - data["actor"].x: [num_actors, ACTOR_FEATURE_DIM] when actor tracking is enabled.
    - data["event"].x: [num_events, EVENT_FEATURE_DIM].
    - data["location"].x: [num_locations, len(FEATURE_NAMES)].
    """

    def __init__(
        self,
        location_feature_dim: int,
        event_feature_dim: int,
        actor_feature_dim: int = 0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.loc_proj = nn.Linear(location_feature_dim, hidden_dim)
        self.evt_proj = nn.Linear(event_feature_dim, hidden_dim)
        self.actor_proj = (
            nn.Linear(actor_feature_dim, hidden_dim) if actor_feature_dim > 0 else None
        )
        self.actor_to_event_conv = (
            SAGEConv((hidden_dim, hidden_dim), hidden_dim) if actor_feature_dim > 0 else None
        )
        self.conv = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        loc_x = F.relu(self.loc_proj(data["location"].x))
        evt_x = F.relu(self.evt_proj(data["event"].x))

        if (
            self.actor_proj is not None
            and self.actor_to_event_conv is not None
            and "actor" in data.node_types
            and ("actor", "participates_in", "event") in data.edge_types
        ):
            actor_edge_index = data["actor", "participates_in", "event"].edge_index
            actor_raw_x = data["actor"].x
            if actor_raw_x.shape[0] > 0 and actor_edge_index.shape[1] > 0:
                actor_x = F.relu(self.actor_proj(actor_raw_x))
                actor_msg = self.actor_to_event_conv((actor_x, evt_x), actor_edge_index)
                evt_x = F.relu(evt_x + actor_msg)

        if ("event", "occurs_in", "location") in data.edge_types:
            edge_index = data["event", "occurs_in", "location"].edge_index
            loc_x = F.relu(self.conv((evt_x, loc_x), edge_index))

        logits = self.head(loc_x).squeeze(-1)
        return logits


def _validate_actor_ablation(ablation: GNNGraphAblation) -> None:
    if ablation.actor_feature_mode not in ACTOR_FEATURE_MODES:
        available = ", ".join(sorted(ACTOR_FEATURE_MODES))
        raise ValueError(
            f"unknown actor_feature_mode '{ablation.actor_feature_mode}'. Available: {available}"
        )
    if ablation.actor_merge_policy not in ACTOR_MERGE_POLICIES:
        available = ", ".join(sorted(ACTOR_MERGE_POLICIES))
        raise ValueError(
            f"unknown actor_merge_policy '{ablation.actor_merge_policy}'. Available: {available}"
        )


def _qid_hash_features(qid: str) -> torch.Tensor:
    values = []
    for index in range(ACTOR_QID_HASH_DIM):
        digest = hashlib.sha256(f"{qid}:{index}".encode("utf-8")).digest()
        raw = int.from_bytes(digest[:4], byteorder="big", signed=False)
        values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
    return torch.tensor(values, dtype=torch.float32)


def _provenance_sources(node: dict[str, Any]) -> set[str]:
    provenance = node.get("provenance")
    if not isinstance(provenance, dict):
        return set()
    sources = provenance.get("sources")
    if isinstance(sources, list):
        return {source for source in sources if isinstance(source, str) and source}
    if isinstance(sources, str) and sources:
        return {sources}
    return set()


def _actor_merge_key(node: dict[str, Any], merge_policy: str) -> tuple[str, str | None]:
    qid = wikidata_qid_for_node(node)
    if merge_policy == "qid" and qid:
        return f"qid:{qid}", qid
    return f"id:{node['id']}", qid


def _build_actor_tensors(
    *,
    snapshot: dict[str, Any],
    event_id_to_idx: dict[str, int],
    ablation: GNNGraphAblation,
) -> tuple[torch.Tensor, list[str], torch.Tensor, dict[str, Any]]:
    actor_node_by_id = {
        node["id"]: node for node in snapshot["nodes"] if node.get("type") == "Actor"
    }

    raw_edges = []
    referenced_actor_ids: set[str] = set()
    for edge in snapshot["edges"]:
        if edge.get("type") != "participates_in":
            continue
        actor_id = edge.get("source")
        event_id = edge.get("target")
        if actor_id in actor_node_by_id and event_id in event_id_to_idx:
            raw_edges.append(edge)
            referenced_actor_ids.add(actor_id)

    group_by_key: dict[str, dict[str, Any]] = {}
    actor_id_to_key: dict[str, str] = {}
    for actor_id in sorted(referenced_actor_ids):
        node = actor_node_by_id[actor_id]
        key, qid = _actor_merge_key(node, ablation.actor_merge_policy)
        actor_id_to_key[actor_id] = key
        group = group_by_key.setdefault(
            key,
            {
                "actor_ids": [],
                "qid": qid,
                "sources": set(),
                "participation_count": 0,
            },
        )
        group["actor_ids"].append(actor_id)
        group["sources"].update(_provenance_sources(node))
        if qid and group["qid"] is None:
            group["qid"] = qid

    for edge in raw_edges:
        actor_id = edge["source"]
        group = group_by_key[actor_id_to_key[actor_id]]
        group["participation_count"] += 1
        attrs = edge.get("attributes") or {}
        source_name = attrs.get("source_name")
        if isinstance(source_name, str) and source_name:
            group["sources"].add(source_name)

    actor_keys = sorted(group_by_key)
    actor_key_to_idx = {key: index for index, key in enumerate(actor_keys)}
    actor_x = torch.zeros((len(actor_keys), ACTOR_FEATURE_DIM), dtype=torch.float32)
    if ablation.actor_feature_mode != "zero":
        for index, key in enumerate(actor_keys):
            group = group_by_key[key]
            qid = group["qid"]
            actor_x[index, 0] = math.log1p(float(group["participation_count"]))
            actor_x[index, 1] = math.log1p(float(len(group["sources"])))
            actor_x[index, 2] = 1.0 if qid else 0.0
            if ablation.actor_feature_mode == "qid" and qid:
                actor_x[index, ACTOR_SCALAR_FEATURE_DIM:] = _qid_hash_features(qid)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    if ablation.use_actor_edges:
        for edge in raw_edges:
            key = actor_id_to_key[edge["source"]]
            edge_src.append(actor_key_to_idx[key])
            edge_dst.append(event_id_to_idx[edge["target"]])

    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if edge_src
        else torch.zeros((2, 0), dtype=torch.long)
    )

    qid_group_sizes: Counter[str] = Counter()
    unresolved_label_keys: Counter[str] = Counter()
    for actor_id in referenced_actor_ids:
        node = actor_node_by_id[actor_id]
        qid = wikidata_qid_for_node(node)
        if qid:
            qid_group_sizes[qid] += 1
        else:
            label = node.get("label")
            if isinstance(label, str) and label.strip():
                unresolved_label_keys[normalize_entity_label(label)] += 1

    diagnostics = {
        "actor_nodes_input": len(referenced_actor_ids),
        "actor_nodes_output": len(actor_keys),
        "actors_with_qid": sum(
            1
            for actor_id in referenced_actor_ids
            if wikidata_qid_for_node(actor_node_by_id[actor_id])
        ),
        "actor_nodes_collapsed_by_qid": sum(
            max(0, size - 1) for size in qid_group_sizes.values()
        )
        if ablation.actor_merge_policy == "qid"
        else 0,
        "unresolved_nodes_merged_by_strict_key": 0,
        "unresolved_label_collision_count": sum(
            1 for size in unresolved_label_keys.values() if size > 1
        ),
        "actor_participates_in_edges": len(raw_edges),
        "actor_feature_mode": ablation.actor_feature_mode,
        "actor_merge_policy": ablation.actor_merge_policy,
        "actor_qid_hash_dim": ACTOR_QID_HASH_DIM,
    }
    return actor_x, actor_keys, edge_index, diagnostics


def build_graph_from_snapshot(
    *,
    snapshot: dict[str, Any],
    feature_rows: list[FeatureRow],
    ablation: GNNGraphAblation | None = None,
) -> HeteroData:
    _ensure_gnn_runtime_configured()
    ablation = ablation or FULL_GRAPH_ABLATION
    _validate_actor_ablation(ablation)
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

    if ablation.use_actor_nodes:
        actor_x, actor_keys, participates_in_edge_index, actor_diagnostics = _build_actor_tensors(
            snapshot=snapshot,
            event_id_to_idx=event_id_to_idx,
            ablation=ablation,
        )
        data["actor"].x = actor_x
        data["actor"].actor_keys = actor_keys
        data["actor", "participates_in", "event"].edge_index = participates_in_edge_index
        data.actor_diagnostics = actor_diagnostics
    else:
        data.actor_diagnostics = {
            "actor_nodes_input": 0,
            "actor_nodes_output": 0,
            "actors_with_qid": 0,
            "actor_nodes_collapsed_by_qid": 0,
            "unresolved_nodes_merged_by_strict_key": 0,
            "unresolved_label_collision_count": 0,
            "actor_participates_in_edges": 0,
            "actor_feature_mode": ablation.actor_feature_mode,
            "actor_merge_policy": ablation.actor_merge_policy,
            "actor_qid_hash_dim": ACTOR_QID_HASH_DIM,
        }

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
    actor_feature_dim = (
        graphs[0]["actor"].x.shape[1] if "actor" in graphs[0].node_types else 0
    )
    model = HeteroGNNModel(
        location_feature_dim=location_feature_dim,
        event_feature_dim=event_feature_dim,
        actor_feature_dim=actor_feature_dim,
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
                model_name=model_name,
                predicted_count=p,
                predicted_occurrence_probability=p,
                target_count_next_7d=target_count,
                target_occurs_next_7d=target_occurs,
                metadata={"ablation": ablation.metadata()} if ablation is not None else {},
            )
        )
    return rows
