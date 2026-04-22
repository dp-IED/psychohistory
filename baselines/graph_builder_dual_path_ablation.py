# path_proj maps directly to ForecastHead input dim (133); GateMLP runs on Path A only — do not pass gates on Path B.

"""Dual-path ablation: gated bag stack (Path A) vs legacy graph embedding projection (Path B), shared ForecastHead."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from baselines.features import FEATURE_NAMES, FeatureRow
from baselines.gnn import HeteroGNNModel, build_graph_from_snapshot
from ingest.io_utils import open_text_auto
from baselines.graph_builder_bag_encoder import BagEncoder
from baselines.graph_builder_forecast_stack import (
    FORECAST_HEAD_IN_DIM,
    ForecastHead,
    GateMLP,
    path_a_head_input,
)
from schemas.graph_builder_retrieval import RetrievedGraphBatch

logger = logging.getLogger(__name__)


def legacy_dim_for_path_proj(
    frozen_hetero_gnn: HeteroGNNModel,
    data: HeteroData | None = None,
) -> int:
    """Width of ``legacy_graph_embedding`` (typically ``hidden_dim``)."""

    if data is not None:
        legacy_vec = frozen_hetero_gnn.legacy_graph_embedding(data)
        return int(legacy_vec.shape[-1])
    return int(frozen_hetero_gnn.hidden_dim)


def make_path_proj(
    frozen_hetero_gnn: HeteroGNNModel,
    data: HeteroData | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Linear:
    legacy_dim = legacy_dim_for_path_proj(frozen_hetero_gnn, data)
    lin = nn.Linear(legacy_dim, FORECAST_HEAD_IN_DIM)
    if device is not None:
        lin = lin.to(device)
    if dtype is not None:
        lin = lin.to(dtype=dtype)
    return lin


def forward_path_a(
    bag_batch: RetrievedGraphBatch,
    bag_encoder: BagEncoder,
    gate_mlp: GateMLP,
    head: ForecastHead,
    *,
    return_metrics: bool = False,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    bag_repr = bag_encoder(bag_batch)
    gates = gate_mlp(bag_repr)
    p = head(path_a_head_input(bag_repr, gates))
    metrics: dict[str, Any] | None = None
    if return_metrics:
        metrics = {}
        if y is not None:
            metrics["brier_bag"] = float((0.5 * ((p - y) ** 2)).mean().item())
    return p, metrics


def forward_path_b(
    data: HeteroData,
    frozen_hetero_gnn: HeteroGNNModel,
    path_proj: nn.Linear,
    head: ForecastHead,
    *,
    return_metrics: bool = False,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    # path_proj maps directly to ForecastHead input dim (133); GateMLP runs on Path A only — do not pass gates on Path B.
    legacy_vec = frozen_hetero_gnn.legacy_graph_embedding(data)
    if legacy_vec.dim() == 1:
        legacy_vec = legacy_vec.unsqueeze(0)
    head_in = path_proj(legacy_vec)
    p = head(head_in)
    metrics: dict[str, Any] | None = None
    if return_metrics:
        metrics = {}
        if y is not None:
            metrics["brier_legacy"] = float((0.5 * ((p - y) ** 2)).mean().item())
    return p, metrics


def run_dual_path_epoch_metrics(
    *,
    brier_bag: float | None = None,
    brier_legacy: float | None = None,
) -> dict[str, float]:
    """Minimal epoch hook: log bag vs legacy Brier (half-MSE) for ablation comparisons."""

    out: dict[str, float] = {}
    if brier_bag is not None:
        out["brier_bag"] = brier_bag
        logger.info("dual_path_epoch brier_bag=%s", brier_bag)
    if brier_legacy is not None:
        out["brier_legacy"] = brier_legacy
        logger.info("dual_path_epoch brier_legacy=%s", brier_legacy)
    return out


def _origin_date_from_snapshot(snapshot: dict[str, Any]) -> dt.date:
    meta = snapshot.get("metadata") or {}
    fo = str(meta.get("forecast_origin") or "")
    if len(fo) >= 10:
        return dt.date.fromisoformat(fo[:10])
    raise ValueError("snapshot metadata.forecast_origin missing or too short (expected ISO-8601 date prefix)")


def _feature_rows_for_snapshot(snapshot: dict[str, Any], origin: dt.date) -> list[FeatureRow]:
    codes = sorted(
        {
            str(n["attributes"]["admin1_code"])
            for n in snapshot.get("nodes", [])
            if n.get("type") == "Location"
            and isinstance(n.get("attributes"), dict)
            and n["attributes"].get("admin1_code")
        }
    )
    return [
        FeatureRow(
            forecast_origin=origin,
            admin1_code=code,
            features={name: 0.0 for name in FEATURE_NAMES},
        )
        for code in codes
    ]


def bench_path_b_wall_clock(
    *,
    snapshot_path: Path,
    repeats: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Time Path B only (``legacy_graph_embedding`` → ``path_proj`` → ``ForecastHead``).

    Uses random-init ``HeteroGNNModel`` weights (not trained GNN). For relative
    speed vs Path A, compare ``ms_per_forward`` across machines and snapshot sizes.
    """
    with open_text_auto(snapshot_path, "r") as handle:
        snapshot = json.load(handle)
    origin = _origin_date_from_snapshot(snapshot)
    rows = _feature_rows_for_snapshot(snapshot, origin)
    data = build_graph_from_snapshot(snapshot=snapshot, feature_rows=rows)
    dev = torch.device(device)
    loc_f = int(data["location"].x.shape[1])
    evt_f = int(data["event"].x.shape[1])
    if "actor" in data.node_types and getattr(data["actor"], "x", None) is not None and data["actor"].x.shape[0] > 0:
        act_f = int(data["actor"].x.shape[1])
    else:
        act_f = 0
    model = HeteroGNNModel(
        location_feature_dim=loc_f,
        event_feature_dim=evt_f,
        actor_feature_dim=act_f,
        hidden_dim=64,
    ).to(dev)
    model.eval()
    data = data.to(dev)
    path_proj = make_path_proj(model, data, device=dev)
    head = ForecastHead().to(dev)
    y = torch.tensor([[0.5]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        forward_path_b(data, model, path_proj, head, return_metrics=True, y=y)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            forward_path_b(data, model, path_proj, head, return_metrics=True, y=y)
    elapsed = time.perf_counter() - t0
    return {
        "snapshot": str(snapshot_path.resolve()),
        "repeats": repeats,
        "seconds_total": elapsed,
        "ms_per_forward": 1000.0 * elapsed / float(repeats),
        "device": device,
    }


__all__ = [
    "FORECAST_HEAD_IN_DIM",
    "bench_path_b_wall_clock",
    "forward_path_a",
    "forward_path_b",
    "legacy_dim_for_path_proj",
    "make_path_proj",
    "run_dual_path_epoch_metrics",
]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dual-path helpers (Path A bag vs Path B legacy).")
    p.add_argument(
        "--bench-path-b",
        type=Path,
        metavar="SNAPSHOT.json",
        help="Wall-clock Path B forwards on one graph snapshot (random GNN weights).",
    )
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.bench_path_b is not None:
        out = bench_path_b_wall_clock(snapshot_path=args.bench_path_b, repeats=args.repeats, device=args.device)
        print(json.dumps(out, indent=2))
    else:
        p.print_help()
        sys.exit(2)
