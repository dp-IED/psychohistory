from __future__ import annotations

import datetime as dt

import pytest
import torch

from baselines.gnn import (
    GNNForecastRow,
    HeteroGNNModel,
    build_graph_from_snapshot,
    predict_gnn,
    train_gnn,
)
from baselines.features import FEATURE_NAMES, FeatureRow


def _minimal_snapshot(origin_date: dt.date, n_events: int = 3) -> dict:
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES

    universe = ["FR11", "FR22"]
    nodes = [
        {"id": "source:gdelt_v2_events", "type": "Source", "label": "GDELT", "provenance": {"sources": ["gdelt_v2_events"]}},
    ]
    for code in universe:
        nodes.append({
            "id": f"location:FR:{code}",
            "type": "Location",
            "label": code,
            "external_ids": {"gdelt_adm1": code},
            "attributes": {"admin1_code": code, "country_code": "FR"},
            "provenance": {"sources": ["gdelt_v2_events"]},
        })
    edges = []
    for i in range(n_events):
        eid = f"event:gdelt:{i}"
        nodes.append({
            "id": eid,
            "type": "Event",
            "label": f"event {i}",
            "external_ids": {"gdelt": eid},
            "time": {"start": (origin_date - dt.timedelta(days=i+1)).isoformat(), "granularity": "day"},
            "provenance": {"sources": ["gdelt_v2_events"]},
            "attributes": {
                "admin1_code": "FR11",
                "source_event_id": eid,
                "source_available_at": "2021-01-06T00:00:00Z",
                "event_class": "protest",
                "event_code": "141",
                "event_base_code": "14",
                "event_root_code": "14",
                "goldstein_scale": -6.5,
                "avg_tone": -1.5,
                "num_mentions": 4,
                "num_articles": 3,
                "num_sources": 1,
                "source_url": None,
            },
        })
        edges.append({
            "source": eid,
            "target": "location:FR:FR11",
            "type": "occurs_in",
            "time": {"start": (origin_date - dt.timedelta(days=i+1)).isoformat(), "granularity": "day"},
            "provenance": {"sources": ["gdelt_v2_events"]},
            "attributes": {"source_event_id": eid},
        })

    target_table = []
    for code in universe:
        target_table.append({
            "target_id": f"france_protest:{origin_date.isoformat()}:{code}:count_next_7d",
            "name": "target_count_next_7d",
            "value": 1 if code == "FR11" else 0,
            "split": "development",
            "slice_id": origin_date.isoformat(),
            "node_ids": [f"location:FR:{code}"],
            "metadata": {
                "admin1_code": code,
                "forecast_origin": f"{origin_date.isoformat()}T00:00:00Z",
                "window_start": origin_date.isoformat(),
                "window_end_exclusive": (origin_date + dt.timedelta(days=7)).isoformat(),
                "label_grace_days": 14,
            },
        })
        target_table.append({
            "target_id": f"france_protest:{origin_date.isoformat()}:{code}:occurs_next_7d",
            "name": "target_occurs_next_7d",
            "value": code == "FR11",
            "split": "development",
            "slice_id": origin_date.isoformat(),
            "node_ids": [f"location:FR:{code}"],
            "metadata": {
                "admin1_code": code,
                "forecast_origin": f"{origin_date.isoformat()}T00:00:00Z",
                "window_start": origin_date.isoformat(),
                "window_end_exclusive": (origin_date + dt.timedelta(days=7)).isoformat(),
                "label_grace_days": 14,
            },
        })

    return {
        "artifact_format": "graph_artifact_v1",
        "probe_id": f"test_{origin_date.isoformat()}",
        "schema_version": "0.2.0",
        "nodes": nodes,
        "edges": edges,
        "task_labels": [],
        "target_table": target_table,
        "metadata": {
            "domain": "france_protest",
            "forecast_origin": f"{origin_date.isoformat()}T00:00:00Z",
            "window_days": 7,
            "label_grace_days": 14,
            "feature_record_count": n_events,
            "label_record_count": 1,
        },
    }


def _minimal_feature_rows(origin_date: dt.date, universe: list[str]) -> list[FeatureRow]:
    return [
        FeatureRow(
            forecast_origin=origin_date,
            admin1_code=code,
            features={name: float(i) for i, name in enumerate(FEATURE_NAMES)},
        )
        for code in universe
    ]


def test_build_graph_from_snapshot_returns_heterodata() -> None:
    from torch_geometric.data import HeteroData

    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    assert isinstance(data, HeteroData)
    assert data["location"].x.shape[1] == len(FEATURE_NAMES)
    assert data["event"].x.shape[1] == 4
    assert ("event", "occurs_in", "location") in data.edge_types


def test_gnn_model_forward_pass_returns_location_logits() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin, n_events=5)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        hidden_dim=16,
    )
    logits = model(data)

    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_train_gnn_runs_without_error() -> None:
    origins = [dt.date(2021, 1, 4) + dt.timedelta(weeks=i) for i in range(4)]
    graphs = []
    for origin in origins:
        snap = _minimal_snapshot(origin)
        feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
        graphs.append(build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows))

    model = train_gnn(graphs=graphs, epochs=2, lr=0.01, hidden_dim=16)

    assert isinstance(model, HeteroGNNModel)


def test_predict_gnn_returns_one_row_per_location_per_origin() -> None:
    train_origins = [dt.date(2021, 1, 4) + dt.timedelta(weeks=i) for i in range(4)]
    train_graphs = []
    for origin in train_origins:
        snap = _minimal_snapshot(origin)
        feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
        train_graphs.append(build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows))

    model = train_gnn(graphs=train_graphs, epochs=2, lr=0.01, hidden_dim=16)

    eval_origin = dt.date(2021, 2, 8)
    eval_snap = _minimal_snapshot(eval_origin)
    eval_feature_rows = _minimal_feature_rows(eval_origin, ["FR11", "FR22"])
    eval_graph = build_graph_from_snapshot(snapshot=eval_snap, feature_rows=eval_feature_rows)

    predictions = predict_gnn(model=model, graph=eval_graph, origin_date=eval_origin)

    assert len(predictions) == 2
    for pred in predictions:
        assert isinstance(pred, GNNForecastRow)
        assert pred.model_name == "gnn_sage"
        assert 0.0 <= pred.predicted_occurrence_probability <= 1.0
