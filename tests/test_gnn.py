from __future__ import annotations

import datetime as dt

import pytest
import torch
from torch_geometric.data import HeteroData

from baselines.gnn import (
    ACTOR_FEATURE_DIM,
    ACTOR_QID_HASH_DIM,
    GNNForecastRow,
    GNNGraphAblation,
    HeteroGNNModel,
    build_graph_from_snapshot,
    predict_gnn,
    qid_hash_vector,
    resolve_gnn_graph_ablations,
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


def _snapshot_with_actor_track(origin_date: dt.date) -> dict:
    snap = _minimal_snapshot(origin_date, n_events=4)
    actor_nodes = [
        {
            "id": "actor:students_fr",
            "type": "Actor",
            "label": "Students",
            "external_ids": {"wikidata": "Q48282"},
            "attributes": {"country_code": "FRA"},
            "provenance": {"sources": ["gdelt_v2_events"]},
        },
        {
            "id": "actor:student_union_fr",
            "type": "Actor",
            "label": "Student union",
            "external_ids": {"wikidata": "Q48282"},
            "attributes": {"country_code": "FRA"},
            "provenance": {"sources": ["acled"]},
        },
        {
            "id": "actor:police_fr",
            "type": "Actor",
            "label": "Police",
            "attributes": {"country_code": "FRA"},
            "provenance": {"sources": ["gdelt_v2_events"]},
        },
        {
            "id": "actor:police_be",
            "type": "Actor",
            "label": "Police",
            "attributes": {"country_code": "BEL"},
            "provenance": {"sources": ["gdelt_v2_events"]},
        },
    ]
    snap["nodes"].extend(actor_nodes)
    snap["edges"].extend(
        [
            {
                "source": "actor:students_fr",
                "target": "event:gdelt:0",
                "type": "participates_in",
                "attributes": {"source_name": "gdelt_v2_events", "role": "actor1"},
            },
            {
                "source": "actor:student_union_fr",
                "target": "event:gdelt:1",
                "type": "participates_in",
                "attributes": {"source_name": "acled", "role": "actor2"},
            },
            {
                "source": "actor:police_fr",
                "target": "event:gdelt:2",
                "type": "participates_in",
                "attributes": {"source_name": "gdelt_v2_events", "role": "actor1"},
            },
            {
                "source": "actor:police_be",
                "target": "event:gdelt:3",
                "type": "participates_in",
                "attributes": {"source_name": "gdelt_v2_events", "role": "actor2"},
            },
        ]
    )
    return snap


def _snapshot_with_location_qids(origin_date: dt.date) -> dict:
    snapshot = _minimal_snapshot(origin_date)
    qids_by_admin1 = {
        "FR11": "Q13917",  # Ile-de-France
        "FR22": "Q18677875",  # Hauts-de-France
    }
    for node in snapshot["nodes"]:
        if node["type"] == "Location":
            admin1_code = node["attributes"]["admin1_code"]
            node["external_ids"]["wikidata"] = qids_by_admin1[admin1_code]
    return snapshot


def test_qid_hash_vector_is_stable_for_real_wikidata_qids() -> None:
    ile_de_france = qid_hash_vector("Q13917", 8)

    assert ile_de_france.shape == (8,)
    assert torch.equal(ile_de_france, qid_hash_vector("Q13917", 8))
    assert not torch.equal(ile_de_france, qid_hash_vector("Q18677875", 8))


def test_build_graph_appends_location_qid_hash_features_from_external_ids() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_location_qids(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        qid_feature_mode="hash",
        qid_dim=8,
    )

    assert data["location"].x.shape == (2, len(FEATURE_NAMES) + 8)
    assert torch.equal(data["location"].x[0, len(FEATURE_NAMES):], qid_hash_vector("Q13917", 8))
    assert torch.equal(data["location"].x[1, len(FEATURE_NAMES):], qid_hash_vector("Q18677875", 8))
    assert data.qid_features == {"mode": "hash", "dim": 8, "bucket_count": 4096}


def test_build_graph_uses_zero_hash_features_when_location_qid_is_missing() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        qid_feature_mode="hash",
        qid_dim=8,
    )

    assert torch.count_nonzero(data["location"].x[:, len(FEATURE_NAMES):]).item() == 0


def test_learned_qid_embedding_forward_pass_uses_location_buckets() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_location_qids(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        qid_feature_mode="learned",
        qid_dim=4,
    )

    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        hidden_dim=16,
        qid_dim=4,
        qid_bucket_count=4096,
    )
    logits = model(data)

    assert logits.shape == (2,)
    assert data["location"].qid_bucket.shape == (2,)
    assert data["location"].qid_bucket[0].item() != 0
    assert data["location"].qid_bucket[0].item() != data["location"].qid_bucket[1].item()


def test_build_graph_from_snapshot_returns_heterodata() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    assert isinstance(data, HeteroData)
    assert data["location"].x.shape[1] == len(FEATURE_NAMES)
    assert data["event"].x.shape[1] == 4
    assert ("event", "occurs_in", "location") in data.edge_types


def test_build_graph_from_snapshot_emits_actor_tensor_and_participation_edges() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_actor_track(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    assert data["actor"].x.shape == (3, ACTOR_FEATURE_DIM)
    assert ("actor", "participates_in", "event") in data.edge_types
    assert data["actor", "participates_in", "event"].edge_index.shape == (2, 4)
    assert torch.count_nonzero(data["actor"].x[:, -ACTOR_QID_HASH_DIM:]).item() > 0
    assert data.actor_diagnostics["actor_nodes_input"] == 4
    assert data.actor_diagnostics["actor_nodes_output"] == 3


def test_actor_qid_merge_does_not_merge_unresolved_homonym_labels() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_actor_track(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    edge_index = data["actor", "participates_in", "event"].edge_index
    assert edge_index[0, 0].item() == edge_index[0, 1].item()
    assert edge_index[0, 2].item() != edge_index[0, 3].item()
    assert data.actor_diagnostics["actor_nodes_collapsed_by_qid"] == 1
    assert data.actor_diagnostics["unresolved_label_collision_count"] == 1


def test_actor_graph_ablations_can_disable_or_zero_actor_features() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_actor_track(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    disabled = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        ablation=GNNGraphAblation(
            name="no_actor_track",
            use_actor_nodes=False,
            use_actor_edges=False,
        ),
    )
    structure_only = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        ablation=GNNGraphAblation(
            name="actor_structure_only",
            actor_feature_mode="zero",
        ),
    )

    assert "actor" not in disabled.node_types
    assert torch.count_nonzero(structure_only["actor"].x).item() == 0
    assert structure_only["actor", "participates_in", "event"].edge_index.shape == (2, 4)


def test_build_graph_from_snapshot_applies_graph_ablations() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])

    no_location_features = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        ablation=GNNGraphAblation(name="no_location_features", use_location_features=False),
    )
    no_event_features = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        ablation=GNNGraphAblation(name="no_event_features", use_event_features=False),
    )
    no_event_edges = build_graph_from_snapshot(
        snapshot=snap,
        feature_rows=feature_rows,
        ablation=GNNGraphAblation(name="no_event_edges", use_event_edges=False),
    )

    assert torch.count_nonzero(no_location_features["location"].x).item() == 0
    assert torch.count_nonzero(no_event_features["event"].x).item() == 0
    assert no_event_edges["event", "occurs_in", "location"].edge_index.shape == (2, 0)
    assert no_event_edges.graph_ablation["name"] == "no_event_edges"


def test_resolve_gnn_graph_ablations_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown GNN ablation"):
        resolve_gnn_graph_ablations(["not_real"])


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


def test_gnn_model_actor_branch_receives_training_signal() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_actor_track(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)

    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        actor_feature_dim=ACTOR_FEATURE_DIM,
        hidden_dim=16,
    )
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.01)

    logits = model(data)
    mask = data["location"].mask
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], data["location"].y[mask]
    )
    loss.backward()

    assert model.actor_proj is not None
    assert model.actor_proj.weight.grad is not None
    assert model.actor_proj.weight.grad.abs().sum().item() > 0


@pytest.mark.torch_train
def test_train_gnn_runs_without_error() -> None:
    origins = [dt.date(2021, 1, 4) + dt.timedelta(weeks=i) for i in range(4)]
    graphs = []
    for origin in origins:
        snap = _minimal_snapshot(origin)
        feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
        graphs.append(build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows))

    model = train_gnn(graphs=graphs, epochs=2, lr=0.01, hidden_dim=16)

    assert isinstance(model, HeteroGNNModel)


@pytest.mark.torch_train
def test_gnn_backtest_writes_output(tmp_path: Path) -> None:
    import json

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    out_path = tmp_path / "gnn_predictions.jsonl"

    from ingest.event_tape import EventTapeRecord
    from ingest.event_warehouse import init_warehouse, upsert_records

    records = []
    for week in range(12):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        records.append(
            EventTapeRecord(
                source_name="gdelt_v2_events",
                source_event_id=f"gdelt:{week}",
                event_date=origin - dt.timedelta(days=3),
                source_available_at=dt.datetime.combine(
                    origin - dt.timedelta(days=2), dt.time(), tzinfo=dt.timezone.utc
                ),
                retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
                country_code="FR",
                admin1_code="FR11",
                location_name=None,
                latitude=None,
                longitude=None,
                event_class="protest",
                event_code="141",
                event_base_code="14",
                event_root_code="14",
                quad_class=3,
                goldstein_scale=-6.5,
                num_mentions=4,
                num_sources=None,
                num_articles=3,
                avg_tone=-1.5,
                actor1_name=None,
                actor1_country_code=None,
                actor2_name=None,
                actor2_country_code=None,
                source_url=None,
                raw={},
            )
        )
    db = tmp_path / "warehouse" / "events.duckdb"
    init_warehouse(db)
    upsert_records(db_path=db, records=records)

    for week in range(12):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        snap = _minimal_snapshot(origin)
        snap_path = snap_dir / f"as_of_{origin.isoformat()}.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

    from baselines.backtest import run_gnn_backtest

    audit = run_gnn_backtest(
        warehouse_path=db,
        snapshots_dir=snap_dir,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 22),
        out_path=out_path,
        epochs=2,
        hidden_dim=16,
    )

    assert out_path.exists()
    assert audit["eval_row_count"] > 0
    assert "brier" in audit
    assert "recall_at_5" in audit


@pytest.mark.torch_train
def test_gnn_backtest_from_payloads_writes_gzip_output(tmp_path: Path) -> None:
    from baselines.backtest import OriginInputs, run_gnn_backtest_from_payloads

    train_inputs = []
    eval_inputs = []
    target_lookup = {}
    for week in range(6):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        snapshot = _snapshot_with_location_qids(origin)
        feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
        item = OriginInputs(origin=origin, snapshot=snapshot, feature_rows=feature_rows)
        for row in snapshot["target_table"]:
            code = row["metadata"]["admin1_code"]
            existing = target_lookup.get((origin, code), (0, False))
            if row["name"] == "target_count_next_7d":
                target_lookup[(origin, code)] = (int(row["value"]), existing[1])
            elif row["name"] == "target_occurs_next_7d":
                target_lookup[(origin, code)] = (existing[0], bool(row["value"]))
        if week < 4:
            train_inputs.append(item)
        else:
            eval_inputs.append(item)

    out_path = tmp_path / "gnn_predictions.jsonl.gz"
    audit = run_gnn_backtest_from_payloads(
        train_inputs=train_inputs,
        eval_inputs=eval_inputs,
        target_lookup=target_lookup,
        out_path=out_path,
        epochs=1,
        hidden_dim=16,
        gnn_qid_features="hash",
        qid_dim=4,
    )

    assert out_path.exists()
    assert audit["eval_row_count"] > 0
    assert audit["qid_features"] == {"mode": "hash", "dim": 4, "bucket_count": 4096}


@pytest.mark.torch_train
def test_gnn_ablation_backtest_writes_variant_audit(tmp_path: Path) -> None:
    import json

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    out_path = tmp_path / "gnn_ablation_predictions.jsonl"

    from ingest.event_tape import EventTapeRecord
    from ingest.event_warehouse import init_warehouse, upsert_records

    records = []
    for week in range(12):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        records.append(
            EventTapeRecord(
                source_name="gdelt_v2_events",
                source_event_id=f"gdelt:{week}",
                event_date=origin - dt.timedelta(days=3),
                source_available_at=dt.datetime.combine(
                    origin - dt.timedelta(days=2), dt.time(), tzinfo=dt.timezone.utc
                ),
                retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
                country_code="FR",
                admin1_code="FR11",
                location_name=None,
                latitude=None,
                longitude=None,
                event_class="protest",
                event_code="141",
                event_base_code="14",
                event_root_code="14",
                quad_class=3,
                goldstein_scale=-6.5,
                num_mentions=4,
                num_sources=None,
                num_articles=3,
                avg_tone=-1.5,
                actor1_name=None,
                actor1_country_code=None,
                actor2_name=None,
                actor2_country_code=None,
                source_url=None,
                raw={},
            )
        )
    db = tmp_path / "warehouse" / "events.duckdb"
    init_warehouse(db)
    upsert_records(db_path=db, records=records)

    for week in range(12):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        snap = _minimal_snapshot(origin)
        snap_path = snap_dir / f"as_of_{origin.isoformat()}.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

    from baselines.backtest import run_gnn_ablation_backtest

    audit = run_gnn_ablation_backtest(
        warehouse_path=db,
        snapshots_dir=snap_dir,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 22),
        out_path=out_path,
        ablation_names=["full_graph", "no_event_edges"],
        epochs=2,
        hidden_dim=16,
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert out_path.exists()
    assert audit["ablation_count"] == 2
    assert {entry["ablation"]["name"] for entry in audit["ablations"]} == {
        "full_graph",
        "no_event_edges",
    }
    assert all("delta_vs_full_graph" in entry for entry in audit["ablations"])
    assert {row["model_name"] for row in rows} == {
        "gnn_sage__full_graph",
        "gnn_sage__no_event_edges",
    }
    assert rows[0]["metadata"]["ablation"]["name"] in {"full_graph", "no_event_edges"}


@pytest.mark.torch_train
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


def _hetero_model_and_actor_graph(hidden_dim: int = 16) -> tuple[HeteroGNNModel, HeteroData]:
    origin = dt.date(2021, 1, 11)
    snap = _snapshot_with_actor_track(origin)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)
    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        actor_feature_dim=ACTOR_FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    return model, data


def test_forward_location_embeddings_shape_matches_head_input() -> None:
    model, data = _hetero_model_and_actor_graph(hidden_dim=32)
    assert isinstance(data, HeteroData)
    num_loc = data["location"].x.shape[0]
    emb = model.forward_location_embeddings(data)
    logits = model(data)

    assert emb.shape == (num_loc, model.hidden_dim)
    assert logits.shape == (num_loc,)
    manual = model.head(emb).squeeze(-1)
    assert torch.allclose(logits, manual, atol=1e-5, rtol=1e-4)


def test_legacy_graph_embedding_mean_pools_locations() -> None:
    model, data = _hetero_model_and_actor_graph(hidden_dim=24)
    assert isinstance(data, HeteroData)
    loc_emb = model.forward_location_embeddings(data)
    legacy = model.legacy_graph_embedding(data)
    assert legacy.shape == (model.hidden_dim,)
    assert torch.allclose(legacy, loc_emb.mean(dim=0))


def test_location_embeddings_match_forward_without_actor() -> None:
    origin = dt.date(2021, 1, 11)
    snap = _minimal_snapshot(origin, n_events=4)
    feature_rows = _minimal_feature_rows(origin, ["FR11", "FR22"])
    data = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)
    model = HeteroGNNModel(
        location_feature_dim=len(FEATURE_NAMES),
        event_feature_dim=4,
        hidden_dim=16,
    )
    emb = model.forward_location_embeddings(data)
    logits = model(data)
    assert torch.allclose(logits, model.head(emb).squeeze(-1))


def test_frozen_model_embeddings_no_grad_under_no_grad() -> None:
    model, data = _hetero_model_and_actor_graph()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        emb = model.forward_location_embeddings(data)
        leg = model.legacy_graph_embedding(data)
    assert not emb.requires_grad
    assert not leg.requires_grad
