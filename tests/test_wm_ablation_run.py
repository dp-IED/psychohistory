from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import torch

from baselines.features import FEATURE_NAMES
from baselines.gnn import build_graph_from_snapshot
from baselines.wm_ablation_run import run_wm_ablation_cli, wm_ablation_aggregate
from baselines.wm_ablation_train import build_loc_temporal_for_graph
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import upsert_records
from test_gnn import _minimal_feature_rows, _minimal_snapshot


def _minimal_tape_record() -> EventTapeRecord:
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id="e1",
        event_date=dt.date(2023, 6, 1),
        source_available_at=early,
        retrieved_at=early,
        country_code="FR",
        admin1_code="FR11",
        location_name="Paris",
        latitude=48.0,
        longitude=2.0,
        event_class="protest",
        event_code="14",
        event_base_code="14",
        event_root_code="14",
        quad_class=1,
        goldstein_scale=-2.0,
        num_mentions=1,
        num_sources=1,
        num_articles=1,
        avg_tone=-1.0,
        actor1_name=None,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def test_build_loc_temporal_matches_graph_location_order() -> None:
    origin = dt.date(2024, 1, 1)
    snap = _minimal_snapshot(origin, n_events=2)
    rows = _minimal_feature_rows(origin, ["FR22", "FR11"])
    graph = build_graph_from_snapshot(snapshot=snap, feature_rows=rows)
    codes = list(graph["location"].admin1_codes)
    xt, mt = build_loc_temporal_for_graph(
        graph,
        origin,
        records=[_minimal_tape_record()],
        scoring_universe=["FR11", "FR22"],
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
        history_weeks=3,
    )
    assert xt.shape == (len(codes), 3, len(FEATURE_NAMES))
    assert mt.shape == (len(codes), 3)
    assert xt.shape[0] == len(codes)


def test_run_wm_ablation_smoke_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "events.duckdb"
    upsert_records(db_path=db, records=[_minimal_tape_record()])
    torch.manual_seed(0)
    rows = run_wm_ablation_cli(
        warehouse_path=db,
        data_root=None,
        train_origin_start=dt.date(2024, 1, 1),
        train_origin_end=dt.date(2024, 1, 1),
        holdout_origin_start=dt.date(2024, 1, 8),
        holdout_origin_end=dt.date(2024, 1, 8),
        variants=["linear", "gnn"],
        seed=1,
        history_weeks=4,
        epochs=1,
        lr=0.01,
        batch_size=8,
        early_stop_patience=0,
        device=torch.device("cpu"),
        progress=False,
    )
    assert len(rows) == 2
    for row in rows:
        assert "variant" in row
        b = row["baselines"]
        assert "holdout_masked_row_count" in b
        assert "train_masked_prevalence" in b
        assert "holdout_prevalence" in b
        assert "brier_always_positive" in b
        assert "brier_predict_train_prevalence" in b
        assert "best_holdout_brier" in row
        assert "best_epoch" in row
        assert "early_stop_patience" in row
        hm = row["holdout_metrics"]
        assert set(hm.keys()) >= {
            "brier",
            "log_loss",
            "pr_auc",
            "balanced_accuracy",
            "label_prevalence",
            "mean_prediction",
            "brier_skill_score",
        }
        for k, v in hm.items():
            assert isinstance(k, str)
            assert isinstance(v, float)
            assert math.isnan(v) or math.isfinite(v)
        hmid = row["holdout_mask_identity"]
        assert "keys_sha256" in hmid and "scored_row_count" in hmid
        assert row.get("collapse_detected") is not None


def test_wm_ablation_aggregate_collapses() -> None:
    runs = [
        [
            {"variant": "gnn", "best_holdout_brier": 0.2, "collapse_detected": True},
            {"variant": "linear", "best_holdout_brier": 0.25, "collapse_detected": False},
        ],
        [
            {"variant": "gnn", "best_holdout_brier": 0.22, "collapse_detected": False},
            {"variant": "linear", "best_holdout_brier": 0.24, "collapse_detected": False},
        ],
    ]
    agg = wm_ablation_aggregate(runs, seeds=[0, 1])
    assert agg["event"] == "wm_ablation_aggregate"
    assert agg["by_variant"]["gnn"]["collapse_count"] == 1
    assert agg["by_variant"]["linear"]["collapse_count"] == 0
