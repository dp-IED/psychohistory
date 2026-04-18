"""Tests for :mod:`baselines.train_loop_skeleton`."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from baselines.features import FEATURE_NAMES
from baselines.train_loop_skeleton import (
    LINEAR_SKELETON_DEFAULTS,
    LinearOccurrenceModel,
    assert_mondays,
    brier_for_constant_predictor,
    collect_samples_for_origins,
    run_linear_skeleton_cli,
)
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import init_warehouse, upsert_records


def test_brier_always_positive_matches_one_minus_prevalence() -> None:
    ys = [0.0, 1.0, 1.0, 0.0]
    b = brier_for_constant_predictor(ys, 1.0)
    assert abs(b - (1.0 - sum(ys) / len(ys))) < 1e-9


def test_brier_train_prevalence_is_mean_squared_error_to_p() -> None:
    ys = [0.0, 1.0]
    p = 0.25
    assert abs(brier_for_constant_predictor(ys, p) - ((p - 0) ** 2 + (p - 1) ** 2) / 2) < 1e-9


def test_assert_mondays_rejects_non_monday() -> None:
    tuesday = dt.date(2024, 1, 2)
    with pytest.raises(ValueError, match="Monday"):
        assert_mondays(tuesday)


def test_linear_occurrence_one_optimizer_step_finite_loss_synthetic() -> None:
    torch.manual_seed(0)
    n, f = 32, 8
    x = torch.randn(n, f)
    y = (torch.rand(n) > 0.5).float()
    mask = torch.ones(n, dtype=torch.bool)
    model = LinearOccurrenceModel(f)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    logits = model(x)
    loss0 = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
    assert torch.isfinite(loss0)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    logits1 = model(x)
    loss1 = F.binary_cross_entropy_with_logits(logits1[mask], y[mask])
    assert torch.isfinite(loss1)


def _minimal_record(
    *,
    event_date: dt.date,
    source_at: dt.datetime,
    admin1: str = "FR11",
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id="e1",
        event_date=event_date,
        source_available_at=source_at,
        retrieved_at=source_at,
        country_code="FR",
        admin1_code=admin1,
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


def test_collect_samples_aligns_with_monday_origin() -> None:
    """One origin, one region — pipeline produces finite tensors."""
    origin = dt.date(2024, 1, 1)
    assert origin.weekday() == 0
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    records = [
        _minimal_record(event_date=dt.date(2023, 6, 1), source_at=early, admin1="FR11"),
    ]
    samples = collect_samples_for_origins(
        records=records,
        origins=[origin],
        scoring_universe=["FR11"],
        source_names=None,
        feature_names=["event_count_prev_1w", "event_count_prev_4w"],
        excluded_admin1=set(),
    )
    assert len(samples) >= 1
    assert samples.x.shape[1] == 2
    assert samples.y.shape[0] == samples.x.shape[0]


def test_collect_samples_rejects_unknown_feature_name() -> None:
    origin = dt.date(2024, 1, 1)
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    records = [
        _minimal_record(event_date=dt.date(2023, 6, 1), source_at=early, admin1="FR11"),
    ]
    with pytest.raises(ValueError, match="feature_names"):
        collect_samples_for_origins(
            records=records,
            origins=[origin],
            scoring_universe=["FR11"],
            source_names=None,
            feature_names=["not_a_real_feature_column"],
            excluded_admin1=set(),
        )


def test_run_linear_rejects_overlapping_train_holdout(tmp_path: Path) -> None:
    """Window validation runs before training; warehouse may be empty."""
    db = tmp_path / "events.duckdb"
    init_warehouse(db)
    with pytest.raises(ValueError, match="train_origin_end"):
        run_linear_skeleton_cli(
            warehouse_path=db,
            data_root=None,
            train_origin_start=dt.date(2020, 1, 6),
            train_origin_end=dt.date(2023, 1, 2),
            holdout_origin_start=dt.date(2023, 1, 2),
            holdout_origin_end=dt.date(2023, 12, 25),
            epochs=1,
            lr=1e-2,
            batch_size=8,
            device=torch.device("cpu"),
            source_names=None,
            feature_names=["event_count_prev_1w"],
            excluded_admin1=set(),
        )


def test_run_linear_rejects_invalid_epochs(tmp_path: Path) -> None:
    db = tmp_path / "events.duckdb"
    init_warehouse(db)
    with pytest.raises(ValueError, match="epochs"):
        run_linear_skeleton_cli(
            warehouse_path=db,
            data_root=None,
            train_origin_start=dt.date(2020, 1, 6),
            train_origin_end=dt.date(2020, 1, 27),
            holdout_origin_start=dt.date(2021, 1, 4),
            holdout_origin_end=dt.date(2021, 1, 11),
            epochs=0,
            lr=1e-2,
            batch_size=8,
            device=torch.device("cpu"),
            source_names=None,
            feature_names=["event_count_prev_1w"],
            excluded_admin1=set(),
        )


def test_run_linear_smoke_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "events.duckdb"
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    upsert_records(
        db_path=db,
        records=[
            _minimal_record(event_date=dt.date(2023, 6, 1), source_at=early, admin1="FR11"),
        ],
    )
    out = run_linear_skeleton_cli(
        warehouse_path=db,
        data_root=None,
        train_origin_start=dt.date(2024, 1, 1),
        train_origin_end=dt.date(2024, 1, 1),
        holdout_origin_start=dt.date(2024, 1, 8),
        holdout_origin_end=dt.date(2024, 1, 8),
        epochs=1,
        lr=1e-2,
        batch_size=4,
        device=torch.device("cpu"),
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
    )
    assert out["epochs"] == 1
    assert out["epochs_requested"] == 1
    assert out["best_epoch"] == 1
    assert "baselines" in out
    assert out["early_stopped"] is False


def test_linear_skeleton_defaults_are_consistent() -> None:
    assert LINEAR_SKELETON_DEFAULTS["epochs"] >= 1
    assert LINEAR_SKELETON_DEFAULTS["batch_size"] >= 1
    assert LINEAR_SKELETON_DEFAULTS["history_weeks"] >= 1


def test_run_linear_golden_deterministic_cpu(tmp_path: Path) -> None:
    db = tmp_path / "events.duckdb"
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    upsert_records(
        db_path=db,
        records=[
            _minimal_record(event_date=dt.date(2023, 6, 1), source_at=early, admin1="FR11"),
        ],
    )
    torch.manual_seed(42)
    out = run_linear_skeleton_cli(
        warehouse_path=db,
        data_root=None,
        train_origin_start=dt.date(2024, 1, 1),
        train_origin_end=dt.date(2024, 1, 1),
        holdout_origin_start=dt.date(2024, 1, 8),
        holdout_origin_end=dt.date(2024, 1, 8),
        epochs=3,
        lr=0.01,
        batch_size=64,
        device=torch.device("cpu"),
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
        progress=False,
        early_stop_patience=0,
        variant="linear",
    )
    assert out["variant"] == "linear"
    assert out["epochs"] == 3
    assert out["best_epoch"] == 3
    assert out["train_rows"] == 1
    assert out["holdout_rows"] == 1
    assert out["baselines"]["holdout_masked_row_count"] == 1
    assert abs(out["baselines"]["train_masked_prevalence"] - 0.0) < 1e-9
    assert abs(out["baselines"]["holdout_prevalence"] - 0.0) < 1e-9
    assert abs(out["baselines"]["brier_always_positive"] - 1.0) < 1e-9
    assert abs(out["baselines"]["brier_predict_train_prevalence"] - 0.0) < 1e-9
    assert abs(out["best_holdout_brier"] - 0.4470410545355321) < 1e-6
    assert abs(out["last_holdout_brier"] - 0.4470410545355321) < 1e-6


def test_run_linear_no_early_stop_runs_all_epochs(tmp_path: Path) -> None:
    db = tmp_path / "events.duckdb"
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    upsert_records(
        db_path=db,
        records=[
            _minimal_record(event_date=dt.date(2023, 6, 1), source_at=early, admin1="FR11"),
        ],
    )
    out = run_linear_skeleton_cli(
        warehouse_path=db,
        data_root=None,
        train_origin_start=dt.date(2024, 1, 1),
        train_origin_end=dt.date(2024, 1, 1),
        holdout_origin_start=dt.date(2024, 1, 8),
        holdout_origin_end=dt.date(2024, 1, 8),
        epochs=4,
        lr=1e-2,
        batch_size=4,
        device=torch.device("cpu"),
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
        early_stop_patience=0,
    )
    assert out["epochs"] == 4
    assert out["early_stopped"] is False
