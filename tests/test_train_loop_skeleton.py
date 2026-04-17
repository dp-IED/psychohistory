"""Tests for :mod:`baselines.train_loop_skeleton`."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from baselines.train_loop_skeleton import (
    LinearOccurrenceModel,
    assert_mondays,
    collect_samples_for_origins,
    run_linear_skeleton_cli,
)
from ingest.event_tape import EventTapeRecord


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


def test_run_linear_rejects_overlapping_train_holdout(tmp_path) -> None:
    """Validation runs before loading tape — empty path is never read."""
    tape = tmp_path / "unused.jsonl"
    tape.write_text("")
    with pytest.raises(ValueError, match="train_origin_end"):
        run_linear_skeleton_cli(
            tape_path=tape,
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


def test_run_linear_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        run_linear_skeleton_cli(
            tape_path=Path("/nonexistent/only/epochs/matter.jsonl"),
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
