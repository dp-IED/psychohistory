from __future__ import annotations

import datetime as dt

import pytest
import torch
import torch.nn.functional as F

from baselines.features import FEATURE_NAMES
from baselines.location_weekly_history import collect_location_weekly_history
from baselines.train_loop_skeleton import collect_samples_for_origins
from ingest.event_tape import EventTapeRecord


def _rec(
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


def test_history_shape_matches_collect_samples_order() -> None:
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    records = [_rec(event_date=dt.date(2023, 6, 1), source_at=early)]
    origin = dt.date(2024, 1, 1)
    su = ["FR11"]
    flat = collect_samples_for_origins(
        records=records,
        origins=[origin],
        scoring_universe=su,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
    )
    hist = collect_location_weekly_history(
        records=records,
        origins=flat.origins,
        admin1_codes=flat.admin1_codes,
        y=flat.y,
        mask=flat.mask,
        scoring_universe=su,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
        history_weeks=4,
    )
    assert hist.x_seq.shape == (1, 4, len(FEATURE_NAMES))
    assert hist.time_mask.shape == (1, 4)
    assert torch.equal(hist.y, flat.y)
    assert torch.equal(hist.mask, flat.mask)


def test_time_mask_false_for_pre_data_weeks() -> None:
    early = dt.datetime(2023, 6, 10, tzinfo=dt.timezone.utc)
    records = [_rec(event_date=dt.date(2023, 6, 5), source_at=early)]
    origin = dt.date(2024, 1, 1)
    su = ["FR11"]
    flat = collect_samples_for_origins(
        records=records,
        origins=[origin],
        scoring_universe=su,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
    )
    hist = collect_location_weekly_history(
        records=records,
        origins=flat.origins,
        admin1_codes=flat.admin1_codes,
        y=flat.y,
        mask=flat.mask,
        scoring_universe=su,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(),
        history_weeks=52,
    )
    assert hist.time_mask.dtype == torch.bool
    assert hist.time_mask[0, 0].item() is False
    assert hist.time_mask[0, -1].item() is True


def test_gru_gradient_smoke_synthetic() -> None:
    from baselines.train_loop_skeleton import OccurrenceGRUModel

    torch.manual_seed(0)
    n, t, f = 4, 5, len(FEATURE_NAMES)
    x = torch.randn(n, t, f, requires_grad=True)
    m = torch.ones(n, t, dtype=torch.bool)
    model = OccurrenceGRUModel(f, hidden_dim=16)
    logits = model(x, m)
    y = torch.zeros(n)
    mask = torch.tensor([True, True, False, True])
    loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_collect_location_weekly_history_rejects_bad_history_weeks() -> None:
    with pytest.raises(ValueError, match="history_weeks"):
        collect_location_weekly_history(
            records=[],
            origins=(),
            admin1_codes=(),
            y=torch.zeros(0),
            mask=torch.zeros(0, dtype=torch.bool),
            scoring_universe=[],
            source_names=None,
            feature_names=list(FEATURE_NAMES),
            excluded_admin1=set(),
            history_weeks=0,
        )
