# Tabular Baseline → GNN Baseline Implementation Plan

> Status: completed as the first validation benchmark. This document is retained
> as implementation history, not as the current product scope. France protest
> forecasting validated the method; it is now the regression benchmark for
> broader graph forecasting work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-engineered tabular baseline (XGBoost) that beats recurrence, then build a GNN baseline that beats the tabular model, both evaluated on the same rolling regional France-protest benchmark.

**Architecture:** Two phases in sequence. Phase 1 extracts per-origin/admin1 feature vectors from the event tape (intensity, tone, momentum, actor signals) and trains XGBoost with rolling-origin cross-validation. Phase 2 builds a heterogeneous temporal GNN over the weekly graph snapshots, using the same feature vectors as initial node embeddings plus graph neighborhood aggregation, and trains with the same split. The completed benchmark shows the GNN beating tabular on the main calibration/error metrics, with ranking metrics still requiring ablation and tuning.

**Tech Stack:** Python 3.11+, pydantic, scikit-learn, xgboost, torch, torch_geometric. No new external dependencies needed.

---

## Data Shape Reference

Event node attributes available per feature event (from `data/gdelt/snapshots/france_protest/as_of_*.json`):
- `admin1_code`, `event_code`, `event_base_code`, `avg_tone`, `goldstein_scale`, `num_mentions`, `num_sources`, `num_articles`, `source_available_at`

Baseline `ForecastRow` shape (from `baselines/recurrence.py`):
```python
forecast_origin: dt.date
admin1_code: str
model_name: str
predicted_count: float
predicted_occurrence_probability: float
target_count_next_7d: int
target_occurs_next_7d: bool
metadata: dict
```

Scoring universe: 22 clean regional admin1 codes (excludes `FR`, `FR00`, `FR_UNKNOWN`).
Development split: 2021-01-04 to 2024-12-30. Holdout: 2025-01-06 to 2025-12-29.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `baselines/features.py` | **Create** | Extract per-origin/admin1 feature vectors from event tape |
| `baselines/tabular.py` | **Create** | XGBoost rolling-origin training, prediction, serialization |
| `baselines/gnn.py` | **Create** | PyG GNN model definition and training loop |
| `baselines/backtest.py` | **Modify** | Add `tabular` and `gnn` subcommands alongside `recurrence` |
| `baselines/metrics.py` | **Modify** | Add `recall_at_k` metric |
| `tests/test_features.py` | **Create** | Unit tests for feature extraction |
| `tests/test_tabular.py` | **Create** | Unit tests for tabular model training and prediction |
| `tests/test_gnn.py` | **Create** | Unit tests for GNN forward pass and training step |
| `tests/test_baselines.py` | **Modify** | Add `recall_at_k` metric test |

---

## Task 1: `recall_at_k` metric

**Files:**
- Modify: `baselines/metrics.py`
- Modify: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_baselines.py`:

```python
def test_recall_at_k_captures_fraction_of_positives() -> None:
    rows = [
        ForecastRow(
            forecast_origin=dt.date(2021, 1, 4),
            admin1_code=f"FR{i:02d}",
            model_name="m",
            predicted_count=float(10 - i),
            predicted_occurrence_probability=float(10 - i) / 10,
            target_count_next_7d=1 if i < 4 else 0,
            target_occurs_next_7d=i < 4,
        )
        for i in range(10)
    ]

    from baselines.metrics import recall_at_k
    # top 5 ranked = FR00..FR04; true positives are FR00..FR03 (4 total)
    # captured = 4 out of 4 total positives
    assert recall_at_k(rows, "m", k=5) == pytest.approx(1.0)
    # top 2 ranked = FR00, FR01; captures 2/4 positives
    assert recall_at_k(rows, "m", k=2) == pytest.approx(0.5)
    # k=0 should return 0.0
    assert recall_at_k(rows, "m", k=0) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_baselines.py::test_recall_at_k_captures_fraction_of_positives -v
```

Expected: `ImportError: cannot import name 'recall_at_k'`

- [ ] **Step 3: Implement `recall_at_k` in `baselines/metrics.py`**

Add after `top_k_hit_rate`:

```python
def recall_at_k(rows: list[ForecastRow], model_name: str, k: int = 5) -> float:
    if k <= 0:
        return 0.0
    selected = _rows_for_model(rows, model_name)
    by_origin: dict[object, list[ForecastRow]] = defaultdict(list)
    for row in selected:
        by_origin[row.forecast_origin].append(row)

    total_positives = 0
    captured = 0
    for origin_rows in by_origin.values():
        ranked = sorted(
            origin_rows,
            key=lambda row: (-row.predicted_count, row.admin1_code),
        )
        top_k = ranked[:k]
        captured += sum(row.target_occurs_next_7d for row in top_k)
        total_positives += sum(row.target_occurs_next_7d for row in origin_rows)
    if total_positives == 0:
        return 0.0
    return captured / total_positives
```

Also add `from collections import defaultdict` at the top of `metrics.py` if not present.

- [ ] **Step 4: Run tests to verify**

```bash
pytest tests/test_baselines.py -v
```

Expected: all pass.

- [ ] **Step 5: Add `recall_at_k` to the backtest audit**

In `baselines/backtest.py`, inside `build_audit`, add alongside `top5_hit_rate_by_model`:

```python
"recall_at_5_by_model": {
    model_name: recall_at_k(rows, model_name, k=5)
    for model_name in model_names
},
```

Add the import: `from baselines.metrics import brier_score, mean_absolute_error, top_k_hit_rate, recall_at_k`

- [ ] **Step 6: Commit**

```bash
git add baselines/metrics.py baselines/backtest.py tests/test_baselines.py
git commit -m "feat: add recall_at_k metric and include in backtest audit"
```

---

## Task 2: Feature extraction (`baselines/features.py`)

**Files:**
- Create: `baselines/features.py`
- Create: `tests/test_features.py`

The feature set per (origin, admin1_code) covers:

| Feature | Description |
|---------|-------------|
| `event_count_prev_1w` | Events in prior 7 days visible at origin |
| `event_count_prev_4w` | Events in prior 4 weeks visible at origin |
| `event_count_prev_12w` | Events in prior 12 weeks visible at origin |
| `rate_change_1w_vs_4w` | `prev_1w / (prev_4w / 4)` — momentum signal, 0 if denominator 0 |
| `mean_goldstein_prev_4w` | Mean Goldstein scale of events in prior 4 weeks, 0 if none |
| `mean_avg_tone_prev_4w` | Mean avg_tone of events in prior 4 weeks, 0 if none |
| `mean_num_mentions_prev_4w` | Mean num_mentions in prior 4 weeks, 0 if none |
| `mean_num_articles_prev_4w` | Mean num_articles in prior 4 weeks, 0 if none |
| `distinct_actor_count_prev_4w` | Count of distinct actor1_name values in prior 4 weeks |
| `national_event_count_prev_1w` | All FR+FR00 events in prior 7 days (national pressure signal) |
| `national_event_count_prev_4w` | All FR+FR00 events in prior 4 weeks |
| `weeks_since_last_event` | Number of full weeks since most recent prior event, capped at 52 |
| `admin1_code_idx` | Integer index of admin1_code in sorted scoring universe (for GNN lookup) |

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
from __future__ import annotations

import datetime as dt

import pytest

from baselines.features import (
    FEATURE_NAMES,
    FeatureRow,
    build_feature_matrix,
    extract_features_for_origin,
)
from ingest.event_tape import EventTapeRecord


def _record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
    actor1_name: str | None = None,
    avg_tone: float | None = None,
    goldstein_scale: float | None = None,
    num_mentions: int | None = None,
    num_articles: int | None = None,
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=dt.date.fromisoformat(event_date),
        source_available_at=dt.datetime.fromisoformat(
            source_available_at.replace("Z", "+00:00")
        ),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name="Paris",
        latitude=None,
        longitude=None,
        event_class="protest",
        event_code="141",
        event_base_code="14",
        event_root_code="14",
        quad_class=3,
        goldstein_scale=goldstein_scale,
        num_mentions=num_mentions,
        num_sources=None,
        num_articles=num_articles,
        avg_tone=avg_tone,
        actor1_name=actor1_name,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def test_feature_names_are_stable() -> None:
    assert "event_count_prev_1w" in FEATURE_NAMES
    assert "rate_change_1w_vs_4w" in FEATURE_NAMES
    assert "mean_goldstein_prev_4w" in FEATURE_NAMES
    assert "admin1_code_idx" in FEATURE_NAMES


def test_extract_features_counts_visible_events_only() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:vis", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
        _record("gdelt:future", event_date="2021-01-05", source_available_at="2021-01-12T00:00:00Z"),
        _record("gdelt:old", event_date="2020-11-01", source_available_at="2020-11-02T00:00:00Z"),
    ]
    scoring_universe = ["FR11", "FR22"]

    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=scoring_universe)

    fr11 = next(r for r in rows if r.admin1_code == "FR11")
    assert fr11.features["event_count_prev_1w"] == 1.0
    assert fr11.features["event_count_prev_4w"] == 1.0
    assert fr11.features["event_count_prev_12w"] == 1.0

    fr22 = next(r for r in rows if r.admin1_code == "FR22")
    assert fr22.features["event_count_prev_1w"] == 0.0


def test_rate_change_zero_when_no_4w_baseline() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:r1", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
    ]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    fr11 = rows[0]
    assert fr11.features["rate_change_1w_vs_4w"] == 0.0


def test_mean_tone_zero_when_no_events() -> None:
    origin = dt.date(2021, 1, 11)
    records = []
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    assert rows[0].features["mean_avg_tone_prev_4w"] == 0.0


def test_weeks_since_last_event_is_capped_at_52() -> None:
    origin = dt.date(2022, 1, 10)
    records = [
        _record("gdelt:old", event_date="2019-01-01", source_available_at="2019-01-02T00:00:00Z"),
    ]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=["FR11"])
    assert rows[0].features["weeks_since_last_event"] == 52.0


def test_admin1_code_idx_matches_sorted_position() -> None:
    origin = dt.date(2021, 1, 11)
    records = []
    universe = ["FRA1", "FRB2", "FRC3"]
    rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=universe)
    idxs = {r.admin1_code: r.features["admin1_code_idx"] for r in rows}
    assert idxs == {"FRA1": 0.0, "FRB2": 1.0, "FRC3": 2.0}


def test_build_feature_matrix_shape_and_order() -> None:
    origin = dt.date(2021, 1, 11)
    records = [
        _record("gdelt:vis", event_date="2021-01-05", source_available_at="2021-01-06T00:00:00Z"),
    ]
    universe = ["FR11", "FR22"]
    feature_rows = extract_features_for_origin(records=records, origin_date=origin, scoring_universe=universe)
    X, admin1_codes = build_feature_matrix(feature_rows)

    assert X.shape == (2, len(FEATURE_NAMES))
    assert list(admin1_codes) == ["FR11", "FR22"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'baselines.features'`

- [ ] **Step 3: Implement `baselines/features.py`**

```python
"""Per-origin per-admin1 feature extraction from the event tape."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from ingest.event_tape import EventTapeRecord

UTC = dt.timezone.utc
WINDOW_DAYS = 7
NATIONAL_CODES = frozenset({"FR", "FR00"})

FEATURE_NAMES: list[str] = [
    "event_count_prev_1w",
    "event_count_prev_4w",
    "event_count_prev_12w",
    "rate_change_1w_vs_4w",
    "mean_goldstein_prev_4w",
    "mean_avg_tone_prev_4w",
    "mean_num_mentions_prev_4w",
    "mean_num_articles_prev_4w",
    "distinct_actor_count_prev_4w",
    "national_event_count_prev_1w",
    "national_event_count_prev_4w",
    "weeks_since_last_event",
    "admin1_code_idx",
]


class FeatureRow:
    __slots__ = ("forecast_origin", "admin1_code", "features")

    def __init__(
        self,
        forecast_origin: dt.date,
        admin1_code: str,
        features: dict[str, float],
    ) -> None:
        self.forecast_origin = forecast_origin
        self.admin1_code = admin1_code
        self.features = features


def _visible_before(records: list[EventTapeRecord], origin_dt: dt.datetime) -> list[EventTapeRecord]:
    return [
        r for r in records
        if r.source_available_at.astimezone(UTC) < origin_dt
        and r.event_date < origin_dt.date()
    ]


def extract_features_for_origin(
    *,
    records: list[EventTapeRecord],
    origin_date: dt.date,
    scoring_universe: list[str],
) -> list[FeatureRow]:
    origin_dt = dt.datetime.combine(origin_date, dt.time(), tzinfo=UTC)
    visible = _visible_before(records, origin_dt)

    w1_start = origin_date - dt.timedelta(days=WINDOW_DAYS)
    w4_start = origin_date - dt.timedelta(days=WINDOW_DAYS * 4)
    w12_start = origin_date - dt.timedelta(days=WINDOW_DAYS * 12)

    national_1w = [r for r in visible if r.admin1_code in NATIONAL_CODES and r.event_date >= w1_start]
    national_4w = [r for r in visible if r.admin1_code in NATIONAL_CODES and r.event_date >= w4_start]
    nat_1w_count = float(len(national_1w))
    nat_4w_count = float(len(national_4w))

    code_to_idx = {code: float(i) for i, code in enumerate(scoring_universe)}

    rows: list[FeatureRow] = []
    for admin1_code in scoring_universe:
        prev_1w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w1_start]
        prev_4w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w4_start]
        prev_12w = [r for r in visible if r.admin1_code == admin1_code and r.event_date >= w12_start]
        all_prev = [r for r in visible if r.admin1_code == admin1_code]

        c1w = float(len(prev_1w))
        c4w = float(len(prev_4w))
        c12w = float(len(prev_12w))

        mean_4w = c4w / 4.0
        rate_change = (c1w / mean_4w) if mean_4w > 0 else 0.0

        goldsteins = [r.goldstein_scale for r in prev_4w if r.goldstein_scale is not None]
        tones = [r.avg_tone for r in prev_4w if r.avg_tone is not None]
        mentions = [r.num_mentions for r in prev_4w if r.num_mentions is not None]
        articles = [r.num_articles for r in prev_4w if r.num_articles is not None]
        actors = {r.actor1_name for r in prev_4w if r.actor1_name}

        if all_prev:
            most_recent = max(r.event_date for r in all_prev)
            weeks_gap = min(52.0, (origin_date - most_recent).days / 7.0)
        else:
            weeks_gap = 52.0

        rows.append(
            FeatureRow(
                forecast_origin=origin_date,
                admin1_code=admin1_code,
                features={
                    "event_count_prev_1w": c1w,
                    "event_count_prev_4w": c4w,
                    "event_count_prev_12w": c12w,
                    "rate_change_1w_vs_4w": rate_change,
                    "mean_goldstein_prev_4w": float(np.mean(goldsteins)) if goldsteins else 0.0,
                    "mean_avg_tone_prev_4w": float(np.mean(tones)) if tones else 0.0,
                    "mean_num_mentions_prev_4w": float(np.mean(mentions)) if mentions else 0.0,
                    "mean_num_articles_prev_4w": float(np.mean(articles)) if articles else 0.0,
                    "distinct_actor_count_prev_4w": float(len(actors)),
                    "national_event_count_prev_1w": nat_1w_count,
                    "national_event_count_prev_4w": nat_4w_count,
                    "weeks_since_last_event": weeks_gap,
                    "admin1_code_idx": code_to_idx[admin1_code],
                },
            )
        )
    return rows


def build_feature_matrix(
    feature_rows: list[FeatureRow],
) -> tuple[np.ndarray, list[str]]:
    X = np.array(
        [[row.features[name] for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float32,
    )
    admin1_codes = [row.admin1_code for row in feature_rows]
    return X, admin1_codes
```

- [ ] **Step 4: Run tests to verify**

```bash
pytest tests/test_features.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add baselines/features.py tests/test_features.py
git commit -m "feat: add per-origin admin1 feature extraction"
```

---

## Task 3: XGBoost tabular baseline (`baselines/tabular.py`)

**Files:**
- Create: `baselines/tabular.py`
- Create: `tests/test_tabular.py`

The tabular model trains an XGBoost binary classifier on development-split rows and predicts occurrence probability on holdout. To preserve temporal hygiene, training uses only rows where `forecast_origin < eval_origin_start`. The model is not retrained per origin; it is trained once on the development split and evaluated on the holdout split. This matches Stage 3 of the roadmap: one model, one evaluation, no leakage.

- [ ] **Step 1: Write failing tests**

Create `tests/test_tabular.py`:

```python
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.features import FEATURE_NAMES, FeatureRow, build_feature_matrix, extract_features_for_origin
from baselines.tabular import (
    TabularForecastRow,
    predict_tabular,
    train_tabular_model,
)
from ingest.event_tape import EventTapeRecord


def _record(
    source_event_id: str,
    *,
    event_date: str,
    source_available_at: str,
    admin1_code: str = "FR11",
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name="gdelt_v2_events",
        source_event_id=source_event_id,
        event_date=dt.date.fromisoformat(event_date),
        source_available_at=dt.datetime.fromisoformat(
            source_available_at.replace("Z", "+00:00")
        ),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name=None,
        latitude=None,
        longitude=None,
        event_class="protest",
        event_code="141",
        event_base_code="14",
        event_root_code="14",
        quad_class=3,
        goldstein_scale=None,
        num_mentions=None,
        num_sources=None,
        num_articles=None,
        avg_tone=None,
        actor1_name=None,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def _make_feature_rows(n: int, n_pos: int, universe: list[str]) -> list[FeatureRow]:
    rows = []
    for i in range(n):
        code = universe[i % len(universe)]
        rows.append(
            FeatureRow(
                forecast_origin=dt.date(2021, 1, 4) + dt.timedelta(weeks=i),
                admin1_code=code,
                features={name: float(i % 5) for name in FEATURE_NAMES},
            )
        )
    return rows


def test_train_tabular_model_returns_callable() -> None:
    universe = ["FR11", "FR22"]
    n = 40
    feature_rows = _make_feature_rows(n, n_pos=20, universe=universe)
    targets = {(r.forecast_origin, r.admin1_code): (i % 2 == 0) for i, r in enumerate(feature_rows)}

    model = train_tabular_model(feature_rows=feature_rows, targets=targets)

    assert callable(model)


def test_predict_tabular_returns_one_row_per_admin1() -> None:
    universe = ["FR11", "FR22", "FR33"]
    n = 60
    feature_rows = _make_feature_rows(n, n_pos=30, universe=universe)
    targets = {(r.forecast_origin, r.admin1_code): (hash(r.admin1_code) % 2 == 0) for r in feature_rows}
    model = train_tabular_model(feature_rows=feature_rows, targets=targets)

    origin = dt.date(2021, 3, 1)
    eval_rows = [
        FeatureRow(
            forecast_origin=origin,
            admin1_code=code,
            features={name: 1.0 for name in FEATURE_NAMES},
        )
        for code in universe
    ]
    predictions = predict_tabular(model=model, feature_rows=eval_rows)

    assert len(predictions) == 3
    for pred in predictions:
        assert isinstance(pred, TabularForecastRow)
        assert 0.0 <= pred.predicted_occurrence_probability <= 1.0
        assert pred.model_name == "xgboost_tabular"


def test_tabular_forecast_row_serializes_to_json() -> None:
    row = TabularForecastRow(
        forecast_origin=dt.date(2021, 1, 4),
        admin1_code="FR11",
        model_name="xgboost_tabular",
        predicted_count=0.5,
        predicted_occurrence_probability=0.5,
        target_count_next_7d=1,
        target_occurs_next_7d=True,
    )
    serialized = row.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["model_name"] == "xgboost_tabular"
    assert parsed["admin1_code"] == "FR11"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tabular.py -v
```

Expected: `ModuleNotFoundError: No module named 'baselines.tabular'`

- [ ] **Step 3: Implement `baselines/tabular.py`**

```python
"""XGBoost tabular baseline for regional France protest forecasting."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import numpy as np
import xgboost as xgb
from pydantic import BaseModel, Field

from baselines.features import FEATURE_NAMES, FeatureRow, build_feature_matrix
from baselines.recurrence import ForecastRow


class TabularForecastRow(BaseModel):
    forecast_origin: dt.date
    admin1_code: str
    model_name: str
    predicted_count: float
    predicted_occurrence_probability: float
    target_count_next_7d: int = 0
    target_occurs_next_7d: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def train_tabular_model(
    *,
    feature_rows: list[FeatureRow],
    targets: dict[tuple[dt.date, str], bool],
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    X, admin1_codes = build_feature_matrix(feature_rows)
    y = np.array(
        [float(targets.get((row.forecast_origin, row.admin1_code), False)) for row in feature_rows],
        dtype=np.float32,
    )
    scale_pos_weight = float(np.sum(y == 0)) / max(1.0, float(np.sum(y == 1)))
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X, y)
    return model.predict_proba


def predict_tabular(
    *,
    model: Callable[[np.ndarray], np.ndarray],
    feature_rows: list[FeatureRow],
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] | None = None,
) -> list[TabularForecastRow]:
    X, admin1_codes = build_feature_matrix(feature_rows)
    proba = model(X)[:, 1]

    rows: list[TabularForecastRow] = []
    for i, feature_row in enumerate(feature_rows):
        key = (feature_row.forecast_origin, feature_row.admin1_code)
        target_count, target_occurs = (target_lookup or {}).get(key, (0, False))
        p = float(proba[i])
        rows.append(
            TabularForecastRow(
                forecast_origin=feature_row.forecast_origin,
                admin1_code=feature_row.admin1_code,
                model_name="xgboost_tabular",
                predicted_count=p,
                predicted_occurrence_probability=p,
                target_count_next_7d=target_count,
                target_occurs_next_7d=target_occurs,
            )
        )
    return rows
```

- [ ] **Step 4: Run tests to verify**

```bash
pytest tests/test_tabular.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add baselines/tabular.py tests/test_tabular.py
git commit -m "feat: add XGBoost tabular baseline"
```

---

## Task 4: Tabular backtest subcommand

**Files:**
- Modify: `baselines/backtest.py`

The tabular backtest:
1. Loads the full event tape.
2. Iterates over all weekly origins in the development split and builds feature rows + target lookup.
3. Trains one XGBoost model on the development split.
4. Iterates over holdout origins and generates predictions.
5. Writes `TabularForecastRow` JSONL and an audit JSON.

Both development and holdout rows are written to the output file for comparison, with `split` field on each row.

- [ ] **Step 1: Write failing test** — add to `tests/test_tabular.py`:

```python
from baselines.backtest import run_tabular_backtest
from pathlib import Path


def test_tabular_backtest_writes_output(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "tabular_predictions.jsonl"

    records = []
    import datetime as dt
    from ingest.event_tape import EventTapeRecord
    for week in range(20):
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
                goldstein_scale=None,
                num_mentions=None,
                num_sources=None,
                num_articles=None,
                avg_tone=None,
                actor1_name=None,
                actor1_country_code=None,
                actor2_name=None,
                actor2_country_code=None,
                source_url=None,
                raw={},
            )
        )

    tape_path.parent.mkdir(parents=True, exist_ok=True)
    tape_path.write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )

    audit = run_tabular_backtest(
        tape_path=tape_path,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 4, 26),
        eval_origin_start=dt.date(2021, 5, 3),
        eval_origin_end=dt.date(2021, 5, 17),
        out_path=out_path,
    )

    assert out_path.exists()
    assert audit["eval_row_count"] > 0
    assert "brier" in audit
    assert "recall_at_5" in audit
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tabular.py::test_tabular_backtest_writes_output -v
```

Expected: `ImportError: cannot import name 'run_tabular_backtest'`

- [ ] **Step 3: Implement `run_tabular_backtest` in `baselines/backtest.py`**

Add imports at top:
```python
from baselines.features import extract_features_for_origin
from baselines.tabular import TabularForecastRow, predict_tabular, train_tabular_model
from baselines.metrics import brier_score, mean_absolute_error, top_k_hit_rate, recall_at_k
```

Add function after `run_recurrence_backtest`:

```python
def run_tabular_backtest(
    *,
    tape_path: Path,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    eval_origin_start: dt.date,
    eval_origin_end: dt.date,
    out_path: Path,
    progress: bool = False,
) -> dict[str, Any]:
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES

    records = load_event_tape(tape_path)
    scoring_universe = sorted(
        {r.admin1_code for r in records if r.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES}
    )

    train_origins = weekly_origins(train_origin_start, train_origin_end)
    eval_origins = weekly_origins(eval_origin_start, eval_origin_end)
    all_origins = train_origins + eval_origins

    feature_rows_by_origin: dict[dt.date, list] = {}
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}

    total = len(all_origins)
    for idx, origin in enumerate(all_origins, start=1):
        rows = extract_features_for_origin(
            records=records,
            origin_date=origin,
            scoring_universe=scoring_universe,
        )
        feature_rows_by_origin[origin] = rows
        from ingest.snapshot_export import build_snapshot_payload
        payload = build_snapshot_payload(records=records, origin_date=origin)
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in EXCLUDED_REGIONAL_ADMIN1_CODES:
                continue
            if row["name"] == "target_count_next_7d":
                existing = target_lookup.get((origin, code), (0, False))
                target_lookup[(origin, code)] = (int(row["value"]), existing[1])
            elif row["name"] == "target_occurs_next_7d":
                existing = target_lookup.get((origin, code), (0, False))
                target_lookup[(origin, code)] = (existing[0], bool(row["value"]))
        if progress:
            print(f"[tabular] features {idx}/{total} origin={origin.isoformat()}", file=sys.stderr, flush=True)

    train_feature_rows = [r for o in train_origins for r in feature_rows_by_origin[o]]
    train_targets = {k: v[1] for k, v in target_lookup.items() if k[0] in set(train_origins)}

    if progress:
        print("[tabular] training XGBoost model...", file=sys.stderr, flush=True)

    model = train_tabular_model(feature_rows=train_feature_rows, targets=train_targets)

    eval_rows: list[TabularForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for origin in eval_origins:
            origin_feature_rows = feature_rows_by_origin[origin]
            preds = predict_tabular(
                model=model,
                feature_rows=origin_feature_rows,
                target_lookup=target_lookup,
            )
            for pred in preds:
                handle.write(pred.model_dump_json() + "\n")
            eval_rows.extend(preds)
        handle.flush()

    model_name = "xgboost_tabular"
    audit: dict[str, Any] = {
        "model_name": model_name,
        "train_origin_start": train_origin_start.isoformat(),
        "train_origin_end": train_origin_end.isoformat(),
        "eval_origin_start": eval_origin_start.isoformat(),
        "eval_origin_end": eval_origin_end.isoformat(),
        "train_row_count": len(train_feature_rows),
        "eval_row_count": len(eval_rows),
        "admin1_count": len(scoring_universe),
        "brier": brier_score(eval_rows, model_name),
        "mae": mean_absolute_error(eval_rows, model_name),
        "top5_hit_rate": top_k_hit_rate(eval_rows, model_name, k=5),
        "recall_at_5": recall_at_k(eval_rows, model_name, k=5),
    }
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
```

Add to `_build_parser` and `main`:

```python
# In _build_parser, add:
tabular = subparsers.add_parser("tabular")
tabular.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
tabular.add_argument("--train-origin-start", default="2021-01-04")
tabular.add_argument("--train-origin-end", default="2024-12-30")
tabular.add_argument("--eval-origin-start", default="2025-01-06")
tabular.add_argument("--eval-origin-end", default="2025-12-29")
tabular.add_argument("--out", default="data/gdelt/baselines/france_protest/tabular_predictions.jsonl")

# In main, add:
if args.command == "tabular":
    try:
        run_tabular_backtest(
            tape_path=Path(args.tape),
            train_origin_start=dt.date.fromisoformat(args.train_origin_start),
            train_origin_end=dt.date.fromisoformat(args.train_origin_end),
            eval_origin_start=dt.date.fromisoformat(args.eval_origin_start),
            eval_origin_end=dt.date.fromisoformat(args.eval_origin_end),
            out_path=Path(args.out),
            progress=True,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
```

- [ ] **Step 4: Run tests to verify**

```bash
pytest tests/test_tabular.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add baselines/backtest.py tests/test_tabular.py baselines/tabular.py
git commit -m "feat: add tabular backtest subcommand"
```

---

## Task 5: GNN model and training loop (`baselines/gnn.py`)

**Files:**
- Create: `baselines/gnn.py`
- Create: `tests/test_gnn.py`

The GNN uses a two-layer heterogeneous message-passing network over the event→location graph in each snapshot. Node initial embeddings: Location nodes are initialized from the 13 tabular features. Event nodes are initialized from goldstein, tone, mentions, articles (4 dims). The GNN aggregates Event→Location messages to produce enriched Location embeddings, then a per-node MLP head predicts occurrence probability.

Training: batched over all development-split weekly snapshots. One graph per origin. Location nodes whose `admin1_code` is in the scoring universe are the prediction targets.

Architecture:
- `HeteroGNNModel`: `torch.nn.Module` with one `SAGEConv` layer per edge type, followed by a 2-layer MLP head per location node.
- `build_graph_from_snapshot`: converts a loaded snapshot JSON into a `torch_geometric.data.HeteroData`.
- `train_gnn`: trains for N epochs over development graphs, returns trained model.
- `predict_gnn`: runs model forward pass on eval graphs, returns `GNNForecastRow` list.

- [ ] **Step 1: Write failing tests**

Create `tests/test_gnn.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_gnn.py -v
```

Expected: `ModuleNotFoundError: No module named 'baselines.gnn'`

- [ ] **Step 3: Implement `baselines/gnn.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify**

```bash
pytest tests/test_gnn.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add baselines/gnn.py tests/test_gnn.py
git commit -m "feat: add heterogeneous GNN model and training loop"
```

---

## Task 6: GNN backtest subcommand

**Files:**
- Modify: `baselines/backtest.py`

The GNN backtest loads weekly snapshots from `data/gdelt/snapshots/france_protest/` directly (already on disk) alongside feature rows from the tape. It trains on development-split graphs and evaluates on holdout graphs.

- [ ] **Step 1: Write failing test** — add to `tests/test_gnn.py`:

```python
from baselines.backtest import run_gnn_backtest
from pathlib import Path
import json


def test_gnn_backtest_writes_output(tmp_path: Path) -> None:
    import datetime as dt

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    tape_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "gnn_predictions.jsonl"

    from ingest.event_tape import EventTapeRecord
    records = []
    for week in range(8):
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
    tape_path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")

    from tests.test_gnn import _minimal_snapshot, _minimal_feature_rows
    from baselines.gnn import build_graph_from_snapshot

    for week in range(8):
        origin = dt.date(2021, 1, 4) + dt.timedelta(weeks=week)
        snap = _minimal_snapshot(origin)
        snap_path = snap_dir / f"as_of_{origin.isoformat()}.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

    audit = run_gnn_backtest(
        tape_path=tape_path,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gnn.py::test_gnn_backtest_writes_output -v
```

Expected: `ImportError: cannot import name 'run_gnn_backtest'`

- [ ] **Step 3: Implement `run_gnn_backtest` in `baselines/backtest.py`**

Add after `run_tabular_backtest`:

```python
def run_gnn_backtest(
    *,
    tape_path: Path,
    snapshots_dir: Path,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    eval_origin_start: dt.date,
    eval_origin_end: dt.date,
    out_path: Path,
    epochs: int = 30,
    hidden_dim: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    import json as _json
    from baselines.features import extract_features_for_origin
    from baselines.gnn import GNNForecastRow, HeteroGNNModel, build_graph_from_snapshot, predict_gnn, train_gnn
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

    records = load_event_tape(tape_path)
    scoring_universe = sorted(
        {r.admin1_code for r in records if r.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES}
    )
    train_origins = weekly_origins(train_origin_start, train_origin_end)
    eval_origins = weekly_origins(eval_origin_start, eval_origin_end)

    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}

    def _load_snapshot(origin: dt.date) -> dict[str, Any]:
        path = snapshots_dir / f"as_of_{origin.isoformat()}.json"
        return _json.loads(path.read_text(encoding="utf-8"))

    def _build_target_lookup_for_origin(origin: dt.date) -> None:
        payload = build_snapshot_payload(records=records, origin_date=origin)
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in EXCLUDED_REGIONAL_ADMIN1_CODES:
                continue
            if row["name"] == "target_count_next_7d":
                existing = target_lookup.get((origin, code), (0, False))
                target_lookup[(origin, code)] = (int(row["value"]), existing[1])
            elif row["name"] == "target_occurs_next_7d":
                existing = target_lookup.get((origin, code), (0, False))
                target_lookup[(origin, code)] = (existing[0], bool(row["value"]))

    total = len(train_origins) + len(eval_origins)
    train_graphs = []
    for idx, origin in enumerate(train_origins, start=1):
        snap = _load_snapshot(origin)
        feature_rows = extract_features_for_origin(
            records=records, origin_date=origin, scoring_universe=scoring_universe
        )
        graph = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)
        train_graphs.append(graph)
        _build_target_lookup_for_origin(origin)
        if progress:
            print(f"[gnn] load train {idx}/{len(train_origins)} origin={origin.isoformat()}", file=sys.stderr, flush=True)

    if progress:
        print(f"[gnn] training GNN epochs={epochs} hidden_dim={hidden_dim}...", file=sys.stderr, flush=True)

    model = train_gnn(graphs=train_graphs, epochs=epochs, hidden_dim=hidden_dim)

    eval_rows: list[GNNForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, origin in enumerate(eval_origins, start=1):
            snap = _load_snapshot(origin)
            feature_rows = extract_features_for_origin(
                records=records, origin_date=origin, scoring_universe=scoring_universe
            )
            graph = build_graph_from_snapshot(snapshot=snap, feature_rows=feature_rows)
            _build_target_lookup_for_origin(origin)
            preds = predict_gnn(
                model=model, graph=graph, origin_date=origin, target_lookup=target_lookup
            )
            for pred in preds:
                handle.write(pred.model_dump_json() + "\n")
            eval_rows.extend(preds)
            if progress:
                print(f"[gnn] eval {idx}/{len(eval_origins)} origin={origin.isoformat()}", file=sys.stderr, flush=True)
        handle.flush()

    model_name = "gnn_sage"
    audit: dict[str, Any] = {
        "model_name": model_name,
        "train_origin_start": train_origin_start.isoformat(),
        "train_origin_end": train_origin_end.isoformat(),
        "eval_origin_start": eval_origin_start.isoformat(),
        "eval_origin_end": eval_origin_end.isoformat(),
        "train_graph_count": len(train_graphs),
        "eval_row_count": len(eval_rows),
        "admin1_count": len(scoring_universe),
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "brier": brier_score(eval_rows, model_name),
        "mae": mean_absolute_error(eval_rows, model_name),
        "top5_hit_rate": top_k_hit_rate(eval_rows, model_name, k=5),
        "recall_at_5": recall_at_k(eval_rows, model_name, k=5),
    }
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
```

Add to `_build_parser` and `main`:

```python
# In _build_parser, add:
gnn = subparsers.add_parser("gnn")
gnn.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
gnn.add_argument("--snapshots-dir", default="data/gdelt/snapshots/france_protest")
gnn.add_argument("--train-origin-start", default="2021-01-04")
gnn.add_argument("--train-origin-end", default="2024-12-30")
gnn.add_argument("--eval-origin-start", default="2025-01-06")
gnn.add_argument("--eval-origin-end", default="2025-12-29")
gnn.add_argument("--out", default="data/gdelt/baselines/france_protest/gnn_predictions.jsonl")
gnn.add_argument("--epochs", type=int, default=30)
gnn.add_argument("--hidden-dim", type=int, default=64)

# In main, add:
if args.command == "gnn":
    try:
        run_gnn_backtest(
            tape_path=Path(args.tape),
            snapshots_dir=Path(args.snapshots_dir),
            train_origin_start=dt.date.fromisoformat(args.train_origin_start),
            train_origin_end=dt.date.fromisoformat(args.train_origin_end),
            eval_origin_start=dt.date.fromisoformat(args.eval_origin_start),
            eval_origin_end=dt.date.fromisoformat(args.eval_origin_end),
            out_path=Path(args.out),
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            progress=True,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
```

- [ ] **Step 4: Run all tests to verify**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add baselines/backtest.py baselines/gnn.py tests/test_gnn.py
git commit -m "feat: add GNN backtest subcommand"
```

---

## Run Commands (after all tasks pass tests)

Run the tabular baseline on the full France protest dataset:

```bash
time caffeinate -dimsu python -m baselines.backtest tabular \
  --tape data/gdelt/tape/france_protest/events.jsonl \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --out data/gdelt/baselines/france_protest/tabular_predictions.jsonl \
  2>&1 | tee .context/logs/tabular_backtest.log
```

Run the GNN baseline:

```bash
time caffeinate -dimsu python -m baselines.backtest gnn \
  --tape data/gdelt/tape/france_protest/events.jsonl \
  --snapshots-dir data/gdelt/snapshots/france_protest \
  --train-origin-start 2021-01-04 \
  --train-origin-end 2024-12-30 \
  --eval-origin-start 2025-01-06 \
  --eval-origin-end 2025-12-29 \
  --out data/gdelt/baselines/france_protest/gnn_predictions.jsonl \
  --epochs 30 \
  --hidden-dim 64 \
  2>&1 | tee .context/logs/gnn_backtest.log
```

Compare audits:

```bash
python -m json.tool data/gdelt/baselines/france_protest/tabular_predictions.audit.json
python -m json.tool data/gdelt/baselines/france_protest/gnn_predictions.audit.json
```

---

## Completed Gate

The GNN result justifies moving to graph expansion because it beats recurrence
and XGBoost on the main holdout calibration/error metrics:

- Brier score: `gnn_brier < tabular_brier`
- MAE: `gnn_mae < tabular_mae`
- top-5 hit rate: `gnn_top5_hit_rate > tabular_top5_hit_rate`

The GNN does not yet dominate every ranking metric; recall@5 remains an
explicit tuning and ablation target. The next step is to extend with additional
node types and sources, starting with ACLED and point-in-time Wikidata grounding,
while measuring each addition against the frozen France protest benchmark.
