"""XGBoost tabular baseline for regional France protest forecasting."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import numpy as np
import xgboost as xgb
from pydantic import BaseModel, Field

from baselines.features import FeatureRow, build_feature_matrix


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
) -> Callable[..., np.ndarray]:
    X, _admin1_codes = build_feature_matrix(feature_rows)
    y = np.array(
        [float(targets.get((row.forecast_origin, row.admin1_code), False)) for row in feature_rows],
        dtype=np.float32,
    )
    unique_classes = np.unique(y)
    if len(unique_classes) == 1:
        constant_probability = float(unique_classes[0])

        def predict_constant(X: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    np.full(X.shape[0], 1.0 - constant_probability, dtype=np.float32),
                    np.full(X.shape[0], constant_probability, dtype=np.float32),
                ]
            )

        return predict_constant

    scale_pos_weight = float(np.sum(y == 0)) / max(1.0, float(np.sum(y == 1)))
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X, y)
    return model.predict_proba


def predict_tabular(
    *,
    model: Callable[..., np.ndarray],
    feature_rows: list[FeatureRow],
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] | None = None,
) -> list[TabularForecastRow]:
    X, _admin1_codes = build_feature_matrix(feature_rows)
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
