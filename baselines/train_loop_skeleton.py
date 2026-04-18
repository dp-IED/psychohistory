"""
Minimal linear training loop on warehouse-derived features (next_steps.md step D).

Validates optimizer, loss, device, and batching before WM / GNN complexity. Not the production model.

Holdout **Brier** (``brier_score``) is computed only on rows where ``target_occurs_next_7d`` is present
(masked ``Location`` rows), matching :func:`baselines.gnn.build_graph_from_snapshot`—not over the full
scoring universe. Compare tabular/GNN Brier on the same basis when interpreting numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baselines.backtest import weekly_origins
from baselines.features import FEATURE_NAMES, extract_features_for_origin
from baselines.location_weekly_history import (
    LocationWeeklyHistorySamples,
    collect_location_weekly_history,
)
from baselines.metrics import brier_score
from baselines.recurrence import ForecastRow
from baselines.training_slice import (
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END,
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_END,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_START,
)
from ingest.event_records import load_event_records
from ingest.event_tape import EventTapeRecord
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

LINEAR_SKELETON_MODEL_NAME = "linear_skeleton"
GRU_SKELETON_MODEL_NAME = "gru_skeleton"

LINEAR_SKELETON_DEFAULTS: dict[str, Any] = {
    "lr": 1e-2,
    "batch_size": 64,
    "early_stop_patience": 1,
    "epochs": 5,
    "history_weeks": 8,
}

_skeleton_torch_configured = False


def _ensure_skeleton_torch_runtime_configured() -> None:
    """Match :func:`baselines.gnn._ensure_gnn_runtime_configured` — tame BLAS/OpenMP on macOS."""
    global _skeleton_torch_configured
    if _skeleton_torch_configured:
        return
    if os.environ.get("PSYCHOHISTORY_TORCH_CONFIGURE", "1") == "0":
        _skeleton_torch_configured = True
        return
    if sys.platform != "darwin":
        _skeleton_torch_configured = True
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
    _skeleton_torch_configured = True


def _holdout_masked_targets(
    holdout_samples: LocationSamples,
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
) -> list[float]:
    ys: list[float] = []
    for i in range(len(holdout_samples)):
        if not holdout_samples.mask[i].item():
            continue
        origin = holdout_samples.origins[i]
        code = holdout_samples.admin1_codes[i]
        _, to = target_lookup.get((origin, code), (0, False))
        ys.append(1.0 if to else 0.0)
    return ys


def brier_for_constant_predictor(ys: list[float], p: float) -> float:
    """Brier score for predicting constant ``p`` on binary labels ``ys`` in ``{0.0, 1.0}``."""
    if not ys:
        return float("nan")
    return sum((p - y) * (p - y) for y in ys) / len(ys)


def assert_mondays(*dates: dt.date, label: str = "date") -> None:
    for d in dates:
        if d.weekday() != 0:
            raise ValueError(f"{label} must be a Monday (weekly_origins invariant): {d.isoformat()}")


def _filter_records_by_sources(
    records: list[EventTapeRecord],
    source_names: set[str] | None,
) -> list[EventTapeRecord]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


def build_scoring_universe(
    records: list[EventTapeRecord],
    *,
    excluded_admin1: set[str],
) -> list[str]:
    return sorted(
        {
            r.admin1_code
            for r in records
            if r.admin1_code not in excluded_admin1
        }
    )


@dataclass(frozen=True)
class LocationSamples:
    """Stacked (origin, admin1) rows for one or more weekly origins."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    origins: tuple[dt.date, ...]
    admin1_codes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.x.shape[0])


def collect_samples_for_origins(
    *,
    records: list[EventTapeRecord],
    origins: list[dt.date],
    scoring_universe: list[str],
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
    progress: bool = False,
    progress_label: str = "collect_samples",
) -> LocationSamples:
    """
    Build supervised rows from ``extract_features_for_origin`` + ``build_snapshot_payload`` targets.

    ``mask[i]`` is True iff ``target_occurs_next_7d`` is present for that (origin, admin1), matching
    :func:`baselines.gnn.build_graph_from_snapshot` semantics.
    """
    for origin in origins:
        assert_mondays(origin, label="forecast_origin")

    xs: list[list[float]] = []
    ys: list[float] = []
    masks: list[bool] = []
    row_origins: list[dt.date] = []
    row_codes: list[str] = []

    origin_pbar = tqdm(
        origins,
        desc=f"train_loop_skeleton · {progress_label}",
        unit="origin",
        file=sys.stderr,
        disable=not progress,
    )
    for origin in origin_pbar:
        if progress:
            origin_pbar.set_postfix_str(origin.isoformat())
        feature_rows = extract_features_for_origin(
            records=records,
            origin_date=origin,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
        target_occurs: dict[str, bool] = {}
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in excluded_admin1:
                continue
            if row["name"] == "target_occurs_next_7d":
                target_occurs[code] = bool(row["value"])

        for fr in feature_rows:
            code = fr.admin1_code
            if code in excluded_admin1:
                continue
            missing = [name for name in feature_names if name not in fr.features]
            if missing:
                raise ValueError(
                    "feature_names contains keys not present on FeatureRow: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            xs.append([float(fr.features[name]) for name in feature_names])
            row_origins.append(origin)
            row_codes.append(code)
            if code in target_occurs:
                ys.append(float(target_occurs[code]))
                masks.append(True)
            else:
                ys.append(0.0)
                masks.append(False)

    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)
    m_t = torch.tensor(masks, dtype=torch.bool)
    return LocationSamples(
        x=x_t,
        y=y_t,
        mask=m_t,
        origins=tuple(row_origins),
        admin1_codes=tuple(row_codes),
    )


class LinearOccurrenceModel(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class OccurrenceGRUModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        b, _, _ = x_seq.shape
        device = x_seq.device
        xr = torch.flip(x_seq, dims=(1,))
        mr = torch.flip(time_mask, dims=(1,))
        lengths = mr.long().sum(dim=1)
        logits = torch.zeros(b, device=device, dtype=x_seq.dtype)
        ok = lengths > 0
        if bool(ok.any().item()):
            idx = torch.where(ok)[0]
            x_sub = xr[ok]
            lens = lengths[ok].detach().cpu()
            packed = pack_padded_sequence(x_sub, lens, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
            h_last = h_n[-1]
            logits[idx] = self.head(h_last).squeeze(-1)
        return logits


def _train_one_epoch(
    model: LinearOccurrenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb, mb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        if mb.sum() == 0:
            continue
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    if n_batches == 0:
        print(
            "[train_loop_skeleton] warning: no training batches with masked labels this epoch.",
            file=sys.stderr,
            flush=True,
        )
        return float("nan")
    return total_loss / n_batches


def _train_one_epoch_gru(
    model: OccurrenceGRUModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x_seq, tmask, yb, mb in loader:
        x_seq = x_seq.to(device)
        tmask = tmask.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        if mb.sum() == 0:
            continue
        optimizer.zero_grad()
        logits = model(x_seq, tmask)
        loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    if n_batches == 0:
        print(
            "[train_loop_skeleton] warning: no training batches with masked labels this epoch.",
            file=sys.stderr,
            flush=True,
        )
        return float("nan")
    return total_loss / n_batches


def _forecast_rows_from_samples(
    model: LinearOccurrenceModel,
    samples: LocationSamples,
    *,
    device: torch.device,
    model_name: str,
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
) -> list[ForecastRow]:
    model.eval()
    rows: list[ForecastRow] = []
    with torch.no_grad():
        logits = model(samples.x.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    for i in range(len(samples)):
        if not samples.mask[i].item():
            continue
        origin = samples.origins[i]
        code = samples.admin1_codes[i]
        tc, to = target_lookup.get((origin, code), (0, False))
        rows.append(
            ForecastRow(
                forecast_origin=origin,
                admin1_code=code,
                model_name=model_name,
                predicted_count=float(probs[i]),
                predicted_occurrence_probability=float(probs[i]),
                target_count_next_7d=int(tc),
                target_occurs_next_7d=bool(to),
            )
        )
    return rows


def _forecast_rows_from_gru_samples(
    model: OccurrenceGRUModel,
    hist: LocationWeeklyHistorySamples,
    *,
    device: torch.device,
    model_name: str,
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
) -> list[ForecastRow]:
    model.eval()
    rows: list[ForecastRow] = []
    with torch.no_grad():
        logits = model(hist.x_seq.to(device), hist.time_mask.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    for i in range(len(hist.origins)):
        if not hist.mask[i].item():
            continue
        origin = hist.origins[i]
        code = hist.admin1_codes[i]
        tc, to = target_lookup.get((origin, code), (0, False))
        rows.append(
            ForecastRow(
                forecast_origin=origin,
                admin1_code=code,
                model_name=model_name,
                predicted_count=float(probs[i]),
                predicted_occurrence_probability=float(probs[i]),
                target_count_next_7d=int(tc),
                target_occurs_next_7d=bool(to),
            )
        )
    return rows


def build_target_lookup_for_origins(
    *,
    records: list[EventTapeRecord],
    origins: list[dt.date],
    source_names: set[str] | None,
    excluded_admin1: set[str],
    progress: bool = False,
    progress_label: str = "target_lookup",
) -> dict[tuple[dt.date, str], tuple[int, bool]]:
    lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}
    origin_pbar = tqdm(
        origins,
        desc=f"train_loop_skeleton · {progress_label}",
        unit="origin",
        file=sys.stderr,
        disable=not progress,
    )
    for origin in origin_pbar:
        if progress:
            origin_pbar.set_postfix_str(origin.isoformat())
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
        for row in payload["target_table"]:
            code = row["metadata"]["admin1_code"]
            if code in excluded_admin1:
                continue
            key = (origin, code)
            existing = lookup.get(key, (0, False))
            if row["name"] == "target_count_next_7d":
                lookup[key] = (int(row["value"]), existing[1])
            elif row["name"] == "target_occurs_next_7d":
                lookup[key] = (existing[0], bool(row["value"]))
    return lookup


def run_linear_skeleton_cli(
    *,
    warehouse_path: Path | None = None,
    data_root: Path | None = None,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    holdout_origin_start: dt.date,
    holdout_origin_end: dt.date,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
    progress: bool = False,
    early_stop_patience: int = 1,
    variant: Literal["linear", "gru"] = "linear",
    history_weeks: int = 8,
    emit_stdout_json: bool = True,
) -> dict[str, Any]:
    _ensure_skeleton_torch_runtime_configured()
    if variant not in {"linear", "gru"}:
        raise ValueError(f"unknown variant: {variant!r}")
    if history_weeks < 1:
        raise ValueError(f"history_weeks must be >= 1, got {history_weeks}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    assert_mondays(
        train_origin_start,
        train_origin_end,
        holdout_origin_start,
        holdout_origin_end,
        label="window boundary",
    )
    if train_origin_end >= holdout_origin_start:
        raise ValueError(
            "train_origin_end must be before holdout_origin_start "
            f"({train_origin_end} >= {holdout_origin_start})"
        )

    if progress:
        if warehouse_path is not None:
            src = f"warehouse={warehouse_path}"
        else:
            src = "warehouse=<data-root>/warehouse/events.duckdb"
        tqdm.write(f"[train_loop_skeleton] loading events ({src})...", file=sys.stderr)

    with tqdm(
        total=1,
        desc="train_loop_skeleton · load_event_records",
        file=sys.stderr,
        disable=not progress,
        leave=False,
    ) as load_pbar:
        records = load_event_records(
            warehouse_db_path=warehouse_path,
            data_root=data_root,
        )
        load_pbar.update(1)
    filtered = _filter_records_by_sources(records, source_names)
    scoring_universe = build_scoring_universe(filtered, excluded_admin1=excluded_admin1)

    train_origins = weekly_origins(train_origin_start, train_origin_end)
    holdout_origins = weekly_origins(holdout_origin_start, holdout_origin_end)

    if progress:
        tqdm.write(
            f"[train_loop_skeleton] loaded records={len(filtered)} scoring_universe={len(scoring_universe)} "
            f"train_weeks={len(train_origins)} holdout_weeks={len(holdout_origins)} "
            f"features={len(feature_names)} device={device}",
            file=sys.stderr,
        )

    train_samples = collect_samples_for_origins(
        records=filtered,
        origins=train_origins,
        scoring_universe=scoring_universe,
        source_names=source_names,
        feature_names=feature_names,
        excluded_admin1=excluded_admin1,
        progress=progress,
        progress_label="train_samples",
    )
    holdout_samples = collect_samples_for_origins(
        records=filtered,
        origins=holdout_origins,
        scoring_universe=scoring_universe,
        source_names=source_names,
        feature_names=feature_names,
        excluded_admin1=excluded_admin1,
        progress=progress,
        progress_label="holdout_samples",
    )

    n_train = len(train_samples)
    if batch_size >= n_train and n_train > 0:
        print(
            f"[train_loop_skeleton] warning: batch_size ({batch_size}) >= train rows ({n_train}); "
            "batching is not exercised meaningfully.",
            file=sys.stderr,
            flush=True,
        )

    masked_y = train_samples.y[train_samples.mask]
    if masked_y.numel() > 0:
        prevalence = float(masked_y.mean().item())
        print(
            f"[train_loop_skeleton] train label prevalence (masked rows): "
            f"{prevalence:.4f} ({int(masked_y.sum().item())}/{int(train_samples.mask.sum().item())})",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[train_loop_skeleton] warning: zero masked training rows — loss may be empty.",
            file=sys.stderr,
            flush=True,
        )

    dim = len(feature_names)
    hist_train: LocationWeeklyHistorySamples | None = None
    hist_hold: LocationWeeklyHistorySamples | None = None
    if variant == "gru":
        if progress:
            tqdm.write(
                "[train_loop_skeleton] GRU: building past-only weekly tensors "
                f"(train_rows={len(train_samples)} holdout_rows={len(holdout_samples)} "
                f"history_weeks={history_weeks}; CPU-heavy — nested tqdm bars follow).",
                file=sys.stderr,
            )
        with tqdm(
            total=2,
            desc="train_loop_skeleton · GRU_loc_hist",
            unit="split",
            file=sys.stderr,
            disable=not progress,
        ) as gru_hist_pbar:
            hist_train = collect_location_weekly_history(
                records=filtered,
                origins=train_samples.origins,
                admin1_codes=train_samples.admin1_codes,
                y=train_samples.y,
                mask=train_samples.mask,
                scoring_universe=scoring_universe,
                source_names=source_names,
                feature_names=feature_names,
                excluded_admin1=excluded_admin1,
                history_weeks=history_weeks,
                progress=progress,
                progress_label="train_loc_hist",
            )
            if progress:
                gru_hist_pbar.set_postfix_str("train done")
            gru_hist_pbar.update(1)
            hist_hold = collect_location_weekly_history(
                records=filtered,
                origins=holdout_samples.origins,
                admin1_codes=holdout_samples.admin1_codes,
                y=holdout_samples.y,
                mask=holdout_samples.mask,
                scoring_universe=scoring_universe,
                source_names=source_names,
                feature_names=feature_names,
                excluded_admin1=excluded_admin1,
                history_weeks=history_weeks,
                progress=progress,
                progress_label="holdout_loc_hist",
            )
            if progress:
                gru_hist_pbar.set_postfix_str("holdout done")
            gru_hist_pbar.update(1)
        model: nn.Module = OccurrenceGRUModel(dim).to(device)
        ds = TensorDataset(
            hist_train.x_seq,
            hist_train.time_mask,
            hist_train.y,
            hist_train.mask,
        )
        model_name_for_brier = GRU_SKELETON_MODEL_NAME
    else:
        model = LinearOccurrenceModel(dim).to(device)
        ds = TensorDataset(train_samples.x, train_samples.y, train_samples.mask)
        model_name_for_brier = LINEAR_SKELETON_MODEL_NAME

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    all_origins = train_origins + holdout_origins
    target_lookup = build_target_lookup_for_origins(
        records=filtered,
        origins=all_origins,
        source_names=source_names,
        excluded_admin1=excluded_admin1,
        progress=progress,
        progress_label="target_lookup",
    )

    holdout_y_list = _holdout_masked_targets(holdout_samples, target_lookup)
    holdout_n_scored = len(holdout_y_list)
    train_masked_prevalence = (
        float(masked_y.mean().item()) if masked_y.numel() > 0 else float("nan")
    )
    holdout_prevalence = (
        sum(holdout_y_list) / len(holdout_y_list) if holdout_y_list else float("nan")
    )
    brier_always_positive = brier_for_constant_predictor(holdout_y_list, 1.0)
    brier_train_prevalence = brier_for_constant_predictor(
        holdout_y_list,
        train_masked_prevalence if not math.isnan(train_masked_prevalence) else 0.5,
    )
    baselines_block: dict[str, Any] = {
        "holdout_masked_row_count": holdout_n_scored,
        "train_masked_prevalence": train_masked_prevalence,
        "holdout_prevalence": holdout_prevalence,
        "brier_always_positive": brier_always_positive,
        "brier_predict_train_prevalence": brier_train_prevalence,
    }

    def _json_safe(value: Any) -> Any:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    if emit_stdout_json:
        print(
            json.dumps({"event": "holdout_baselines", **{k: _json_safe(v) for k, v in baselines_block.items()}}),
            flush=True,
        )
    def _fmt6(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "nan"
        return f"{x:.6f}"

    if progress or emit_stdout_json:
        print(
            (
                f"[train_loop_skeleton] holdout baselines (same {holdout_n_scored} masked holdout rows): "
                f"brier_always_positive={_fmt6(brier_always_positive)} "
                f"brier_predict_train_prev={_fmt6(brier_train_prevalence)} "
                f"train_masked_prev={_fmt6(train_masked_prevalence)} holdout_prev={_fmt6(holdout_prevalence)}"
            ),
            file=sys.stderr,
            flush=True,
        )

    last_train_loss = float("nan")
    last_holdout_brier = float("nan")
    best_holdout_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epochs_run = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        if progress:
            print(
                f"[train_loop_skeleton] epoch {epoch}/{epochs} train (batch_size={batch_size})...",
                file=sys.stderr,
                flush=True,
            )
        if variant == "linear":
            last_train_loss = _train_one_epoch(
                model,  # type: ignore[arg-type]
                loader,
                optimizer,
                device,
            )
            holdout_rows = _forecast_rows_from_samples(
                model,  # type: ignore[arg-type]
                holdout_samples,
                device=device,
                model_name=model_name_for_brier,
                target_lookup=target_lookup,
            )
        else:
            last_train_loss = _train_one_epoch_gru(
                model,  # type: ignore[arg-type]
                loader,
                optimizer,
                device,
            )
            assert hist_hold is not None
            holdout_rows = _forecast_rows_from_gru_samples(
                model,  # type: ignore[arg-type]
                hist_hold,
                device=device,
                model_name=model_name_for_brier,
                target_lookup=target_lookup,
            )
        holdout_n = len(holdout_rows)
        if holdout_rows:
            last_holdout_brier = brier_score(holdout_rows, model_name_for_brier)
            mean_pred_p = sum(row.predicted_occurrence_probability for row in holdout_rows) / len(
                holdout_rows
            )
            holdout_pos_rate = sum(
                1.0 if row.target_occurs_next_7d else 0.0 for row in holdout_rows
            ) / len(holdout_rows)
        else:
            last_holdout_brier = float("nan")
            mean_pred_p = float("nan")
            holdout_pos_rate = float("nan")

        epochs_run = epoch

        if not math.isnan(last_holdout_brier) and last_holdout_brier < best_holdout_brier - 1e-12:
            best_holdout_brier = last_holdout_brier
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        elif not math.isnan(last_holdout_brier):
            epochs_no_improve += 1

        def _json_float(x: float) -> float | None:
            return None if isinstance(x, float) and math.isnan(x) else x

        best_so_far = best_holdout_brier if best_holdout_brier < float("inf") else float("nan")
        metrics_line = json.dumps(
            {
                "epoch": epoch,
                "train_loss": _json_float(last_train_loss),
                "holdout_brier": _json_float(last_holdout_brier),
                "holdout_brier_row_count": holdout_n,
                "holdout_positive_rate": _json_float(holdout_pos_rate),
                "holdout_mean_predicted_p": _json_float(mean_pred_p),
                "best_holdout_brier_so_far": _json_float(best_so_far),
                "best_epoch_so_far": best_epoch if best_epoch > 0 else None,
            }
        )
        if emit_stdout_json:
            print(metrics_line, flush=True)
        if progress:
            tl = "nan" if isinstance(last_train_loss, float) and math.isnan(last_train_loss) else f"{last_train_loss:.6f}"
            hb = "nan" if isinstance(last_holdout_brier, float) and math.isnan(last_holdout_brier) else f"{last_holdout_brier:.6f}"
            hpr = (
                "nan"
                if isinstance(holdout_pos_rate, float) and math.isnan(holdout_pos_rate)
                else f"{holdout_pos_rate:.6f}"
            )
            mp = (
                "nan"
                if isinstance(mean_pred_p, float) and math.isnan(mean_pred_p)
                else f"{mean_pred_p:.6f}"
            )
            print(
                f"[train_loop_skeleton] epoch {epoch}/{epochs} done train_loss={tl} "
                f"holdout_brier={hb} holdout_rows={holdout_n} "
                f"holdout_prev={hpr} mean_pred_p={mp}",
                file=sys.stderr,
                flush=True,
            )

        if (
            early_stop_patience > 0
            and not math.isnan(last_holdout_brier)
            and epochs_no_improve >= early_stop_patience
        ):
            early_stopped = True
            if progress:
                print(
                    f"[train_loop_skeleton] early stop: no holdout Brier improvement for "
                    f"{early_stop_patience} epoch(s) after best epoch {best_epoch}",
                    file=sys.stderr,
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if progress:
            print(
                f"[train_loop_skeleton] restored best checkpoint epoch={best_epoch} "
                f"holdout_brier={best_holdout_brier:.6f}",
                file=sys.stderr,
                flush=True,
            )

    summary: dict[str, Any] = {
        "event": "run_summary",
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "best_epoch": best_epoch,
        "best_holdout_brier": _json_safe(best_holdout_brier if best_holdout_brier < float("inf") else float("nan")),
        "baselines": {k: _json_safe(v) for k, v in baselines_block.items()},
        "last_train_loss": _json_safe(last_train_loss),
        "last_holdout_brier": _json_safe(last_holdout_brier),
        "train_rows": n_train,
        "holdout_rows": len(holdout_samples),
    }
    if emit_stdout_json:
        print(json.dumps(summary), flush=True)

    return {
        "variant": variant,
        "epochs": epochs_run,
        "epochs_requested": epochs,
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "best_epoch": best_epoch,
        "best_holdout_brier": best_holdout_brier if best_holdout_brier < float("inf") else float("nan"),
        "last_train_loss": last_train_loss,
        "last_holdout_brier": last_holdout_brier,
        "train_rows": n_train,
        "holdout_rows": len(holdout_samples),
        "baselines": baselines_block,
        "history_weeks": history_weeks,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Linear occurrence baseline training skeleton.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for data/ layout; default warehouse is <data-root>/warehouse/events.duckdb.",
    )
    parser.add_argument(
        "--warehouse-path",
        type=Path,
        default=None,
        help="Path to events.duckdb (overrides the default derived from --data-root).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(LINEAR_SKELETON_DEFAULTS["epochs"]),
        help="Must be >= 1.",
    )
    parser.add_argument("--lr", type=float, default=float(LINEAR_SKELETON_DEFAULTS["lr"]))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(LINEAR_SKELETON_DEFAULTS["batch_size"]),
        help="Must be >= 1.",
    )
    parser.add_argument(
        "--variant",
        choices=["linear", "gru"],
        default="linear",
        help="linear: per-origin tabular features; gru: past-only weekly sequences.",
    )
    parser.add_argument(
        "--history-weeks",
        type=int,
        default=int(LINEAR_SKELETON_DEFAULTS["history_weeks"]),
        help="Past-only history depth for --variant gru.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence stderr progress (per-origin sample/lookup lines and epoch banners).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=int(LINEAR_SKELETON_DEFAULTS["early_stop_patience"]),
        help=(
            "Stop after this many consecutive epochs without beating best holdout Brier "
            "(default 1). Use 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Train for all epochs regardless of holdout Brier (sets patience to 0).",
    )
    parser.add_argument("--train-origin-start", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_START.isoformat())
    parser.add_argument("--train-origin-end", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_END.isoformat())
    parser.add_argument("--holdout-origin-start", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START.isoformat())
    parser.add_argument("--holdout-origin-end", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END.isoformat())
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass

    device = torch.device(args.device)
    early_stop_patience = 0 if args.no_early_stop else args.early_stop_patience
    run_linear_skeleton_cli(
        warehouse_path=args.warehouse_path,
        data_root=args.data_root,
        train_origin_start=dt.date.fromisoformat(args.train_origin_start),
        train_origin_end=dt.date.fromisoformat(args.train_origin_end),
        holdout_origin_start=dt.date.fromisoformat(args.holdout_origin_start),
        holdout_origin_end=dt.date.fromisoformat(args.holdout_origin_end),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        source_names=None,
        feature_names=list(FEATURE_NAMES),
        excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
        progress=not args.no_progress,
        early_stop_patience=early_stop_patience,
        variant=args.variant,
        history_weeks=args.history_weeks,
    )


if __name__ == "__main__":
    main()
