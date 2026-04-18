"""GNN / GRU+GNN training with skeleton-style masked BCE, holdout Brier, and early stopping.

**Evaluation parity** (fixed): same train/holdout dates, same masked location rows, same checkpoint
rule (best holdout Brier), same ``holdout_metrics`` / ``holdout_mask_identity`` contract as the
skeleton. **Optimization** (may vary): ``lr``, patience, etc.—tune without changing what is scored.
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm

from baselines.backtest import weekly_origins
from baselines.features import extract_features_for_origin
from baselines.gnn import (
    FULL_GRAPH_ABLATION,
    GNNForecastRow,
    HeteroGNNModel,
    build_graph_from_snapshot,
    predict_gnn,
)
from baselines.location_weekly_history import collect_location_weekly_history
from baselines.metrics import (
    brier_score,
    holdout_mask_identity,
    wm_holdout_metrics_dict,
)
from baselines.train_loop_skeleton import (
    _ensure_skeleton_torch_runtime_configured,
    _holdout_masked_targets,
    assert_mondays,
    build_scoring_universe,
    build_target_lookup_for_origins,
    brier_for_constant_predictor,
    collect_samples_for_origins,
)
from ingest.event_records import load_event_records
from ingest.event_tape import EventTapeRecord
from ingest.paths import resolve_data_root, warehouse_path as default_warehouse_path
from ingest.snapshot_export import build_snapshot_payload

GNN_WM_MODEL_NAME = "gnn_sage"

# Collapse detection (wm ablation interpretability): tune in one place.
_COLLAPSE_SATURATION_EPS = 0.02
_LOSS_SPIKE_RATIO = 10.0
_GRAD_NORM_CEILING = 1e6
# Sustained non-finite / huge gradients: more than this many consecutive bad batches.
_SUSTAINED_BAD_BATCHES = 2


def _filter_records_by_sources(
    records: list[EventTapeRecord],
    source_names: set[str] | None,
) -> list[EventTapeRecord]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


def build_loc_temporal_for_graph(
    graph: HeteroData,
    forecast_origin: dt.date,
    *,
    records: list[EventTapeRecord],
    scoring_universe: list[str],
    source_names: set[str] | None,
    feature_names: list[str],
    excluded_admin1: set[str],
    history_weeks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = list(graph["location"].admin1_codes)
    n = len(codes)
    hist = collect_location_weekly_history(
        records=records,
        origins=tuple(forecast_origin for _ in range(n)),
        admin1_codes=tuple(codes),
        y=graph["location"].y.detach().cpu(),
        mask=graph["location"].mask.detach().cpu(),
        scoring_universe=scoring_universe,
        source_names=source_names,
        feature_names=feature_names,
        excluded_admin1=excluded_admin1,
        history_weeks=history_weeks,
    )
    return hist.x_seq, hist.time_mask


def run_wm_gnn_training(
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
    early_stop_patience: int = 1,
    hidden_dim: int = 64,
    use_loc_temporal: bool = False,
    history_weeks: int = 8,
    progress: bool = False,
    log_grad_norm: bool = False,
) -> dict[str, Any]:
    _ensure_skeleton_torch_runtime_configured()
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if train_origin_end >= holdout_origin_start:
        raise ValueError(
            "train_origin_end must be before holdout_origin_start "
            f"({train_origin_end} >= {holdout_origin_start})"
        )
    assert_mondays(
        train_origin_start,
        train_origin_end,
        holdout_origin_start,
        holdout_origin_end,
        label="window boundary",
    )

    if progress:
        db_display = (
            Path(warehouse_path).expanduser().resolve()
            if warehouse_path is not None
            else default_warehouse_path(resolve_data_root(data_root))
        )
        tqdm.write(f"[wm_ablation_train] loading events warehouse_db={db_display}", file=sys.stderr)

    with tqdm(
        total=1,
        desc="wm_ablation_train · load_event_records",
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

    train_graphs: list[tuple[dt.date, HeteroData]] = []
    hold_graphs: list[tuple[dt.date, HeteroData]] = []

    if not train_origins:
        raise ValueError("train origins must be non-empty")
    if not holdout_origins:
        raise ValueError("holdout origins must be non-empty")

    if progress:
        tqdm.write(
            f"[wm_ablation_train] building weekly snapshots → graphs "
            f"({len(train_origins)} train + {len(holdout_origins)} holdout weeks)...",
            file=sys.stderr,
        )

    train_graph_pbar = tqdm(
        train_origins,
        desc="wm_ablation_train · train_graph",
        unit="wk",
        file=sys.stderr,
        disable=not progress,
    )
    for origin in train_graph_pbar:
        if progress:
            train_graph_pbar.set_postfix_str(origin.isoformat())
        snap = build_snapshot_payload(
            records=filtered,
            origin_date=origin,
            source_names=source_names,
        )
        feats = extract_features_for_origin(
            records=filtered,
            origin_date=origin,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        g = build_graph_from_snapshot(
            snapshot=snap,
            feature_rows=feats,
            ablation=FULL_GRAPH_ABLATION,
        )
        train_graphs.append((origin, g))

    hold_graph_pbar = tqdm(
        holdout_origins,
        desc="wm_ablation_train · holdout_graph",
        unit="wk",
        file=sys.stderr,
        disable=not progress,
    )
    for origin in hold_graph_pbar:
        if progress:
            hold_graph_pbar.set_postfix_str(origin.isoformat())
        snap = build_snapshot_payload(
            records=filtered,
            origin_date=origin,
            source_names=source_names,
        )
        feats = extract_features_for_origin(
            records=filtered,
            origin_date=origin,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        g = build_graph_from_snapshot(
            snapshot=snap,
            feature_rows=feats,
            ablation=FULL_GRAPH_ABLATION,
        )
        hold_graphs.append((origin, g))

    if progress:
        tqdm.write(
            f"[wm_ablation_train] loaded records={len(filtered)} scoring_universe={len(scoring_universe)} "
            f"train_weeks={len(train_graphs)} holdout_weeks={len(hold_graphs)} "
            f"use_loc_temporal={use_loc_temporal} history_weeks={history_weeks} device={device}",
            file=sys.stderr,
        )

    all_origins = train_origins + holdout_origins
    target_lookup = build_target_lookup_for_origins(
        records=filtered,
        origins=all_origins,
        source_names=source_names,
        excluded_admin1=excluded_admin1,
        progress=progress,
        progress_label="target_lookup",
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

    masked_y = train_samples.y[train_samples.mask]
    train_masked_prevalence = (
        float(masked_y.mean().item()) if masked_y.numel() > 0 else float("nan")
    )
    holdout_y_list = _holdout_masked_targets(holdout_samples, target_lookup)
    holdout_n_scored = len(holdout_y_list)
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

    def _fmt6(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "nan"
        return f"{x:.6f}"

    if progress:
        print(
            (
                f"[wm_ablation_train] holdout baselines (same {holdout_n_scored} masked holdout rows): "
                f"brier_always_positive={_fmt6(brier_always_positive)} "
                f"brier_predict_train_prev={_fmt6(brier_train_prevalence)} "
                f"train_masked_prev={_fmt6(train_masked_prevalence)} holdout_prev={_fmt6(holdout_prevalence)}"
            ),
            file=sys.stderr,
            flush=True,
        )
        if masked_y.numel() > 0:
            prevalence = float(masked_y.mean().item())
            print(
                f"[wm_ablation_train] train label prevalence (masked rows): "
                f"{prevalence:.4f} ({int(masked_y.sum().item())}/{int(train_samples.mask.sum().item())})",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "[wm_ablation_train] warning: zero masked training rows — loss may be empty.",
                file=sys.stderr,
                flush=True,
            )

    loc_f = int(train_graphs[0][1]["location"].x.shape[1])
    evt_f = int(train_graphs[0][1]["event"].x.shape[1])
    model = HeteroGNNModel(
        location_feature_dim=loc_f,
        event_feature_dim=int(evt_f),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _train_one_epoch() -> tuple[float, float | None, bool, str | None]:
        """Returns mean loss, mean grad norm (or None), grad_pathology flag, pathology reason."""

        model.train()
        total_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0
        n_grad = 0
        bad_streak = 0
        pathology = False
        pathology_reason: str | None = None
        order = list(range(len(train_graphs)))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            batch_has_mask = False
            for j in chunk:
                _, g0 = train_graphs[j]
                if int(g0["location"].mask.sum().item()) == 0:
                    continue
                batch_has_mask = True
                break
            if not batch_has_mask:
                continue
            optimizer.zero_grad()
            acc_loss: torch.Tensor | None = None
            n_sub = 0
            for j in chunk:
                origin, g0 = train_graphs[j]
                g = g0.to(device)
                loc_t = loc_m = None
                if use_loc_temporal:
                    xt, mt = build_loc_temporal_for_graph(
                        g0,
                        origin,
                        records=filtered,
                        scoring_universe=scoring_universe,
                        source_names=source_names,
                        feature_names=feature_names,
                        excluded_admin1=excluded_admin1,
                        history_weeks=history_weeks,
                    )
                    loc_t = xt.to(device)
                    loc_m = mt.to(device)
                mask = g["location"].mask
                if mask.sum() == 0:
                    continue
                logits = model(g, loc_t, loc_m)
                part = F.binary_cross_entropy_with_logits(
                    logits[mask], g["location"].y[mask]
                )
                acc_loss = part if acc_loss is None else acc_loss + part
                n_sub += 1
            if acc_loss is None or n_sub == 0:
                continue
            loss = acc_loss / float(n_sub)
            loss.backward()
            gn_t = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            gn = float(gn_t.detach().item()) if bool(torch.isfinite(gn_t).item()) else float("nan")
            if log_grad_norm and math.isfinite(gn):
                grad_norm_sum += gn
                n_grad += 1
            batch_bad = (not math.isfinite(gn)) or gn > _GRAD_NORM_CEILING
            if batch_bad:
                bad_streak += 1
                if bad_streak >= _SUSTAINED_BAD_BATCHES and not pathology:
                    pathology = True
                    pathology_reason = "sustained_bad_gradient_norm"
            else:
                bad_streak = 0
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        if n_batches == 0:
            return float("nan"), None, False, None
        mean_gn: float | None
        if log_grad_norm and n_grad > 0:
            mean_gn = grad_norm_sum / float(n_grad)
        else:
            mean_gn = None
        return total_loss / n_batches, mean_gn, pathology, pathology_reason

    def _holdout_rows() -> list[GNNForecastRow]:
        model.eval()
        rows: list[GNNForecastRow] = []
        with torch.no_grad():
            for origin, g0 in hold_graphs:
                g = g0.to(device)
                loc_t = loc_m = None
                if use_loc_temporal:
                    xt, mt = build_loc_temporal_for_graph(
                        g0,
                        origin,
                        records=filtered,
                        scoring_universe=scoring_universe,
                        source_names=source_names,
                        feature_names=feature_names,
                        excluded_admin1=excluded_admin1,
                        history_weeks=history_weeks,
                    )
                    loc_t = xt.to(device)
                    loc_m = mt.to(device)
                rows.extend(
                    predict_gnn(
                        model=model,
                        graph=g,
                        origin_date=origin,
                        target_lookup=target_lookup,
                        model_name=GNN_WM_MODEL_NAME,
                        ablation=FULL_GRAPH_ABLATION,
                        loc_temporal=loc_t,
                        loc_time_mask=loc_m,
                    )
                )
        return rows

    last_train_loss = float("nan")
    last_mean_grad_norm: float | None = None
    last_holdout_brier = float("nan")
    best_holdout_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epochs_run = 0
    early_stopped = False
    train_loss_ema: float | None = None
    _ema_alpha = 0.25
    collapse_events: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if progress:
            print(
                f"[wm_ablation_train] epoch {epoch}/{epochs} gnn temporal={use_loc_temporal}...",
                file=sys.stderr,
                flush=True,
            )
        last_train_loss, last_mean_grad_norm, grad_path, grad_path_reason = _train_one_epoch()
        holdout_rows = _holdout_rows()
        holdout_n = len(holdout_rows)
        if holdout_rows:
            last_holdout_brier = brier_score(holdout_rows, GNN_WM_MODEL_NAME)
            mean_pred_p = sum(r.predicted_occurrence_probability for r in holdout_rows) / holdout_n
            holdout_pos_rate = sum(1.0 if r.target_occurs_next_7d else 0.0 for r in holdout_rows) / holdout_n
        else:
            last_holdout_brier = float("nan")
            mean_pred_p = float("nan")
            holdout_pos_rate = float("nan")
        epochs_run = epoch
        if progress:
            tl = (
                "nan"
                if isinstance(last_train_loss, float) and math.isnan(last_train_loss)
                else f"{last_train_loss:.6f}"
            )
            hb = (
                "nan"
                if isinstance(last_holdout_brier, float) and math.isnan(last_holdout_brier)
                else f"{last_holdout_brier:.6f}"
            )
            hpr = (
                "nan"
                if isinstance(holdout_pos_rate, float) and math.isnan(holdout_pos_rate)
                else f"{holdout_pos_rate:.6f}"
            )
            mp = "nan" if isinstance(mean_pred_p, float) and math.isnan(mean_pred_p) else f"{mean_pred_p:.6f}"
            gn_s = (
                f"{last_mean_grad_norm:.6f}"
                if last_mean_grad_norm is not None and math.isfinite(last_mean_grad_norm)
                else "na"
            )
            print(
                f"[wm_ablation_train] epoch {epoch}/{epochs} done train_loss={tl} "
                f"holdout_brier={hb} holdout_rows={holdout_n} "
                f"holdout_prev={hpr} mean_pred_p={mp} mean_grad_norm={gn_s}",
                file=sys.stderr,
                flush=True,
            )
        reasons: list[str] = []
        if holdout_n > 0 and isinstance(mean_pred_p, float) and math.isfinite(mean_pred_p):
            if mean_pred_p < _COLLAPSE_SATURATION_EPS or mean_pred_p > 1.0 - _COLLAPSE_SATURATION_EPS:
                reasons.append("mean_holdout_prediction_saturated")
        if (
            train_loss_ema is not None
            and math.isfinite(last_train_loss)
            and last_train_loss > _LOSS_SPIKE_RATIO * max(train_loss_ema, 1e-8)
        ):
            reasons.append("train_loss_spike_vs_ema")
        if grad_path:
            reasons.append(grad_path_reason or "gradient_pathology")
        if reasons:
            collapse_events.append(
                {
                    "epoch": epoch,
                    "reasons": reasons,
                    "mean_pred_holdout": mean_pred_p,
                    "train_loss": last_train_loss,
                }
            )
            print(
                f"[wm_ablation_train] collapse_warning epoch={epoch} reasons={reasons}",
                file=sys.stderr,
                flush=True,
            )
        if math.isfinite(last_train_loss):
            train_loss_ema = (
                last_train_loss
                if train_loss_ema is None
                else _ema_alpha * last_train_loss + (1.0 - _ema_alpha) * train_loss_ema
            )
        if not math.isnan(last_holdout_brier) and last_holdout_brier < best_holdout_brier - 1e-12:
            best_holdout_brier = last_holdout_brier
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        elif not math.isnan(last_holdout_brier):
            epochs_no_improve += 1
        if (
            early_stop_patience > 0
            and not math.isnan(last_holdout_brier)
            and epochs_no_improve >= early_stop_patience
        ):
            early_stopped = True
            if progress:
                print(
                    f"[wm_ablation_train] early stop: no holdout Brier improvement for "
                    f"{early_stop_patience} epoch(s) after best epoch {best_epoch}",
                    file=sys.stderr,
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if progress:
            hb = (
                best_holdout_brier
                if best_holdout_brier < float("inf")
                else float("nan")
            )
            print(
                f"[wm_ablation_train] restored best checkpoint epoch={best_epoch} holdout_brier={_fmt6(hb)}",
                file=sys.stderr,
                flush=True,
            )

    final_holdout_rows = _holdout_rows()
    hm = wm_holdout_metrics_dict(final_holdout_rows, GNN_WM_MODEL_NAME)
    hmi = holdout_mask_identity(final_holdout_rows)

    return {
        "epochs": epochs_run,
        "epochs_requested": epochs,
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "best_epoch": best_epoch,
        "best_holdout_brier": best_holdout_brier if best_holdout_brier < float("inf") else float("nan"),
        "last_train_loss": last_train_loss,
        "last_holdout_brier": last_holdout_brier,
        "train_rows": len(train_samples),
        "holdout_rows": len(holdout_samples),
        "baselines": baselines_block,
        "use_loc_temporal": use_loc_temporal,
        "history_weeks": history_weeks,
        "holdout_metrics": hm,
        "holdout_mask_identity": hmi,
        "collapse_events": collapse_events,
        "collapse_detected": bool(collapse_events),
        "log_grad_norm": log_grad_norm,
    }
