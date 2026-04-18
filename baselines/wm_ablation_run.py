"""Run WM linear / GRU / GNN / GRU+GNN ablations with a shared JSON reporting surface (warehouse-backed)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from baselines.train_loop_skeleton import (
    LINEAR_SKELETON_DEFAULTS,
    run_linear_skeleton_cli,
)
from baselines.training_slice import (
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END,
    FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_END,
    FRANCE_SCAFFOLD_TRAIN_ORIGIN_START,
)
from baselines.wm_ablation_train import run_wm_gnn_training
from baselines.features import FEATURE_NAMES
from ingest.paths import resolve_data_root, warehouse_path
from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES


def _resolved_warehouse_db(warehouse_path: Path | None, data_root: Path | None) -> Path:
    if warehouse_path is not None:
        return Path(warehouse_path).expanduser().resolve()
    return warehouse_path(resolve_data_root(data_root))


def _fmt_metric(x: Any) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def wm_ablation_aggregate(
    runs: list[list[dict[str, Any]]],
    *,
    seeds: list[int],
) -> dict[str, Any]:
    """Mean/std of ``best_holdout_brier`` and collapse counts per variant across seeds."""

    by_variant: dict[str, dict[str, Any]] = {}
    for rows in runs:
        for row in rows:
            v = str(row["variant"])
            if v not in by_variant:
                by_variant[v] = {"briers": [], "collapses": 0}
            br = row.get("best_holdout_brier")
            if isinstance(br, (int, float)) and not (
                isinstance(br, float) and (math.isnan(br) or math.isinf(br))
            ):
                by_variant[v]["briers"].append(float(br))
            if row.get("collapse_detected"):
                by_variant[v]["collapses"] += 1
    out: dict[str, Any] = {"event": "wm_ablation_aggregate", "seeds": seeds, "by_variant": {}}
    for v, d in sorted(by_variant.items()):
        xs = d["briers"]
        if len(xs) >= 2:
            mean = statistics.fmean(xs)
            std = statistics.pstdev(xs)
        elif len(xs) == 1:
            mean = xs[0]
            std = 0.0
        else:
            mean = float("nan")
            std = float("nan")
        out["by_variant"][v] = {
            "best_holdout_brier_mean": mean,
            "best_holdout_brier_std": std,
            "collapse_count": d["collapses"],
            "n_seeds": len(seeds),
        }
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def run_wm_ablation_cli(
    *,
    warehouse_path: Path | None = None,
    data_root: Path | None = None,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    holdout_origin_start: dt.date,
    holdout_origin_end: dt.date,
    variants: list[str],
    seed: int,
    history_weeks: int,
    epochs: int | None = None,
    lr: float | None = None,
    lr_linear: float | None = None,
    lr_gnn: float | None = None,
    batch_size: int | None = None,
    early_stop_patience: int | None = None,
    early_stop_patience_gnn: int | None = None,
    log_grad_norm_gnn: bool = False,
    device: torch.device | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    ep = int(epochs) if epochs is not None else int(LINEAR_SKELETON_DEFAULTS["epochs"])
    lr_default = float(lr) if lr is not None else float(LINEAR_SKELETON_DEFAULTS["lr"])
    lr_lin = float(lr_linear) if lr_linear is not None else lr_default
    lr_g = float(lr_gnn) if lr_gnn is not None else lr_default
    bs = int(batch_size) if batch_size is not None else int(LINEAR_SKELETON_DEFAULTS["batch_size"])
    pat = (
        int(early_stop_patience)
        if early_stop_patience is not None
        else int(LINEAR_SKELETON_DEFAULTS["early_stop_patience"])
    )
    pat_gnn = int(early_stop_patience_gnn) if early_stop_patience_gnn is not None else pat
    dev = device or torch.device("cpu")

    if progress:
        db_resolved = _resolved_warehouse_db(warehouse_path, data_root)
        print(
            "[wm_ablation_run] WM ablation (warehouse-backed)",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[wm_ablation_run] warehouse_db={db_resolved}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[wm_ablation_run] train_origins {train_origin_start.isoformat()}..{train_origin_end.isoformat()} | "
            f"holdout {holdout_origin_start.isoformat()}..{holdout_origin_end.isoformat()}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[wm_ablation_run] variants={variants} epochs={ep} lr_linear={lr_lin} lr_gnn={lr_g} "
            f"batch_size={bs} early_stop_patience_linear={pat} early_stop_patience_gnn={pat_gnn} "
            f"history_weeks={history_weeks} device={dev}",
            file=sys.stderr,
            flush=True,
        )

    out: list[dict[str, Any]] = []
    pbar = tqdm(
        variants,
        desc="wm_ablation",
        unit="variant",
        file=sys.stderr,
        disable=not progress,
    )
    for name in pbar:
        torch.manual_seed(seed)
        if progress:
            pbar.set_description_str(f"wm_ablation · {name}")
            pbar.set_postfix_str(f"seed={seed} epochs={ep} hw={history_weeks} {dev}")
        if progress and name in ("gnn", "gru_gnn"):
            tqdm.write(
                f"[wm_ablation_run] variant={name}: building weekly graphs for every train/holdout origin "
                "(CPU-heavy; progress lines follow from wm_ablation_train).",
                file=sys.stderr,
            )
        if name == "linear":
            r = run_linear_skeleton_cli(
                warehouse_path=warehouse_path,
                data_root=data_root,
                train_origin_start=train_origin_start,
                train_origin_end=train_origin_end,
                holdout_origin_start=holdout_origin_start,
                holdout_origin_end=holdout_origin_end,
                epochs=ep,
                lr=lr_lin,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                progress=progress,
                early_stop_patience=pat,
                variant="linear",
                history_weeks=history_weeks,
                emit_stdout_json=False,
            )
        elif name == "gru":
            r = run_linear_skeleton_cli(
                warehouse_path=warehouse_path,
                data_root=data_root,
                train_origin_start=train_origin_start,
                train_origin_end=train_origin_end,
                holdout_origin_start=holdout_origin_start,
                holdout_origin_end=holdout_origin_end,
                epochs=ep,
                lr=lr_lin,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                progress=progress,
                early_stop_patience=pat,
                variant="gru",
                history_weeks=history_weeks,
                emit_stdout_json=False,
            )
        elif name == "gnn":
            r = run_wm_gnn_training(
                warehouse_path=warehouse_path,
                data_root=data_root,
                train_origin_start=train_origin_start,
                train_origin_end=train_origin_end,
                holdout_origin_start=holdout_origin_start,
                holdout_origin_end=holdout_origin_end,
                epochs=ep,
                lr=lr_g,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                early_stop_patience=pat_gnn,
                use_loc_temporal=False,
                history_weeks=history_weeks,
                progress=progress,
                log_grad_norm=log_grad_norm_gnn,
            )
        elif name == "gru_gnn":
            r = run_wm_gnn_training(
                warehouse_path=warehouse_path,
                data_root=data_root,
                train_origin_start=train_origin_start,
                train_origin_end=train_origin_end,
                holdout_origin_start=holdout_origin_start,
                holdout_origin_end=holdout_origin_end,
                epochs=ep,
                lr=lr_g,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                early_stop_patience=pat_gnn,
                use_loc_temporal=True,
                history_weeks=history_weeks,
                progress=progress,
                log_grad_norm=log_grad_norm_gnn,
            )
        elif name == "gru_multi":
            r = run_linear_skeleton_cli(
                warehouse_path=warehouse_path,
                data_root=data_root,
                train_origin_start=train_origin_start,
                train_origin_end=train_origin_end,
                holdout_origin_start=holdout_origin_start,
                holdout_origin_end=holdout_origin_end,
                epochs=ep,
                lr=lr_lin,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                progress=progress,
                early_stop_patience=pat,
                variant="gru_multi",
                history_weeks=history_weeks,
                emit_stdout_json=False,
            )
        else:
            raise ValueError(f"unknown variant {name!r}")
        row = dict(r)
        row["variant"] = name
        required = {
            "holdout_masked_row_count",
            "train_masked_prevalence",
            "holdout_prevalence",
            "brier_always_positive",
            "brier_predict_train_prevalence",
        }
        missing = required - set(row.get("baselines", {}))
        if missing:
            raise RuntimeError(f"variant {name}: baselines missing keys {sorted(missing)}")
        if "holdout_metrics" not in row or "holdout_mask_identity" not in row:
            raise RuntimeError(f"variant {name}: missing holdout_metrics or holdout_mask_identity")
        out.append(row)
        if progress:
            tqdm.write(
                f"[wm_ablation_run] variant={name} finished best_holdout_brier="
                f"{_fmt_metric(row.get('best_holdout_brier'))} best_epoch={row.get('best_epoch')} "
                f"early_stopped={row.get('early_stopped')} epochs_run={row.get('epochs')}",
                file=sys.stderr,
            )
            pbar.set_postfix(
                last_brier=_fmt_metric(row.get("best_holdout_brier")),
                last_epoch=str(row.get("best_epoch")),
            )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "WM ablation driver: linear, GRU, GNN, GRU+GNN, optional gru_multi (multi-horizon GRU). "
            "Evaluation parity: same split/mask/metrics; optimization may differ via --lr-linear vs --lr-gnn."
        )
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--warehouse-path", type=Path, default=None)
    p.add_argument(
        "--variant",
        default="all",
        help="all|linear|gru|gnn|gru_gnn|gru_multi (comma-separated). 'all' is the four base variants.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated ints (e.g. 0,1,2). Emits one JSON line per seed, then an aggregate line.",
    )
    p.add_argument("--history-weeks", type=int, default=int(LINEAR_SKELETON_DEFAULTS["history_weeks"]))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-linear", type=float, default=None, help="LR for linear / GRU / gru_multi (fallback: --lr).")
    p.add_argument("--lr-gnn", type=float, default=None, help="LR for gnn / gru_gnn (fallback: --lr).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument(
        "--early-stop-patience-gnn",
        type=int,
        default=None,
        help="Patience for GNN variants (fallback: --early-stop-patience).",
    )
    p.add_argument(
        "--log-grad-norm-gnn",
        action="store_true",
        help="Log mean gradient norm per epoch for GNN training.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence stderr progress (variant banners, per-epoch lines, data-load hints).",
    )
    p.add_argument("--train-origin-start", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_START.isoformat())
    p.add_argument("--train-origin-end", default=FRANCE_SCAFFOLD_TRAIN_ORIGIN_END.isoformat())
    p.add_argument("--holdout-origin-start", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_START.isoformat())
    p.add_argument("--holdout-origin-end", default=FRANCE_SCAFFOLD_HOLDOUT_ORIGIN_END.isoformat())
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass

    raw = args.variant.strip().lower()
    if raw == "all":
        names = ["linear", "gru", "gnn", "gru_gnn"]
    else:
        names = [x.strip() for x in raw.split(",") if x.strip()]
        allowed = {"linear", "gru", "gnn", "gru_gnn", "gru_multi"}
        bad = [x for x in names if x not in allowed]
        if bad:
            raise SystemExit(f"unknown variant(s): {bad}; allowed {sorted(allowed)} or all")

    explicit_multi_seed = bool(args.seeds and str(args.seeds).strip())
    if explicit_multi_seed:
        seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    else:
        seeds = [int(args.seed)]

    runs: list[list[dict[str, Any]]] = []
    for seed in seeds:
        rows = run_wm_ablation_cli(
            warehouse_path=args.warehouse_path,
            data_root=args.data_root,
            train_origin_start=dt.date.fromisoformat(args.train_origin_start),
            train_origin_end=dt.date.fromisoformat(args.train_origin_end),
            holdout_origin_start=dt.date.fromisoformat(args.holdout_origin_start),
            holdout_origin_end=dt.date.fromisoformat(args.holdout_origin_end),
            variants=names,
            seed=int(seed),
            history_weeks=int(args.history_weeks),
            epochs=args.epochs,
            lr=args.lr,
            lr_linear=args.lr_linear,
            lr_gnn=args.lr_gnn,
            batch_size=args.batch_size,
            early_stop_patience=args.early_stop_patience,
            early_stop_patience_gnn=args.early_stop_patience_gnn,
            log_grad_norm_gnn=bool(args.log_grad_norm_gnn),
            device=torch.device(args.device),
            progress=not args.no_progress,
        )
        runs.append(rows)

    if explicit_multi_seed or len(seeds) > 1:
        for seed, rows in zip(seeds, runs, strict=True):
            seed_line = {"event": "wm_ablation_seed", "seed": int(seed), "variants": rows}
            print(json.dumps(_json_safe(seed_line)), flush=True)
        if not args.no_progress:
            print(
                "[wm_ablation_run] writing aggregate JSON line to stdout.",
                file=sys.stderr,
                flush=True,
            )
        agg = wm_ablation_aggregate(runs, seeds=seeds)
        print(json.dumps(_json_safe(agg)), flush=True)
    else:
        if not args.no_progress:
            print("[wm_ablation_run] finished all variants; writing JSON summary to stdout.", file=sys.stderr, flush=True)
        print(json.dumps(_json_safe(runs[0]), indent=2), flush=True)


if __name__ == "__main__":
    main()
