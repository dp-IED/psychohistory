"""Run WM linear / GRU / GNN / GRU+GNN ablations with a shared JSON reporting surface (warehouse-backed)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
    batch_size: int | None = None,
    early_stop_patience: int | None = None,
    device: torch.device | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    ep = int(epochs) if epochs is not None else int(LINEAR_SKELETON_DEFAULTS["epochs"])
    lr_v = float(lr) if lr is not None else float(LINEAR_SKELETON_DEFAULTS["lr"])
    bs = int(batch_size) if batch_size is not None else int(LINEAR_SKELETON_DEFAULTS["batch_size"])
    pat = (
        int(early_stop_patience)
        if early_stop_patience is not None
        else int(LINEAR_SKELETON_DEFAULTS["early_stop_patience"])
    )
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
            f"[wm_ablation_run] variants={variants} epochs={ep} lr={lr_v} batch_size={bs} "
            f"early_stop_patience={pat} history_weeks={history_weeks} device={dev}",
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
                lr=lr_v,
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
                lr=lr_v,
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
                lr=lr_v,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                early_stop_patience=pat,
                use_loc_temporal=False,
                history_weeks=history_weeks,
                progress=progress,
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
                lr=lr_v,
                batch_size=bs,
                device=dev,
                source_names=None,
                feature_names=list(FEATURE_NAMES),
                excluded_admin1=set(EXCLUDED_REGIONAL_ADMIN1_CODES),
                early_stop_patience=pat,
                use_loc_temporal=True,
                history_weeks=history_weeks,
                progress=progress,
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
    p = argparse.ArgumentParser(description="WM ablation driver (linear, GRU, GNN, GRU+GNN).")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--warehouse-path", type=Path, default=None)
    p.add_argument("--variant", default="all", help="all|linear|gru|gnn|gru_gnn (comma-separated)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history-weeks", type=int, default=int(LINEAR_SKELETON_DEFAULTS["history_weeks"]))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
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
        allowed = {"linear", "gru", "gnn", "gru_gnn"}
        bad = [x for x in names if x not in allowed]
        if bad:
            raise SystemExit(f"unknown variant(s): {bad}; allowed {sorted(allowed)} or all")

    rows = run_wm_ablation_cli(
        warehouse_path=args.warehouse_path,
        data_root=args.data_root,
        train_origin_start=dt.date.fromisoformat(args.train_origin_start),
        train_origin_end=dt.date.fromisoformat(args.train_origin_end),
        holdout_origin_start=dt.date.fromisoformat(args.holdout_origin_start),
        holdout_origin_end=dt.date.fromisoformat(args.holdout_origin_end),
        variants=names,
        seed=int(args.seed),
        history_weeks=int(args.history_weeks),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        device=torch.device(args.device),
        progress=not args.no_progress,
    )
    if not args.no_progress:
        print("[wm_ablation_run] finished all variants; writing JSON summary to stdout.", file=sys.stderr, flush=True)
    print(json.dumps(_json_safe(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
