"""Run baseline backtests over normalized event tapes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from baselines.features import extract_features_for_origin
from baselines.metrics import brier_score, mean_absolute_error, recall_at_k, top_k_hit_rate
from baselines.recurrence import (
    RECURRENCE_MODEL_NAMES,
    ForecastRow,
    build_recurrence_forecasts_for_origin,
)
from baselines.tabular import TabularForecastRow, predict_tabular, train_tabular_model
from ingest.event_tape import load_event_tape


def weekly_origins(start: dt.date, end: dt.date) -> list[dt.date]:
    if start.weekday() != 0:
        raise ValueError(f"origin_start must be a Monday: {start.isoformat()}")
    if end.weekday() != 0:
        raise ValueError(f"origin_end must be a Monday: {end.isoformat()}")
    if start > end:
        raise ValueError(
            f"origin_start must be on or before origin_end: {start.isoformat()} > {end.isoformat()}"
        )
    origins: list[dt.date] = []
    current = start
    while current <= end:
        origins.append(current)
        current += dt.timedelta(days=7)
    return origins


def build_recurrence_backtest_rows(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
) -> list[ForecastRow]:
    records = load_event_tape(tape_path)
    rows: list[ForecastRow] = []
    for forecast_origin in weekly_origins(origin_start, origin_end):
        rows.extend(
            build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=forecast_origin,
            )
        )
    return rows


def build_audit(
    rows: list[ForecastRow],
    *,
    origin_start: dt.date,
    origin_end: dt.date,
) -> dict[str, Any]:
    model_names = [
        model_name
        for model_name in RECURRENCE_MODEL_NAMES
        if any(row.model_name == model_name for row in rows)
    ]
    return {
        "row_count": len(rows),
        "origin_start": origin_start.isoformat(),
        "origin_end": origin_end.isoformat(),
        "model_names": model_names,
        "admin1_count": len({row.admin1_code for row in rows}),
        "mean_brier_by_model": {
            model_name: brier_score(rows, model_name)
            for model_name in model_names
        },
        "mean_absolute_error_by_model": {
            model_name: mean_absolute_error(rows, model_name)
            for model_name in model_names
        },
        "top5_hit_rate_by_model": {
            model_name: top_k_hit_rate(rows, model_name, k=5)
            for model_name in model_names
        },
        "recall_at_5_by_model": {
            model_name: recall_at_k(rows, model_name, k=5)
            for model_name in model_names
        },
    }


def run_recurrence_backtest(
    *,
    tape_path: Path,
    origin_start: dt.date,
    origin_end: dt.date,
    out_path: Path,
    progress: bool = False,
) -> dict[str, Any]:
    records = load_event_tape(tape_path)
    origins = weekly_origins(origin_start, origin_end)
    rows: list[ForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, forecast_origin in enumerate(origins, start=1):
            origin_rows = build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=forecast_origin,
            )
            rows.extend(origin_rows)
            for row in origin_rows:
                handle.write(row.model_dump_json() + "\n")
            handle.flush()
            if progress:
                print(
                    (
                        f"[backtest] {index}/{len(origins)} "
                        f"origin={forecast_origin.isoformat()} rows={len(rows)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    audit = build_audit(rows, origin_start=origin_start, origin_end=origin_end)
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


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
    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )

    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

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
            print(
                f"[tabular] features {idx}/{total} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )

    train_feature_rows = [r for o in train_origins for r in feature_rows_by_origin[o]]
    train_origin_set = set(train_origins)
    train_targets = {k: v[1] for k, v in target_lookup.items() if k[0] in train_origin_set}

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
    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )
    from baselines.gnn import GNNForecastRow, build_graph_from_snapshot, predict_gnn, train_gnn
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
        return json.loads(path.read_text(encoding="utf-8"))

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
            print(
                f"[gnn] load train {idx}/{len(train_origins)} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )

    if progress:
        print(
            f"[gnn] training GNN epochs={epochs} hidden_dim={hidden_dim}...",
            file=sys.stderr,
            flush=True,
        )

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
                print(
                    f"[gnn] eval {idx}/{len(eval_origins)} origin={origin.isoformat()}",
                    file=sys.stderr,
                    flush=True,
                )
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recurrence = subparsers.add_parser("recurrence")
    recurrence.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    recurrence.add_argument("--origin-start", default="2021-01-04")
    recurrence.add_argument("--origin-end", default="2025-12-29")
    recurrence.add_argument(
        "--out",
        default="data/gdelt/baselines/france_protest/recurrence_predictions.jsonl",
    )
    tabular = subparsers.add_parser("tabular")
    tabular.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    tabular.add_argument("--train-origin-start", default="2021-01-04")
    tabular.add_argument("--train-origin-end", default="2024-12-30")
    tabular.add_argument("--eval-origin-start", default="2025-01-06")
    tabular.add_argument("--eval-origin-end", default="2025-12-29")
    tabular.add_argument(
        "--out",
        default="data/gdelt/baselines/france_protest/tabular_predictions.jsonl",
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "recurrence":
        try:
            run_recurrence_backtest(
                tape_path=Path(args.tape),
                origin_start=dt.date.fromisoformat(args.origin_start),
                origin_end=dt.date.fromisoformat(args.origin_end),
                out_path=Path(args.out),
                progress=True,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
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
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
