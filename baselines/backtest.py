"""Run baseline backtests over normalized event tapes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from baselines.features import FeatureRow
from baselines.metrics import brier_score, mean_absolute_error, recall_at_k, top_k_hit_rate
from baselines.recurrence import (
    RECURRENCE_MODEL_NAMES,
    ForecastRow,
    build_recurrence_forecasts_for_origin,
)
from ingest.event_tape import load_event_tape
from ingest.io_utils import open_text_auto


@dataclass(frozen=True)
class OriginInputs:
    origin: dt.date
    snapshot: dict[str, Any]
    feature_rows: list[FeatureRow]


def _filter_records_by_sources(records: list[Any], source_names: set[str] | None) -> list[Any]:
    if source_names is None:
        return records
    return [record for record in records if record.source_name in source_names]


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
    source_names: set[str] | None = None,
) -> list[ForecastRow]:
    records = load_event_tape(tape_path)
    rows: list[ForecastRow] = []
    for forecast_origin in weekly_origins(origin_start, origin_end):
        rows.extend(
            build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=forecast_origin,
                source_names=source_names,
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
    source_names: set[str] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    records = load_event_tape(tape_path)
    origins = weekly_origins(origin_start, origin_end)
    rows: list[ForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for index, forecast_origin in enumerate(origins, start=1):
            origin_rows = build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=forecast_origin,
                source_names=source_names,
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
    source_names: set[str] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )

    from baselines.features import extract_features_for_origin
    from baselines.tabular import TabularForecastRow, predict_tabular, train_tabular_model
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

    records = load_event_tape(tape_path)
    filtered_records = _filter_records_by_sources(records, source_names)
    # Derive scoring universe from all tape records (not just train-visible ones). France's
    # administrative regions are stable, so including codes that first appear post-cutoff
    # is a deliberate simplification that avoids per-origin universe re-computation.
    scoring_universe = sorted(
        {
            r.admin1_code
            for r in filtered_records
            if r.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES
        }
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
            source_names=source_names,
        )
        feature_rows_by_origin[origin] = rows
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
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
    with open_text_auto(out_path, "w") as handle:
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
    source_names: set[str] | None = None,
    gnn_ablation: Any | None = None,
    epochs: int = 30,
    hidden_dim: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )
    if progress:
        print(
            "[gnn] importing PyTorch (first use this run)...",
            file=sys.stderr,
            flush=True,
        )
    from baselines.features import extract_features_for_origin
    from baselines.gnn import GNNForecastRow, build_graph_from_snapshot, predict_gnn, train_gnn
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

    records = load_event_tape(tape_path)
    filtered_records = _filter_records_by_sources(records, source_names)
    # Derive scoring universe from all tape records (not just train-visible ones). France's
    # administrative regions are stable, so including codes that first appear post-cutoff
    # is a deliberate simplification that avoids per-origin universe re-computation.
    scoring_universe = sorted(
        {
            r.admin1_code
            for r in filtered_records
            if r.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES
        }
    )
    train_origins = weekly_origins(train_origin_start, train_origin_end)
    eval_origins = weekly_origins(eval_origin_start, eval_origin_end)

    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}

    def _load_snapshot(origin: dt.date) -> dict[str, Any]:
        for path in (
            snapshots_dir / f"as_of_{origin.isoformat()}.json",
            snapshots_dir / f"as_of_{origin.isoformat()}.json.gz",
        ):
            if path.exists():
                with open_text_auto(path, "r") as handle:
                    return json.load(handle)
        raise FileNotFoundError(f"missing snapshot for origin {origin.isoformat()} under {snapshots_dir}")

    def _build_target_lookup_for_origin(origin: dt.date) -> None:
        payload = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_names=source_names,
        )
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
        if progress:
            print(
                f"[gnn] load train {idx}/{len(train_origins)} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )
        snap = _load_snapshot(origin)
        feature_rows = extract_features_for_origin(
            records=records,
            origin_date=origin,
            scoring_universe=scoring_universe,
            source_names=source_names,
        )
        graph = build_graph_from_snapshot(
            snapshot=snap,
            feature_rows=feature_rows,
            ablation=gnn_ablation,
        )
        train_graphs.append(graph)
        _build_target_lookup_for_origin(origin)

    if progress:
        print(
            f"[gnn] training GNN epochs={epochs} hidden_dim={hidden_dim}...",
            file=sys.stderr,
            flush=True,
        )

    model = train_gnn(graphs=train_graphs, epochs=epochs, hidden_dim=hidden_dim)

    eval_rows: list[GNNForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for idx, origin in enumerate(eval_origins, start=1):
            if progress:
                print(
                    f"[gnn] eval {idx}/{len(eval_origins)} origin={origin.isoformat()}",
                    file=sys.stderr,
                    flush=True,
                )
            snap = _load_snapshot(origin)
            feature_rows = extract_features_for_origin(
                records=records,
                origin_date=origin,
                scoring_universe=scoring_universe,
                source_names=source_names,
            )
            graph = build_graph_from_snapshot(
                snapshot=snap,
                feature_rows=feature_rows,
                ablation=gnn_ablation,
            )
            _build_target_lookup_for_origin(origin)
            preds = predict_gnn(
                model=model,
                graph=graph,
                origin_date=origin,
                target_lookup=target_lookup,
                ablation=gnn_ablation,
            )
            for pred in preds:
                handle.write(pred.model_dump_json() + "\n")
            eval_rows.extend(preds)
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


def run_gnn_backtest_from_payloads(
    *,
    train_inputs: list[OriginInputs],
    eval_inputs: list[OriginInputs],
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
    out_path: Path,
    epochs: int,
    hidden_dim: int,
    gnn_ablation: Any | None = None,
    predictions_format: str = "jsonl.gz",
    progress: bool = False,
) -> dict[str, Any]:
    if progress:
        print(
            "[gnn] importing PyTorch (first use this run)...",
            file=sys.stderr,
            flush=True,
        )
    from baselines.gnn import GNNForecastRow, build_graph_from_snapshot, predict_gnn, train_gnn

    if not train_inputs:
        raise ValueError("at least one train input is required")
    if not eval_inputs:
        raise ValueError("at least one eval input is required")
    if predictions_format not in {"jsonl", "jsonl.gz"}:
        raise ValueError(f"unknown predictions format: {predictions_format}")

    train_graphs = []
    for index, item in enumerate(train_inputs, start=1):
        if progress:
            print(
                f"[gnn] build train {index}/{len(train_inputs)} origin={item.origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )
        train_graphs.append(
            build_graph_from_snapshot(
                snapshot=item.snapshot,
                feature_rows=item.feature_rows,
                ablation=gnn_ablation,
            )
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
    with open_text_auto(out_path, "w") as handle:
        for index, item in enumerate(eval_inputs, start=1):
            if progress:
                print(
                    f"[gnn] eval {index}/{len(eval_inputs)} origin={item.origin.isoformat()}",
                    file=sys.stderr,
                    flush=True,
                )
            graph = build_graph_from_snapshot(
                snapshot=item.snapshot,
                feature_rows=item.feature_rows,
                ablation=gnn_ablation,
            )
            preds = predict_gnn(
                model=model,
                graph=graph,
                origin_date=item.origin,
                target_lookup=target_lookup,
                ablation=gnn_ablation,
            )
            for pred in preds:
                handle.write(pred.model_dump_json() + "\n")
            eval_rows.extend(preds)
        handle.flush()

    model_name = "gnn_sage"
    all_inputs = train_inputs + eval_inputs
    audit: dict[str, Any] = {
        "model_name": model_name,
        "train_origin_start": min(item.origin for item in train_inputs).isoformat(),
        "train_origin_end": max(item.origin for item in train_inputs).isoformat(),
        "eval_origin_start": min(item.origin for item in eval_inputs).isoformat(),
        "eval_origin_end": max(item.origin for item in eval_inputs).isoformat(),
        "train_graph_count": len(train_graphs),
        "eval_row_count": len(eval_rows),
        "admin1_count": len({row.admin1_code for item in all_inputs for row in item.feature_rows}),
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "predictions_format": predictions_format,
        "brier": brier_score(eval_rows, model_name),
        "mae": mean_absolute_error(eval_rows, model_name),
        "top5_hit_rate": top_k_hit_rate(eval_rows, model_name, k=5),
        "recall_at_5": recall_at_k(eval_rows, model_name, k=5),
    }
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _parse_gnn_ablation_names(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    names: list[str] = []
    for value in values:
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return names


def run_gnn_ablation_backtest(
    *,
    tape_path: Path,
    snapshots_dir: Path,
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    eval_origin_start: dt.date,
    eval_origin_end: dt.date,
    out_path: Path,
    ablation_names: Sequence[str] | None = None,
    epochs: int = 30,
    hidden_dim: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )
    from baselines.features import extract_features_for_origin
    from baselines.gnn import (
        GNNForecastRow,
        build_graph_from_snapshot,
        predict_gnn,
        resolve_gnn_graph_ablations,
        train_gnn,
    )
    from ingest.snapshot_export import EXCLUDED_REGIONAL_ADMIN1_CODES, build_snapshot_payload

    ablations = resolve_gnn_graph_ablations(
        list(ablation_names) if ablation_names is not None else None
    )

    records = load_event_tape(tape_path)
    scoring_universe = sorted(
        {r.admin1_code for r in records if r.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES}
    )
    train_origins = weekly_origins(train_origin_start, train_origin_end)
    eval_origins = weekly_origins(eval_origin_start, eval_origin_end)

    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}

    def _load_snapshot(origin: dt.date) -> dict[str, Any]:
        for path in (
            snapshots_dir / f"as_of_{origin.isoformat()}.json",
            snapshots_dir / f"as_of_{origin.isoformat()}.json.gz",
        ):
            if path.exists():
                with open_text_auto(path, "r") as handle:
                    return json.load(handle)
        raise FileNotFoundError(f"missing snapshot for origin {origin.isoformat()} under {snapshots_dir}")

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

    train_inputs = []
    for idx, origin in enumerate(train_origins, start=1):
        train_inputs.append(
            (
                origin,
                _load_snapshot(origin),
                extract_features_for_origin(
                    records=records,
                    origin_date=origin,
                    scoring_universe=scoring_universe,
                ),
            )
        )
        _build_target_lookup_for_origin(origin)
        if progress:
            print(
                f"[gnn-ablation] load train {idx}/{len(train_origins)} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )

    eval_inputs = []
    for idx, origin in enumerate(eval_origins, start=1):
        eval_inputs.append(
            (
                origin,
                _load_snapshot(origin),
                extract_features_for_origin(
                    records=records,
                    origin_date=origin,
                    scoring_universe=scoring_universe,
                ),
            )
        )
        _build_target_lookup_for_origin(origin)
        if progress:
            print(
                f"[gnn-ablation] load eval {idx}/{len(eval_origins)} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_eval_rows: list[GNNForecastRow] = []
    ablation_audits: list[dict[str, Any]] = []
    with open_text_auto(out_path, "w") as handle:
        for ablation_idx, ablation in enumerate(ablations, start=1):
            model_name = f"gnn_sage__{ablation.name}"
            if progress:
                print(
                    (
                        f"[gnn-ablation] train {ablation_idx}/{len(ablations)} "
                        f"name={ablation.name} epochs={epochs} hidden_dim={hidden_dim}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            train_graphs = [
                build_graph_from_snapshot(
                    snapshot=snapshot,
                    feature_rows=feature_rows,
                    ablation=ablation,
                )
                for _, snapshot, feature_rows in train_inputs
            ]
            model = train_gnn(graphs=train_graphs, epochs=epochs, hidden_dim=hidden_dim)

            eval_rows: list[GNNForecastRow] = []
            for idx, (origin, snapshot, feature_rows) in enumerate(eval_inputs, start=1):
                graph = build_graph_from_snapshot(
                    snapshot=snapshot,
                    feature_rows=feature_rows,
                    ablation=ablation,
                )
                preds = predict_gnn(
                    model=model,
                    graph=graph,
                    origin_date=origin,
                    target_lookup=target_lookup,
                    model_name=model_name,
                    ablation=ablation,
                )
                for pred in preds:
                    handle.write(pred.model_dump_json() + "\n")
                eval_rows.extend(preds)
                if progress:
                    print(
                        (
                            f"[gnn-ablation] eval {idx}/{len(eval_inputs)} "
                            f"name={ablation.name} origin={origin.isoformat()}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            all_eval_rows.extend(eval_rows)
            ablation_audits.append(
                {
                    "ablation": ablation.metadata(),
                    "model_name": model_name,
                    "train_graph_count": len(train_graphs),
                    "eval_row_count": len(eval_rows),
                    "brier": brier_score(eval_rows, model_name),
                    "mae": mean_absolute_error(eval_rows, model_name),
                    "top5_hit_rate": top_k_hit_rate(eval_rows, model_name, k=5),
                    "recall_at_5": recall_at_k(eval_rows, model_name, k=5),
                }
            )
        handle.flush()

    full_graph_audit = next(
        (entry for entry in ablation_audits if entry["ablation"]["name"] == "full_graph"),
        None,
    )
    if full_graph_audit is not None:
        for entry in ablation_audits:
            entry["delta_vs_full_graph"] = {
                "brier": entry["brier"] - full_graph_audit["brier"],
                "mae": entry["mae"] - full_graph_audit["mae"],
                "top5_hit_rate": entry["top5_hit_rate"] - full_graph_audit["top5_hit_rate"],
                "recall_at_5": entry["recall_at_5"] - full_graph_audit["recall_at_5"],
            }

    audit: dict[str, Any] = {
        "model_family": "gnn_sage",
        "train_origin_start": train_origin_start.isoformat(),
        "train_origin_end": train_origin_end.isoformat(),
        "eval_origin_start": eval_origin_start.isoformat(),
        "eval_origin_end": eval_origin_end.isoformat(),
        "admin1_count": len(scoring_universe),
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "ablation_count": len(ablations),
        "eval_row_count": len(all_eval_rows),
        "ablations": ablation_audits,
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
    tabular.add_argument("--no-progress", dest="progress", action="store_false", default=True)
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
    gnn.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    gnn_ablations = subparsers.add_parser("gnn-ablations")
    gnn_ablations.add_argument("--tape", default="data/gdelt/tape/france_protest/events.jsonl")
    gnn_ablations.add_argument("--snapshots-dir", default="data/gdelt/snapshots/france_protest")
    gnn_ablations.add_argument("--train-origin-start", default="2021-01-04")
    gnn_ablations.add_argument("--train-origin-end", default="2024-12-30")
    gnn_ablations.add_argument("--eval-origin-start", default="2025-01-06")
    gnn_ablations.add_argument("--eval-origin-end", default="2025-12-29")
    gnn_ablations.add_argument(
        "--out",
        default="data/gdelt/baselines/france_protest/gnn_ablation_predictions.jsonl",
    )
    gnn_ablations.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Optional space- or comma-separated ablation names. Defaults to all configured ablations.",
    )
    gnn_ablations.add_argument("--epochs", type=int, default=30)
    gnn_ablations.add_argument("--hidden-dim", type=int, default=64)
    gnn_ablations.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    source_experiments = subparsers.add_parser("source-experiments")
    source_experiments.add_argument(
        "--tape",
        default=None,
        help="Legacy JSONL tape path. If omitted, the central warehouse is used.",
    )
    source_experiments.add_argument("--data-root", default=None)
    source_experiments.add_argument("--warehouse-path", default=None)
    source_experiments.add_argument(
        "--snapshots-root",
        default=None,
    )
    source_experiments.add_argument(
        "--out-root",
        default=None,
    )
    source_experiments.add_argument("--train-origin-start", default="2021-01-04")
    source_experiments.add_argument("--train-origin-end", default="2024-12-30")
    source_experiments.add_argument("--eval-origin-start", default="2025-01-06")
    source_experiments.add_argument("--eval-origin-end", default="2025-12-29")
    source_experiments.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional space- or comma-separated source experiment names. Defaults to all.",
    )
    source_experiments.add_argument("--epochs", type=int, default=30)
    source_experiments.add_argument("--hidden-dim", type=int, default=64)
    source_experiments.add_argument(
        "--snapshot-mode",
        choices=["in_memory", "in-memory", "materialize"],
        default="in-memory",
    )
    source_experiments.add_argument(
        "--snapshot-format",
        choices=["json", "json.gz"],
        default="json.gz",
    )
    source_experiments.add_argument(
        "--predictions-format",
        choices=["jsonl", "jsonl.gz"],
        default="jsonl.gz",
    )
    source_experiments.add_argument(
        "--progress",
        action="store_true",
        help="Print per-origin progress to stderr (default: on).",
    )
    source_experiments.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence per-origin progress.",
    )
    source_experiments.add_argument(
        "--no-recurrence",
        dest="run_recurrence",
        action="store_false",
        default=True,
        help="Skip recurrence baselines (GNN does not use them; they are for comparison metrics).",
    )
    source_experiments.add_argument(
        "--no-tabular",
        dest="run_tabular",
        action="store_false",
        default=True,
        help="Skip XGBoost tabular training (GNN uses snapshots only; tabular is for comparison metrics).",
    )
    source_experiments.add_argument(
        "--grounding-cache",
        default=None,
        help="Optional path to a JSON cache file for Wikidata Actor/Location grounding (API search).",
    )
    source_experiments.add_argument(
        "--grounding-dump-manifest",
        default=None,
        help=(
            "Optional path to a Wikidata dump-slice manifest JSON. "
            "For each origin date, the latest dump with dump_date <= origin is used."
        ),
    )
    source_experiments.add_argument(
        "--grounding-no-api-fallback",
        action="store_true",
        help="Disable live Wikidata API fallback when dump-slice lookup misses.",
    )
    source_experiments.add_argument(
        "--grounding-request-delay",
        type=float,
        default=0.25,
        help="Seconds to sleep after each Wikidata API cache miss (default: 0.25).",
    )
    source_experiments.add_argument(
        "--no-grounding-log",
        action="store_true",
        help="Silence stderr lines from Wikidata grounding (snapshot lines still use --progress).",
    )
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
                progress=args.progress,
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
                progress=args.progress,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "gnn-ablations":
        try:
            run_gnn_ablation_backtest(
                tape_path=Path(args.tape),
                snapshots_dir=Path(args.snapshots_dir),
                train_origin_start=dt.date.fromisoformat(args.train_origin_start),
                train_origin_end=dt.date.fromisoformat(args.train_origin_end),
                eval_origin_start=dt.date.fromisoformat(args.eval_origin_start),
                eval_origin_end=dt.date.fromisoformat(args.eval_origin_end),
                out_path=Path(args.out),
                ablation_names=_parse_gnn_ablation_names(args.ablations),
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                progress=args.progress,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "source-experiments":
        from baselines.source_experiments import (
            _parse_experiment_names,
            run_source_layer_experiments,
        )
        from ingest.paths import resolve_data_root, runs_root

        try:
            data_root = resolve_data_root(args.data_root)
            out_root = (
                Path(args.out_root)
                if args.out_root is not None
                else runs_root(data_root) / "source_experiments"
            )
            data_root_arg = data_root if args.tape is None or args.data_root is not None else None
            run_source_layer_experiments(
                tape_path=Path(args.tape) if args.tape is not None else None,
                warehouse_path=Path(args.warehouse_path) if args.warehouse_path is not None else None,
                data_root=data_root_arg,
                snapshots_root=Path(args.snapshots_root) if args.snapshots_root is not None else None,
                out_root=out_root,
                train_origin_start=dt.date.fromisoformat(args.train_origin_start),
                train_origin_end=dt.date.fromisoformat(args.train_origin_end),
                eval_origin_start=dt.date.fromisoformat(args.eval_origin_start),
                eval_origin_end=dt.date.fromisoformat(args.eval_origin_end),
                experiment_names=_parse_experiment_names(args.experiments),
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                snapshot_mode=args.snapshot_mode.replace("-", "_"),
                snapshot_format=args.snapshot_format,
                predictions_format=args.predictions_format,
                progress=not args.no_progress,
                run_recurrence=args.run_recurrence,
                run_tabular=args.run_tabular,
                grounding_cache=(
                    Path(args.grounding_cache).expanduser().resolve()
                    if args.grounding_cache
                    else None
                ),
                grounding_request_delay_s=args.grounding_request_delay,
                grounding_log=not args.no_grounding_log,
                grounding_dump_manifest=(
                    Path(args.grounding_dump_manifest).expanduser().resolve()
                    if args.grounding_dump_manifest
                    else None
                ),
                grounding_api_fallback=not args.grounding_no_api_fallback,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
