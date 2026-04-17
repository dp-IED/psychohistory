"""Source-layer experiment orchestration for mixed event tapes."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from baselines.backtest import OriginInputs, build_audit, weekly_origins
from baselines.features import extract_features_for_origin
from baselines.metrics import (
    balanced_accuracy,
    brier_score,
    mean_absolute_error,
    positive_rate,
    pr_auc,
    recall_at_k,
    top_k_hit_rate,
)
from baselines.recurrence import ForecastRow, RECURRENCE_MODEL_NAMES
from baselines.recurrence import build_recurrence_forecasts_for_origin
from baselines.tabular import predict_tabular, train_tabular_model
from evals.graph_artifact_contract import GraphArtifactV1
from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.event_warehouse import query_records, source_counts
from ingest.io_utils import open_text_auto, write_json_atomic
from ingest.paths import resolve_data_root, warehouse_path as default_warehouse_path
from ingest.snapshot_export import (
    EXCLUDED_REGIONAL_ADMIN1_CODES,
    SourceIdentityMode,
    build_snapshot_payload,
)


@dataclass(frozen=True)
class SourceExperiment:
    name: str
    source_names: set[str]
    source_identity_mode: SourceIdentityMode = "preserve"
    use_event_features: bool = True
    description: str = ""


SOURCE_EXPERIMENTS: tuple[SourceExperiment, ...] = (
    SourceExperiment(
        name="gdelt_only",
        source_names={"gdelt_v2_events"},
        description="GDELT event tape and GDELT source identity only.",
    ),
    SourceExperiment(
        name="acled_only",
        source_names={"acled"},
        description="ACLED event tape and ACLED source identity only.",
    ),
    SourceExperiment(
        name="gdelt_plus_acled",
        source_names={"gdelt_v2_events", "acled"},
        description="Combined GDELT and ACLED evidence with source identity preserved.",
    ),
    SourceExperiment(
        name="gdelt_plus_acled_no_source_identity",
        source_names={"gdelt_v2_events", "acled"},
        source_identity_mode="collapse",
        description="Combined evidence with source nodes collapsed.",
    ),
    SourceExperiment(
        name="gdelt_plus_acled_no_event_attributes",
        source_names={"gdelt_v2_events", "acled"},
        use_event_features=False,
        description="Combined evidence with event attribute vectors zeroed.",
    ),
)


def resolve_source_experiments(names: Sequence[str] | None = None) -> list[SourceExperiment]:
    by_name = {experiment.name: experiment for experiment in SOURCE_EXPERIMENTS}
    if len(by_name) != len(SOURCE_EXPERIMENTS):
        name_counts = Counter(experiment.name for experiment in SOURCE_EXPERIMENTS)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        raise ValueError(f"duplicate source experiment names: {duplicates}")

    if names is None:
        return list(SOURCE_EXPERIMENTS)

    resolved: list[SourceExperiment] = []
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate source experiment: {name}")
        try:
            resolved.append(by_name[name])
        except KeyError as exc:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"unknown source experiment '{name}'. Available: {available}") from exc
        seen.add(name)
    if not resolved:
        raise ValueError("at least one source experiment is required")
    return resolved


def _validate_requested_sources(
    *,
    experiment: SourceExperiment,
    available_sources: set[str],
    data_context: str,
) -> None:
    missing = experiment.source_names - available_sources
    if missing:
        raise ValueError(
            "source experiment "
            f"'{experiment.name}' requested missing sources {sorted(missing)}; "
            f"available={sorted(available_sources)}; {data_context}"
        )


def _filtered_records(
    records: list[EventTapeRecord],
    source_names: set[str],
) -> list[EventTapeRecord]:
    return [record for record in records if record.source_name in source_names]


def _load_forecast_rows(path: Path) -> list[ForecastRow]:
    with open_text_auto(path, "r") as handle:
        return [ForecastRow.model_validate_json(line) for line in handle if line.strip()]


def _metrics_for_model(rows: list[ForecastRow], model_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row.model_name == model_name]
    if not selected:
        return {
            "row_count": 0,
            "origin_count": 0,
            "admin1_count": 0,
            "positive_count": 0,
            "positive_rate": 0.0,
            "brier": 0.0,
            "mae": 0.0,
            "pr_auc": 0.0,
            "balanced_accuracy": 0.0,
            "top5_hit_rate": 0.0,
            "recall_at_5": 0.0,
        }
    return {
        "row_count": len(selected),
        "origin_count": len({row.forecast_origin for row in selected}),
        "admin1_count": len({row.admin1_code for row in selected}),
        "positive_count": sum(1 for row in selected if row.target_occurs_next_7d),
        "positive_rate": positive_rate(rows, model_name),
        "brier": brier_score(rows, model_name),
        "mae": mean_absolute_error(rows, model_name),
        "pr_auc": pr_auc(rows, model_name),
        "balanced_accuracy": balanced_accuracy(rows, model_name),
        "top5_hit_rate": top_k_hit_rate(rows, model_name, k=5),
        "recall_at_5": recall_at_k(rows, model_name, k=5),
    }


def _snapshot_count_totals(snapshot_paths: list[Path], key: str) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for path in snapshot_paths:
        with open_text_auto(path, "r") as handle:
            payload = json.load(handle)
        totals.update(payload.get("metadata", {}).get(key, {}))
    return dict(sorted(totals.items()))


def _payload_count_totals(payloads: list[dict[str, Any]], key: str) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for payload in payloads:
        totals.update(payload.get("metadata", {}).get(key, {}))
    return dict(sorted(totals.items()))


def _aggregate_wikidata_grounding(payloads: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for payload in payloads:
        block = payload.get("metadata", {}).get("wikidata_grounding")
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return dict(sorted(totals.items()))


def _prediction_path(out_dir: Path, stem: str, predictions_format: str) -> Path:
    if predictions_format == "jsonl":
        return out_dir / f"{stem}.jsonl"
    if predictions_format == "jsonl.gz":
        return out_dir / f"{stem}.jsonl.gz"
    raise ValueError(f"unknown predictions format: {predictions_format}")


def _snapshot_path(out_dir: Path, origin: dt.date, snapshot_format: str) -> Path:
    if snapshot_format == "json":
        return out_dir / f"as_of_{origin.isoformat()}.json"
    if snapshot_format == "json.gz":
        return out_dir / f"as_of_{origin.isoformat()}.json.gz"
    raise ValueError(f"unknown snapshot format: {snapshot_format}")


def _update_target_lookup(
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
    *,
    origin: dt.date,
    payload: dict[str, Any],
) -> None:
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


def _build_origin_inputs(
    *,
    records: list[EventTapeRecord],
    origins: list[dt.date],
    scoring_universe: list[str],
    source_identity_mode: SourceIdentityMode,
    progress_label: str,
    progress: bool,
    grounding_cache: Path | None = None,
    grounding_request_delay_s: float = 0.25,
    grounding_log: bool = True,
) -> tuple[list[OriginInputs], dict[tuple[dt.date, str], tuple[int, bool]]]:
    inputs: list[OriginInputs] = []
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]] = {}
    for index, origin in enumerate(origins, start=1):
        if progress:
            print(
                (
                    f"[source-experiments] {progress_label} {index}/{len(origins)} "
                    f"start origin={origin.isoformat()} "
                    f"records={len(records)} grounding_cache={grounding_cache}"
                ),
                file=sys.stderr,
                flush=True,
            )
        snapshot = build_snapshot_payload(
            records=records,
            origin_date=origin,
            source_identity_mode=source_identity_mode,
            grounding_cache=grounding_cache,
            grounding_request_delay_s=grounding_request_delay_s,
            grounding_log=grounding_log,
        )
        feature_rows = extract_features_for_origin(
            records=records,
            origin_date=origin,
            scoring_universe=scoring_universe,
        )
        _update_target_lookup(target_lookup, origin=origin, payload=snapshot)
        inputs.append(OriginInputs(origin=origin, snapshot=snapshot, feature_rows=feature_rows))
        if progress:
            print(
                f"[source-experiments] {progress_label} {index}/{len(origins)} origin={origin.isoformat()}",
                file=sys.stderr,
                flush=True,
            )
    return inputs, target_lookup


def _write_snapshot_payloads(
    *,
    out_dir: Path,
    inputs: list[OriginInputs],
    snapshot_format: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in inputs:
        try:
            GraphArtifactV1.model_validate(item.snapshot)
        except Exception:
            invalid_path = out_dir / f"as_of_{item.origin.isoformat()}.invalid.json"
            write_json_atomic(invalid_path, item.snapshot)
            raise
        path = _snapshot_path(out_dir, item.origin, snapshot_format)
        write_json_atomic(path, item.snapshot)
        written.append(path)
    return written


def _run_recurrence_from_records(
    *,
    records: list[EventTapeRecord],
    origin_start: dt.date,
    origin_end: dt.date,
    out_path: Path,
    progress: bool,
) -> dict[str, Any]:
    origins = weekly_origins(origin_start, origin_end)
    rows: list[ForecastRow] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for index, origin in enumerate(origins, start=1):
            origin_rows = build_recurrence_forecasts_for_origin(
                records=records,
                forecast_origin=origin,
            )
            rows.extend(origin_rows)
            for row in origin_rows:
                handle.write(row.model_dump_json() + "\n")
            if progress:
                print(
                    f"[recurrence] {index}/{len(origins)} origin={origin.isoformat()} rows={len(rows)}",
                    file=sys.stderr,
                    flush=True,
                )
    audit = build_audit(rows, origin_start=origin_start, origin_end=origin_end)
    out_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _run_tabular_from_inputs(
    *,
    train_inputs: list[OriginInputs],
    eval_inputs: list[OriginInputs],
    target_lookup: dict[tuple[dt.date, str], tuple[int, bool]],
    out_path: Path,
    progress: bool,
) -> dict[str, Any]:
    train_feature_rows = [row for item in train_inputs for row in item.feature_rows]
    train_origin_set = {item.origin for item in train_inputs}
    train_targets = {key: value[1] for key, value in target_lookup.items() if key[0] in train_origin_set}
    if progress:
        print("[tabular] training XGBoost model...", file=sys.stderr, flush=True)
    model = train_tabular_model(feature_rows=train_feature_rows, targets=train_targets)

    eval_rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text_auto(out_path, "w") as handle:
        for item in eval_inputs:
            preds = predict_tabular(
                model=model,
                feature_rows=item.feature_rows,
                target_lookup=target_lookup,
            )
            for pred in preds:
                handle.write(pred.model_dump_json() + "\n")
            eval_rows.extend(preds)

    model_name = "xgboost_tabular"
    audit: dict[str, Any] = {
        "model_name": model_name,
        "train_origin_start": min(item.origin for item in train_inputs).isoformat(),
        "train_origin_end": max(item.origin for item in train_inputs).isoformat(),
        "eval_origin_start": min(item.origin for item in eval_inputs).isoformat(),
        "eval_origin_end": max(item.origin for item in eval_inputs).isoformat(),
        "train_row_count": len(train_feature_rows),
        "eval_row_count": len(eval_rows),
        "admin1_count": len({row.admin1_code for item in train_inputs + eval_inputs for row in item.feature_rows}),
        "brier": brier_score(eval_rows, model_name),
        "mae": mean_absolute_error(eval_rows, model_name),
        "top5_hit_rate": top_k_hit_rate(eval_rows, model_name, k=5),
        "recall_at_5": recall_at_k(eval_rows, model_name, k=5),
    }
    out_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _parse_experiment_names(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    names: list[str] = []
    for value in values:
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return names


def run_source_layer_experiments(
    *,
    tape_path: Path | None = None,
    warehouse_path: Path | None = None,
    data_root: Path | None = None,
    snapshots_root: Path | None = None,
    out_root: Path = Path("data/mixed/baselines/france_protest_source_experiments"),
    train_origin_start: dt.date,
    train_origin_end: dt.date,
    eval_origin_start: dt.date,
    eval_origin_end: dt.date,
    experiment_names: Sequence[str] | None = None,
    epochs: int = 30,
    hidden_dim: int = 64,
    snapshot_mode: Literal["in_memory", "materialize"] = "in_memory",
    snapshot_format: Literal["json", "json.gz"] = "json.gz",
    predictions_format: Literal["jsonl", "jsonl.gz"] = "jsonl.gz",
    progress: bool = False,
    run_recurrence: bool = True,
    run_tabular: bool = True,
    grounding_cache: Path | None = None,
    grounding_request_delay_s: float = 0.25,
    grounding_log: bool = True,
) -> dict[str, Any]:
    from baselines.backtest import run_gnn_backtest_from_payloads
    from baselines.gnn import GNNGraphAblation

    if train_origin_end >= eval_origin_start:
        raise ValueError(
            f"train_origin_end ({train_origin_end}) must be before eval_origin_start ({eval_origin_start})"
        )
    if snapshot_mode == "in-memory":
        snapshot_mode = "in_memory"
    if snapshot_mode not in {"in_memory", "materialize"}:
        raise ValueError(f"unknown snapshot mode: {snapshot_mode}")
    if snapshot_format not in {"json", "json.gz"}:
        raise ValueError(f"unknown snapshot format: {snapshot_format}")
    if predictions_format not in {"jsonl", "jsonl.gz"}:
        raise ValueError(f"unknown predictions format: {predictions_format}")

    experiments = resolve_source_experiments(experiment_names)
    requested_sources = set().union(*(experiment.source_names for experiment in experiments))

    data_context: str
    if warehouse_path is not None or data_root is not None or tape_path is None:
        resolved_data_root = resolve_data_root(data_root)
        resolved_warehouse_path = (
            Path(warehouse_path).expanduser().resolve()
            if warehouse_path is not None
            else default_warehouse_path(resolved_data_root)
        )
        if not resolved_warehouse_path.exists():
            raise FileNotFoundError(
                "missing event warehouse: "
                f"{resolved_warehouse_path}; run python -m ingest.event_warehouse init and import data first"
            )
        available_counts = source_counts(resolved_warehouse_path)
        available_sources = set(available_counts)
        records = query_records(
            db_path=resolved_warehouse_path,
            source_names=requested_sources,
        )
        data_context = f"warehouse={resolved_warehouse_path}"
    else:
        records = load_event_tape(tape_path)
        available_sources = {record.source_name for record in records}
        data_context = f"tape={tape_path}"

    out_root.mkdir(parents=True, exist_ok=True)
    if snapshot_mode == "materialize":
        snapshots_root = snapshots_root or (out_root / "snapshots")

    audit: dict[str, Any] = {
        "model_family": "source_layer_experiments",
        "train_origin_start": train_origin_start.isoformat(),
        "train_origin_end": train_origin_end.isoformat(),
        "eval_origin_start": eval_origin_start.isoformat(),
        "eval_origin_end": eval_origin_end.isoformat(),
        "data_context": data_context,
        "snapshot_mode": snapshot_mode,
        "snapshot_format": snapshot_format,
        "predictions_format": predictions_format,
        "experiments": [],
    }

    train_origins = weekly_origins(train_origin_start, train_origin_end)
    eval_origins = weekly_origins(eval_origin_start, eval_origin_end)

    for index, experiment in enumerate(experiments, start=1):
        _validate_requested_sources(
            experiment=experiment,
            available_sources=available_sources,
            data_context=data_context,
        )
        experiment_records = _filtered_records(records, experiment.source_names)
        scoring_universe = sorted(
            {
                record.admin1_code
                for record in experiment_records
                if record.admin1_code not in EXCLUDED_REGIONAL_ADMIN1_CODES
            }
        )
        if not scoring_universe:
            raise ValueError(
                f"source experiment '{experiment.name}' has no regional scoring admin1 codes; "
                f"requested={sorted(experiment.source_names)} {data_context}"
            )

        experiment_snapshot_dir = snapshots_root / experiment.name if snapshots_root is not None else None
        experiment_out_dir = out_root / experiment.name
        experiment_out_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            print(
                (
                    f"[source-experiments] {index}/{len(experiments)} "
                    f"name={experiment.name} sources={sorted(experiment.source_names)}"
                ),
                file=sys.stderr,
                flush=True,
            )

        train_inputs, train_targets = _build_origin_inputs(
            records=experiment_records,
            origins=train_origins,
            scoring_universe=scoring_universe,
            source_identity_mode=experiment.source_identity_mode,
            progress_label=f"{experiment.name} train payloads",
            progress=progress,
            grounding_cache=grounding_cache,
            grounding_request_delay_s=grounding_request_delay_s,
            grounding_log=grounding_log,
        )
        eval_inputs, eval_targets = _build_origin_inputs(
            records=experiment_records,
            origins=eval_origins,
            scoring_universe=scoring_universe,
            source_identity_mode=experiment.source_identity_mode,
            progress_label=f"{experiment.name} eval payloads",
            progress=progress,
            grounding_cache=grounding_cache,
            grounding_request_delay_s=grounding_request_delay_s,
            grounding_log=grounding_log,
        )
        target_lookup = {**train_targets, **eval_targets}
        snapshot_payloads = [item.snapshot for item in train_inputs + eval_inputs]
        snapshot_paths: list[Path] = []
        if snapshot_mode == "materialize":
            assert experiment_snapshot_dir is not None
            snapshot_paths = _write_snapshot_payloads(
                out_dir=experiment_snapshot_dir,
                inputs=train_inputs + eval_inputs,
                snapshot_format=snapshot_format,
            )

        recurrence_path: Path | None = None
        if run_recurrence:
            recurrence_path = _prediction_path(
                experiment_out_dir,
                "recurrence_predictions",
                predictions_format,
            )
            _run_recurrence_from_records(
                records=experiment_records,
                origin_start=eval_origin_start,
                origin_end=eval_origin_end,
                out_path=recurrence_path,
                progress=progress,
            )

        tabular_path: Path | None = None
        if run_tabular:
            tabular_path = _prediction_path(
                experiment_out_dir,
                "tabular_predictions",
                predictions_format,
            )
            _run_tabular_from_inputs(
                train_inputs=train_inputs,
                eval_inputs=eval_inputs,
                target_lookup=target_lookup,
                out_path=tabular_path,
                progress=progress,
            )

        gnn_path = _prediction_path(experiment_out_dir, "gnn_predictions", predictions_format)
        gnn_ablation = None
        if not experiment.use_event_features:
            gnn_ablation = GNNGraphAblation(
                name="source_experiment_no_event_attributes",
                use_event_features=False,
                description="Source experiment GNN with event attributes zeroed.",
            )
        if progress:
            print(
                (
                    f"[source-experiments] {experiment.name} starting gnn_sage "
                    f"train_origins={len(train_inputs)} eval_origins={len(eval_inputs)} "
                    f"epochs={epochs} hidden_dim={hidden_dim} out={gnn_path}"
                ),
                file=sys.stderr,
                flush=True,
            )
        run_gnn_backtest_from_payloads(
            train_inputs=train_inputs,
            eval_inputs=eval_inputs,
            target_lookup=target_lookup,
            out_path=gnn_path,
            gnn_ablation=gnn_ablation,
            epochs=epochs,
            hidden_dim=hidden_dim,
            predictions_format=predictions_format,
            progress=progress,
        )

        recurrence_rows = (
            _load_forecast_rows(recurrence_path) if recurrence_path is not None else []
        )
        tabular_rows = _load_forecast_rows(tabular_path) if tabular_path is not None else []
        gnn_rows = _load_forecast_rows(gnn_path)

        experiment_audit = {
            "name": experiment.name,
            "description": experiment.description,
            "source_names": sorted(experiment.source_names),
            "source_identity_mode": experiment.source_identity_mode,
            "use_event_features": experiment.use_event_features,
            "recurrence_skipped": not run_recurrence,
            "tabular_skipped": not run_tabular,
            "models": {
                **{
                    model_name: _metrics_for_model(recurrence_rows, model_name)
                    for model_name in RECURRENCE_MODEL_NAMES
                },
                "xgboost_tabular": _metrics_for_model(tabular_rows, "xgboost_tabular"),
                "gnn_sage": _metrics_for_model(gnn_rows, "gnn_sage"),
            },
            "data_audit": {
                "event_count": len(experiment_records),
                "feature_source_counts_total": _payload_count_totals(
                    snapshot_payloads,
                    "feature_source_counts",
                ),
                "label_source_counts_total": _payload_count_totals(
                    snapshot_payloads,
                    "label_source_counts",
                ),
                "admin1_count": len(scoring_universe),
                "materialized_snapshot_count": len(snapshot_paths),
                "wikidata_grounding_total": _aggregate_wikidata_grounding(snapshot_payloads),
            },
        }
        audit["experiments"].append(experiment_audit)

    audit_path = out_root / "source_experiments.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
