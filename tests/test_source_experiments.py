from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from baselines.recurrence import ForecastRow
from baselines.source_experiments import (
    SOURCE_EXPERIMENTS,
    resolve_source_experiments,
    run_source_layer_experiments,
)
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import init_warehouse, upsert_records
from ingest.io_utils import open_text_auto


def _record(
    source_name: str,
    source_event_id: str,
    *,
    event_date: dt.date,
    admin1_code: str,
) -> EventTapeRecord:
    return EventTapeRecord(
        source_name=source_name,
        source_event_id=source_event_id,
        event_date=event_date,
        source_available_at=dt.datetime.combine(
            event_date + dt.timedelta(days=1),
            dt.time(),
            tzinfo=dt.timezone.utc,
        ),
        retrieved_at=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        country_code="FR",
        admin1_code=admin1_code,
        location_name=None,
        latitude=None,
        longitude=None,
        event_class="protest",
        event_code="141" if source_name == "gdelt_v2_events" else "Protests",
        event_base_code="14" if source_name == "gdelt_v2_events" else "Protests",
        event_root_code="14" if source_name == "gdelt_v2_events" else "Protests",
        quad_class=3 if source_name == "gdelt_v2_events" else None,
        goldstein_scale=-6.5 if source_name == "gdelt_v2_events" else None,
        num_mentions=4 if source_name == "gdelt_v2_events" else None,
        num_sources=None,
        num_articles=3 if source_name == "gdelt_v2_events" else None,
        avg_tone=-1.5 if source_name == "gdelt_v2_events" else None,
        actor1_name=None,
        actor1_country_code=None,
        actor2_name=None,
        actor2_country_code=None,
        source_url=None,
        raw={},
    )


def _warehouse_with_records(tmp_path: Path, records: list[EventTapeRecord]) -> Path:
    db_path = tmp_path / "warehouse" / "events.duckdb"
    init_warehouse(db_path)
    upsert_records(db_path=db_path, records=records)
    return db_path


def _mixed_records() -> list[EventTapeRecord]:
    records: list[EventTapeRecord] = []
    first_origin = dt.date(2021, 1, 4)
    for week in range(12):
        origin = first_origin + dt.timedelta(weeks=week)
        admin1_code = "FR11" if week % 2 == 0 else "FR22"
        event_date = origin - dt.timedelta(days=3)
        records.append(
            _record(
                "gdelt_v2_events",
                f"gdelt:{week}",
                event_date=event_date,
                admin1_code=admin1_code,
            )
        )
        records.append(
            _record(
                "acled",
                f"acled:FRA{week}",
                event_date=event_date,
                admin1_code=admin1_code,
            )
        )
    return records


def test_resolve_source_experiments_defaults_and_rejects_bad_names() -> None:
    defaults = resolve_source_experiments()

    assert [experiment.name for experiment in defaults] == [
        "gdelt_only",
        "acled_only",
        "gdelt_plus_acled",
        "gdelt_plus_acled_no_source_identity",
        "gdelt_plus_acled_no_event_attributes",
    ]
    assert len(SOURCE_EXPERIMENTS) == 5
    with pytest.raises(ValueError, match="unknown source experiment"):
        resolve_source_experiments(["not_real"])
    with pytest.raises(ValueError, match="duplicate source experiment"):
        resolve_source_experiments(["gdelt_only", "gdelt_only"])


@pytest.mark.torch_train
def test_source_experiment_runner_writes_audit_and_predictions(tmp_path: Path) -> None:
    snapshots_root = tmp_path / "snapshots"
    out_root = tmp_path / "out"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    audit = run_source_layer_experiments(
        warehouse_path=wh,
        snapshots_root=snapshots_root,
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
    )

    audit_path = out_root / "source_experiments.audit.json"
    assert audit_path.exists()
    assert audit["experiments"][0]["models"]["gnn_sage"]["row_count"] > 0
    assert (out_root / "gdelt_plus_acled" / "recurrence_predictions.jsonl.gz").exists()
    assert (out_root / "gdelt_plus_acled" / "tabular_predictions.jsonl.gz").exists()
    assert (out_root / "gdelt_plus_acled" / "gnn_predictions.jsonl.gz").exists()
    assert not snapshots_root.exists()


def test_source_experiment_runner_skip_tabular_omits_xgboost_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_gnn(
        *,
        out_path: Path,
        train_inputs: list,
        eval_inputs: list,
        **kwargs: object,
    ) -> dict[str, object]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        origin = eval_inputs[0].origin
        row = ForecastRow(
            forecast_origin=origin,
            admin1_code="FR11",
            model_name="gnn_sage",
            predicted_count=0.0,
            predicted_occurrence_probability=0.5,
            target_count_next_7d=0,
            target_occurs_next_7d=False,
        )
        with open_text_auto(out_path, "w") as handle:
            handle.write(row.model_dump_json() + "\n")
        return {"model_name": "gnn_sage", "eval_row_count": 1}

    def tabular_should_not_run(**kwargs: object) -> None:
        raise AssertionError("XGBoost tabular path should be skipped when run_tabular=False")

    monkeypatch.setattr("baselines.backtest.run_gnn_backtest_from_payloads", fake_gnn)
    monkeypatch.setattr("baselines.source_experiments._run_tabular_from_inputs", tabular_should_not_run)

    out_root = tmp_path / "out"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    audit = run_source_layer_experiments(
        warehouse_path=wh,
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
        run_tabular=False,
    )

    assert audit["experiments"][0]["tabular_skipped"] is True
    gnn = audit["experiments"][0]["models"]["gnn_sage"]
    assert gnn["row_count"] > 0
    assert "positive_rate" in gnn
    assert "pr_auc" in gnn
    assert "balanced_accuracy" in gnn
    assert not (out_root / "gdelt_plus_acled" / "tabular_predictions.jsonl.gz").exists()
    assert (out_root / "gdelt_plus_acled" / "gnn_predictions.jsonl.gz").exists()


def test_source_experiment_runner_skip_recurrence_omits_recurrence_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_gnn(
        *,
        out_path: Path,
        train_inputs: list,
        eval_inputs: list,
        **kwargs: object,
    ) -> dict[str, object]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        origin = eval_inputs[0].origin
        row = ForecastRow(
            forecast_origin=origin,
            admin1_code="FR11",
            model_name="gnn_sage",
            predicted_count=0.0,
            predicted_occurrence_probability=0.5,
            target_count_next_7d=0,
            target_occurs_next_7d=False,
        )
        with open_text_auto(out_path, "w") as handle:
            handle.write(row.model_dump_json() + "\n")
        return {"model_name": "gnn_sage", "eval_row_count": 1}

    def recurrence_should_not_run(**kwargs: object) -> None:
        raise AssertionError("recurrence should be skipped when run_recurrence=False")

    monkeypatch.setattr("baselines.backtest.run_gnn_backtest_from_payloads", fake_gnn)
    monkeypatch.setattr(
        "baselines.source_experiments._run_recurrence_from_records",
        recurrence_should_not_run,
    )

    out_root = tmp_path / "out"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    audit = run_source_layer_experiments(
        warehouse_path=wh,
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
        run_recurrence=False,
        run_tabular=False,
    )

    assert audit["experiments"][0]["recurrence_skipped"] is True
    assert audit["experiments"][0]["models"]["gnn_sage"]["row_count"] > 0
    assert not (out_root / "gdelt_plus_acled" / "recurrence_predictions.jsonl.gz").exists()
    assert (out_root / "gdelt_plus_acled" / "gnn_predictions.jsonl.gz").exists()


@pytest.mark.torch_train
def test_source_experiment_runner_uses_collapsed_source_snapshots(tmp_path: Path) -> None:
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    run_source_layer_experiments(
        warehouse_path=wh,
        snapshots_root=tmp_path / "snapshots",
        out_root=tmp_path / "out",
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled_no_source_identity"],
        epochs=1,
        hidden_dim=8,
        snapshot_mode="materialize",
    )

    snapshot_path = (
        tmp_path
        / "snapshots"
        / "gdelt_plus_acled_no_source_identity"
        / "as_of_2021-03-01.json.gz"
    )
    with open_text_auto(snapshot_path, "r") as handle:
        payload = json.load(handle)
    assert {node["id"] for node in payload["nodes"] if node["type"] == "Source"} == {
        "source:events"
    }


@pytest.mark.torch_train
def test_source_experiment_runner_zeroes_event_features_for_gnn(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    run_source_layer_experiments(
        warehouse_path=wh,
        snapshots_root=tmp_path / "snapshots",
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled_no_event_attributes"],
        epochs=1,
        hidden_dim=8,
    )

    with open_text_auto(
        out_root / "gdelt_plus_acled_no_event_attributes" / "gnn_predictions.jsonl.gz",
        "r",
    ) as handle:
        first_row = json.loads(next(handle))
    assert first_row["metadata"]["ablation"]["use_event_features"] is False


def test_source_experiment_runner_fails_when_requested_source_is_missing(tmp_path: Path) -> None:
    wh = _warehouse_with_records(
        tmp_path,
        [record for record in _mixed_records() if record.source_name == "gdelt_v2_events"],
    )

    with pytest.raises(ValueError, match="requested missing sources"):
        run_source_layer_experiments(
            warehouse_path=wh,
            snapshots_root=tmp_path / "snapshots",
            out_root=tmp_path / "out",
            train_origin_start=dt.date(2021, 1, 4),
            train_origin_end=dt.date(2021, 2, 22),
            eval_origin_start=dt.date(2021, 3, 1),
            eval_origin_end=dt.date(2021, 3, 15),
            experiment_names=["acled_only"],
            epochs=1,
            hidden_dim=8,
        )


@pytest.mark.torch_train
def test_source_experiment_runner_reads_from_warehouse(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse" / "events.duckdb"
    out_root = tmp_path / "out"
    init_warehouse(db_path)
    upsert_records(db_path=db_path, records=_mixed_records())

    audit = run_source_layer_experiments(
        warehouse_path=db_path,
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
    )

    assert audit["data_context"].endswith(str(db_path))
    assert (out_root / "gdelt_plus_acled" / "gnn_predictions.jsonl.gz").exists()


@pytest.mark.torch_train
def test_source_experiment_materializes_snapshots_as_gzip(tmp_path: Path) -> None:
    snapshots_root = tmp_path / "snapshots"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    run_source_layer_experiments(
        warehouse_path=wh,
        snapshots_root=snapshots_root,
        out_root=tmp_path / "out",
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
        snapshot_mode="materialize",
    )

    assert (
        snapshots_root / "gdelt_plus_acled" / "as_of_2021-03-01.json.gz"
    ).exists()


def test_source_experiment_wikidata_grounding_total_in_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_apply(payload: dict, *, cache_path: Path, request_delay_s: float = 0.25, **kwargs: object) -> dict:
        payload.setdefault("metadata", {})["wikidata_grounding"] = {
            "resolved": 1,
            "attempted": 1,
            "skipped_existing_qid": 0,
            "skipped_no_label": 0,
            "api_calls": 0,
        }
        return payload

    monkeypatch.setattr("evals.wikidata_grounding.apply_wikidata_grounding", fake_apply)

    def fake_gnn(
        *,
        out_path: Path,
        train_inputs: list,
        eval_inputs: list,
        **kwargs: object,
    ) -> dict[str, object]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        origin = eval_inputs[0].origin
        row = ForecastRow(
            forecast_origin=origin,
            admin1_code="FR11",
            model_name="gnn_sage",
            predicted_count=0.0,
            predicted_occurrence_probability=0.5,
            target_count_next_7d=0,
            target_occurs_next_7d=False,
        )
        with open_text_auto(out_path, "w") as handle:
            handle.write(row.model_dump_json() + "\n")
        return {"model_name": "gnn_sage", "eval_row_count": 1}

    monkeypatch.setattr("baselines.backtest.run_gnn_backtest_from_payloads", fake_gnn)

    out_root = tmp_path / "out"
    wh = _warehouse_with_records(tmp_path, _mixed_records())

    audit = run_source_layer_experiments(
        warehouse_path=wh,
        out_root=out_root,
        train_origin_start=dt.date(2021, 1, 4),
        train_origin_end=dt.date(2021, 2, 22),
        eval_origin_start=dt.date(2021, 3, 1),
        eval_origin_end=dt.date(2021, 3, 15),
        experiment_names=["gdelt_plus_acled"],
        epochs=1,
        hidden_dim=8,
        run_tabular=False,
        grounding_cache=tmp_path / "wikidata_cache.json",
    )

    total = audit["experiments"][0]["data_audit"]["wikidata_grounding_total"]
    assert total.get("resolved", 0) >= 1
