from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from baselines.source_experiments import (
    SOURCE_EXPERIMENTS,
    resolve_source_experiments,
    run_source_layer_experiments,
)
from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import upsert_records
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


def _write_tape(path: Path, records: list[EventTapeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


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
    tape_path = tmp_path / "events.jsonl"
    snapshots_root = tmp_path / "snapshots"
    out_root = tmp_path / "out"
    _write_tape(tape_path, _mixed_records())

    audit = run_source_layer_experiments(
        tape_path=tape_path,
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


@pytest.mark.torch_train
def test_source_experiment_runner_uses_collapsed_source_snapshots(tmp_path: Path) -> None:
    tape_path = tmp_path / "events.jsonl"
    _write_tape(tape_path, _mixed_records())

    run_source_layer_experiments(
        tape_path=tape_path,
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
    tape_path = tmp_path / "events.jsonl"
    out_root = tmp_path / "out"
    _write_tape(tape_path, _mixed_records())

    run_source_layer_experiments(
        tape_path=tape_path,
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
    tape_path = tmp_path / "events.jsonl"
    _write_tape(tape_path, [record for record in _mixed_records() if record.source_name == "gdelt_v2_events"])

    with pytest.raises(ValueError, match="requested missing sources"):
        run_source_layer_experiments(
            tape_path=tape_path,
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
    tape_path = tmp_path / "events.jsonl"
    snapshots_root = tmp_path / "snapshots"
    _write_tape(tape_path, _mixed_records())

    run_source_layer_experiments(
        tape_path=tape_path,
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
