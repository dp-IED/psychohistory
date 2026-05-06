from __future__ import annotations

from pathlib import Path

import pytest

from ingest.universal_event_staging import (
    UniversalEventRecord,
    build_staging_row,
    init_universal_event_staging,
    upsert_universal_event_rows,
    validate_universal_event_staging_schema,
)


def _valid_record() -> UniversalEventRecord:
    return UniversalEventRecord.model_validate(
        {
            "subject_id": "printing_press",
            "subject_name": "Printing press",
            "subject_type": "PHENOMENON",
            "object_id": "urban_printers",
            "object_name": "Urban printers",
            "object_type": "GROUP",
            "relation_type": "INFLUENCES",
            "relation_description": "Printing press influences urban printers.",
            "braudel_layer": "longue_duree",
            "structural_mechanism": (
                "Lower copying costs and reproducible type changed the economics of text circulation, "
                "letting workshops scale production, coordinate with merchants, and weaken older manuscript "
                "bottlenecks that had concentrated information in clerical or courtly institutions."
            ),
            "date_start": "1450-01-01",
            "date_end": "1500-12-31",
            "date_precision": "CENTURY",
            "location_type": "NETWORK",
            "geo_country": None,
            "geo_admin1": None,
            "concept_domain": "print_network",
            "description": (
                "The printing press influenced urban printers by lowering copying costs and making "
                "large-scale text circulation commercially viable across connected workshops."
            ),
            "outcome": "Commercial print networks expanded.",
            "outcome_date": "1500-12-31",
            "lag_years": 50,
            "lag_precision": "GENERATIONAL",
            "source_confidence": 0.8,
        }
    )


def _valid_row() -> dict[str, object]:
    return build_staging_row(
        record=_valid_record(),
        model_id="Qwen/Qwen3.5-397B-A17B-FP8-dottxt",
        source_url="https://en.wikipedia.org/wiki/Printing_press",
        revision_id="12345",
        fetched_at="2026-04-27T00:00:00Z",
        batch_id="batch-1",
        custom_id="wiki-printing-press-397-001",
        article_id="printing-press",
    )


def test_validate_universal_event_staging_schema_accepts_initialized_table(tmp_path: Path) -> None:
    db_path = tmp_path / "staging.duckdb"
    init_universal_event_staging(db_path)

    summary = validate_universal_event_staging_schema(db_path=db_path)

    assert summary["table"] == "universal_event_staging"
    assert summary["column_count"] >= 35
    assert summary["primary_key"] == ["event_id", "model_id"]
    assert summary["embedding_status_default"] == "pending"


def test_validate_universal_event_staging_schema_rejects_incomplete_table(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "bad.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE universal_event_staging(event_id TEXT NOT NULL, model_id TEXT NOT NULL)")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_universal_event_staging_schema(db_path=db_path)


def test_upsert_universal_event_rows_rejects_rows_before_duckdb_coercion(tmp_path: Path) -> None:
    db_path = tmp_path / "staging.duckdb"
    row = _valid_row()
    row.pop("structural_mechanism")

    with pytest.raises(ValueError, match="row 0 missing required staging columns"):
        upsert_universal_event_rows(db_path=db_path, rows=[row])


def test_upsert_universal_event_rows_rejects_unknown_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "staging.duckdb"
    row = _valid_row()
    row["unexpected_payload"] = "silent drift"

    with pytest.raises(ValueError, match="row 0 has unexpected staging columns"):
        upsert_universal_event_rows(db_path=db_path, rows=[row])


def test_upsert_universal_event_rows_is_idempotent_for_same_event_model(tmp_path: Path) -> None:
    db_path = tmp_path / "staging.duckdb"
    row = _valid_row()

    first = upsert_universal_event_rows(db_path=db_path, rows=[row])
    second = upsert_universal_event_rows(db_path=db_path, rows=[row])

    assert first["input_count"] == 1
    assert first["upserted_count"] == 1
    assert first["total_row_count"] == 1
    assert second["input_count"] == 1
    assert second["upserted_count"] == 1
    assert second["total_row_count"] == 1
