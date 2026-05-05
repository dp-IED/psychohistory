"""DuckDB staging for Wikipedia-derived UniversalEventRecord rows."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, model_validator


UTC = dt.timezone.utc

ACTOR_TYPES = (
    "PERSON",
    "GROUP",
    "INSTITUTION",
    "MOVEMENT",
    "IDEA",
    "PHENOMENON",
    "PLACE",
)

BRAUDEL_LAYERS = ("evenements", "conjonctures", "longue_duree")
EMBEDDING_TEXT_FIELD = "structural_mechanism"

RELATION_TYPES = (
    "CREATES",
    "DESTROYS",
    "INFLUENCES",
    "SUPPRESSES",
    "COOPERATES",
    "OPPOSES",
    "ADOPTS",
    "SPREADS_TO",
    "TRANSFORMS",
    "PRECEDES",
    "FUNDS",
    "LEGITIMISES",
)

DATE_PRECISIONS = ("DAY", "MONTH", "YEAR", "DECADE", "CENTURY", "ESTIMATED")

LOCATION_TYPES = ("GEOGRAPHIC", "CONCEPTUAL", "NETWORK", "UNKNOWN")

LAG_PRECISIONS = (
    "SYNCHRONOUS",
    "SHORT",
    "MEDIUM",
    "LONG",
    "GENERATIONAL",
    "EPOCHAL",
    "UNKNOWN",
)

REQUIRED_FIELDS = (
    "subject_id",
    "subject_name",
    "subject_type",
    "object_id",
    "object_name",
    "object_type",
    "relation_type",
    "relation_description",
    "braudel_layer",
    "structural_mechanism",
    "lag_years",
    "date_precision",
    "location_type",
    "description",
    "source_confidence",
    "lag_precision",
)

STAGING_INSERT_COLUMNS = (
    "event_id",
    "model_id",
    "source",
    "source_url",
    "revision_id",
    "fetched_at",
    "batch_id",
    "custom_id",
    "article_id",
    "embedding_status",
    "subject_id",
    "subject_name",
    "subject_type",
    "object_id",
    "object_name",
    "object_type",
    "relation_type",
    "relation_description",
    "braudel_layer",
    "structural_mechanism",
    "date_start",
    "date_end",
    "date_precision",
    "location_type",
    "geo_country",
    "geo_admin1",
    "concept_domain",
    "description",
    "outcome",
    "outcome_date",
    "lag_years",
    "lag_precision",
    "source_confidence",
    "raw_json",
)

STAGING_SCHEMA_COLUMNS = {
    "event_id": {"type": "VARCHAR", "not_null": True, "pk": True},
    "model_id": {"type": "VARCHAR", "not_null": True, "pk": True},
    "source": {"type": "VARCHAR", "not_null": True},
    "source_url": {"type": "VARCHAR", "not_null": False},
    "revision_id": {"type": "VARCHAR", "not_null": False},
    "fetched_at": {"type": "TIMESTAMP WITH TIME ZONE", "not_null": False},
    "batch_id": {"type": "VARCHAR", "not_null": False},
    "custom_id": {"type": "VARCHAR", "not_null": False},
    "article_id": {"type": "VARCHAR", "not_null": False},
    "embedding_status": {"type": "VARCHAR", "not_null": True},
    "subject_id": {"type": "VARCHAR", "not_null": True},
    "subject_name": {"type": "VARCHAR", "not_null": True},
    "subject_type": {"type": "VARCHAR", "not_null": True},
    "object_id": {"type": "VARCHAR", "not_null": False},
    "object_name": {"type": "VARCHAR", "not_null": False},
    "object_type": {"type": "VARCHAR", "not_null": False},
    "relation_type": {"type": "VARCHAR", "not_null": True},
    "relation_description": {"type": "VARCHAR", "not_null": True},
    "braudel_layer": {"type": "VARCHAR", "not_null": True},
    "structural_mechanism": {"type": "VARCHAR", "not_null": True},
    "date_start": {"type": "DATE", "not_null": False},
    "date_end": {"type": "DATE", "not_null": False},
    "date_precision": {"type": "VARCHAR", "not_null": True},
    "location_type": {"type": "VARCHAR", "not_null": True},
    "geo_country": {"type": "VARCHAR", "not_null": False},
    "geo_admin1": {"type": "VARCHAR", "not_null": False},
    "concept_domain": {"type": "VARCHAR", "not_null": False},
    "description": {"type": "VARCHAR", "not_null": True},
    "outcome": {"type": "VARCHAR", "not_null": False},
    "outcome_date": {"type": "DATE", "not_null": False},
    "lag_years": {"type": "INTEGER", "not_null": False},
    "lag_precision": {"type": "VARCHAR", "not_null": True},
    "source_confidence": {"type": "DOUBLE", "not_null": True},
    "raw_json": {"type": "VARCHAR", "not_null": True},
    "created_at": {"type": "TIMESTAMP WITH TIME ZONE", "not_null": True},
}

LAG_PRECISION_INDEX = {name: i for i, name in enumerate(LAG_PRECISIONS)}
DESCRIPTION_MIN_CHARS = 90
DESCRIPTION_MAX_CHARS = 500
STRUCTURAL_MECHANISM_MIN_CHARS = 160
STRUCTURAL_MECHANISM_MAX_CHARS = 1000


class UniversalEventRecord(BaseModel):
    """Model-level representation of an extracted event relationship."""

    subject_id: str
    subject_name: str
    subject_type: str
    object_id: str
    object_name: str
    object_type: str
    relation_type: str
    relation_description: str
    braudel_layer: str = "conjonctures"
    structural_mechanism: str = ""
    date_start: str | None = None
    date_end: str | None = None
    date_precision: str
    location_type: str
    geo_country: str | None = None
    geo_admin1: str | None = None
    concept_domain: str | None = None
    description: str
    outcome: str | None = None
    outcome_date: str | None = None
    lag_years: int | None = None
    lag_precision: str
    source_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_enums(self) -> "UniversalEventRecord":
        if self.subject_type not in ACTOR_TYPES:
            raise ValueError(f"invalid subject_type: {self.subject_type}")
        if self.object_type not in ACTOR_TYPES:
            raise ValueError(f"invalid object_type: {self.object_type}")
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"invalid relation_type: {self.relation_type}")
        if self.braudel_layer not in BRAUDEL_LAYERS:
            raise ValueError(f"invalid braudel_layer: {self.braudel_layer}")
        if self.date_precision not in DATE_PRECISIONS:
            raise ValueError(f"invalid date_precision: {self.date_precision}")
        if self.location_type not in LOCATION_TYPES:
            raise ValueError(f"invalid location_type: {self.location_type}")
        if self.lag_precision not in LAG_PRECISIONS:
            raise ValueError(f"invalid lag_precision: {self.lag_precision}")

        if self.date_start is not None and self.date_end is not None:
            if self.date_start > self.date_end:
                raise ValueError("date_start must be <= date_end")

        if self.lag_precision not in {"SYNCHRONOUS", "UNKNOWN"} and self.lag_years is None:
            raise ValueError("lag_years is required when lag_precision is not SYNCHRONOUS or UNKNOWN")
        if self.location_type in {"CONCEPTUAL", "NETWORK"} and not (self.concept_domain or "").strip():
            raise ValueError("concept_domain is required when location_type is CONCEPTUAL or NETWORK")
        if not self.object_id.strip() or not self.object_name.strip():
            raise ValueError("object_id and object_name are required")

        desc_len = len(self.description.strip())
        if desc_len < DESCRIPTION_MIN_CHARS:
            raise ValueError(
                f"description too short; expected at least {DESCRIPTION_MIN_CHARS} characters"
            )
        if desc_len > DESCRIPTION_MAX_CHARS:
            self.description = self.description[:DESCRIPTION_MAX_CHARS]

        rel_len = len(self.relation_description.strip())
        if rel_len == 0:
            raise ValueError("relation_description must not be blank")
        mechanism_len = len(self.structural_mechanism.strip())
        if mechanism_len < STRUCTURAL_MECHANISM_MIN_CHARS:
            raise ValueError(
                "structural_mechanism must explain the causal pathway; "
                f"expected at least {STRUCTURAL_MECHANISM_MIN_CHARS} characters"
            )
        if mechanism_len > STRUCTURAL_MECHANISM_MAX_CHARS:
            self.structural_mechanism = self.structural_mechanism[:STRUCTURAL_MECHANISM_MAX_CHARS]

        self.subject_id = normalize_subject_id(self.subject_id)
        self.object_id = normalize_subject_id(self.object_id)
        self.source_confidence = min(1.0, max(0.0, float(self.source_confidence)))
        return self


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required; install project dependencies first") from exc
    return duckdb


def normalize_subject_id(value: str) -> str:
    """Normalize ids for stable downstream joins."""

    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s_]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unknown_subject"


def event_id_for(source_url: str, description: str) -> str:
    payload = f"{source_url}|{description}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def embedding_text_for_record(record: UniversalEventRecord) -> str:
    """Locked text source for downstream 128-d UniversalEventRecord embeddings."""

    return record.structural_mechanism


def init_universal_event_staging(db_path: Path | str) -> None:
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with _duckdb().connect(str(db)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS universal_event_staging (
                event_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'wikipedia',
                source_url TEXT,
                revision_id TEXT,
                fetched_at TIMESTAMPTZ,
                batch_id TEXT,
                custom_id TEXT,
                article_id TEXT,
                embedding_status TEXT NOT NULL DEFAULT 'pending',
                subject_id TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                object_id TEXT,
                object_name TEXT,
                object_type TEXT,
                relation_type TEXT NOT NULL,
                relation_description TEXT NOT NULL,
                braudel_layer TEXT NOT NULL DEFAULT 'conjonctures',
                structural_mechanism TEXT NOT NULL DEFAULT '',
                date_start DATE,
                date_end DATE,
                date_precision TEXT NOT NULL,
                location_type TEXT NOT NULL,
                geo_country TEXT,
                geo_admin1 TEXT,
                concept_domain TEXT,
                description TEXT NOT NULL,
                outcome TEXT,
                outcome_date DATE,
                lag_years INTEGER,
                lag_precision TEXT NOT NULL,
                source_confidence DOUBLE NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (event_id, model_id)
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_universal_event_article ON universal_event_staging(article_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_universal_event_model ON universal_event_staging(model_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_universal_event_embedding_status ON universal_event_staging(embedding_status)"
        )
        existing = {
            str(row[1])
            for row in con.execute("PRAGMA table_info('universal_event_staging')").fetchall()
        }
        if "braudel_layer" not in existing:
            con.execute(
                "ALTER TABLE universal_event_staging "
                "ADD COLUMN braudel_layer TEXT NOT NULL DEFAULT 'conjonctures'"
            )
        if "structural_mechanism" not in existing:
            con.execute(
                "ALTER TABLE universal_event_staging "
                "ADD COLUMN structural_mechanism TEXT NOT NULL DEFAULT ''"
            )


def _normalize_duckdb_default(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def validate_universal_event_staging_schema(*, db_path: Path | str) -> dict[str, Any]:
    """Validate that the DuckDB staging table matches the locked UniversalEvent contract."""

    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"missing staging db: {db}")

    with _duckdb().connect(str(db), read_only=True) as con:
        rows = con.execute("PRAGMA table_info('universal_event_staging')").fetchall()
    if not rows:
        raise ValueError("missing universal_event_staging table")

    observed = {
        str(row[1]): {
            "ordinal": int(row[0]),
            "type": str(row[2]).upper(),
            "not_null": bool(row[3]),
            "default": _normalize_duckdb_default(row[4]),
            "pk": bool(row[5]),
        }
        for row in rows
    }
    expected_names = set(STAGING_SCHEMA_COLUMNS)
    observed_names = set(observed)
    missing = sorted(expected_names - observed_names)
    if missing:
        raise ValueError(f"missing required columns in universal_event_staging: {missing}")

    mismatches: list[str] = []
    for column, expected in STAGING_SCHEMA_COLUMNS.items():
        actual = observed[column]
        if actual["type"] != expected["type"]:
            mismatches.append(
                f"{column} type expected {expected['type']}, got {actual['type']}"
            )
        if actual["not_null"] != expected["not_null"]:
            mismatches.append(
                f"{column} not_null expected {expected['not_null']}, got {actual['not_null']}"
            )
        expected_pk = bool(expected.get("pk", False))
        if actual["pk"] != expected_pk:
            mismatches.append(f"{column} pk expected {expected_pk}, got {actual['pk']}")
    if mismatches:
        raise ValueError("universal_event_staging schema mismatch: " + "; ".join(mismatches))

    return {
        "table": "universal_event_staging",
        "column_count": len(observed),
        "primary_key": [
            name
            for name, meta in sorted(observed.items(), key=lambda item: item[1]["ordinal"])
            if meta["pk"]
        ],
        "embedding_status_default": observed["embedding_status"]["default"],
        "created_at_default": observed["created_at"]["default"],
    }


def _ensure_aware(value: str | dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_staging_row(
    *,
    record: UniversalEventRecord,
    model_id: str,
    source_url: str,
    revision_id: str | None,
    fetched_at: str | dt.datetime | None,
    batch_id: str | None,
    custom_id: str,
    article_id: str,
    source: str = "wikipedia",
) -> dict[str, Any]:
    event_id = event_id_for(source_url=source_url, description=record.description)
    return {
        "event_id": event_id,
        "model_id": model_id,
        "source": source,
        "source_url": source_url,
        "revision_id": revision_id,
        "fetched_at": _ensure_aware(fetched_at),
        "batch_id": batch_id,
        "custom_id": custom_id,
        "article_id": article_id,
        "embedding_status": "pending",
        "subject_id": record.subject_id,
        "subject_name": record.subject_name,
        "subject_type": record.subject_type,
        "object_id": record.object_id,
        "object_name": record.object_name,
        "object_type": record.object_type,
        "relation_type": record.relation_type,
        "relation_description": record.relation_description,
        "braudel_layer": record.braudel_layer,
        "structural_mechanism": record.structural_mechanism,
        "date_start": record.date_start,
        "date_end": record.date_end,
        "date_precision": record.date_precision,
        "location_type": record.location_type,
        "geo_country": record.geo_country,
        "geo_admin1": record.geo_admin1,
        "concept_domain": record.concept_domain,
        "description": record.description,
        "outcome": record.outcome,
        "outcome_date": record.outcome_date,
        "lag_years": record.lag_years,
        "lag_precision": record.lag_precision,
        "source_confidence": float(record.source_confidence),
        "raw_json": json.dumps(record.model_dump(mode="json"), sort_keys=True),
    }


def _validate_staging_rows(rows: list[dict[str, Any]]) -> None:
    expected = set(STAGING_INSERT_COLUMNS)
    for index, row in enumerate(rows):
        observed = set(row)
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(f"row {index} missing required staging columns: {missing}")
        unexpected = sorted(observed - expected)
        if unexpected:
            raise ValueError(f"row {index} has unexpected staging columns: {unexpected}")


def upsert_universal_event_rows(
    *,
    db_path: Path | str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    rows_list = list(rows)
    init_universal_event_staging(db_path)
    validate_universal_event_staging_schema(db_path=db_path)
    if not rows_list:
        return {"upserted_count": 0}
    _validate_staging_rows(rows_list)

    columns = list(STAGING_INSERT_COLUMNS)
    placeholders = ", ".join(["?"] * len(columns))
    updates = ", ".join(
        f"{column}=excluded.{column}" for column in columns if column not in {"event_id", "model_id"}
    )
    sql = (
        f"INSERT INTO universal_event_staging ({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        "ON CONFLICT (event_id, model_id) "
        f"DO UPDATE SET {updates}"
    )
    values = [[row.get(column) for column in columns] for row in rows_list]
    with _duckdb().connect(str(db_path)) as con:
        con.executemany(sql, values)
    return {"upserted_count": len(rows_list)}


def load_predictions(
    *,
    db_path: Path | str,
    model_id: str | None = None,
    article_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"missing staging db: {db}")

    where = ["1=1"]
    params: list[Any] = []
    if model_id is not None:
        where.append("model_id = ?")
        params.append(model_id)
    if article_ids:
        ordered = sorted(article_ids)
        where.append(f"article_id IN ({', '.join(['?'] * len(ordered))})")
        params.extend(ordered)

    sql = (
        "SELECT article_id, model_id, subject_id, subject_name, subject_type, "
        "object_id, object_name, object_type, relation_type, relation_description, "
        "braudel_layer, structural_mechanism, date_start, date_end, date_precision, location_type, geo_country, geo_admin1, "
        "concept_domain, description, outcome, outcome_date, lag_years, lag_precision, "
        "source_confidence "
        "FROM universal_event_staging "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY article_id, description"
    )
    with _duckdb().connect(str(db), read_only=True) as con:
        rows = con.execute(sql, params).fetchall()

    output: list[dict[str, Any]] = []
    for row in rows:
        payload = {
            "article_id": row[0],
            "model_id": row[1],
            "subject_id": row[2],
            "subject_name": row[3],
            "subject_type": row[4],
            "object_id": row[5],
            "object_name": row[6],
            "object_type": row[7],
            "relation_type": row[8],
            "relation_description": row[9],
            "braudel_layer": row[10],
            "structural_mechanism": row[11],
            "date_start": row[12].isoformat() if row[12] is not None else None,
            "date_end": row[13].isoformat() if row[13] is not None else None,
            "date_precision": row[14],
            "location_type": row[15],
            "geo_country": row[16],
            "geo_admin1": row[17],
            "concept_domain": row[18],
            "description": row[19],
            "outcome": row[20],
            "outcome_date": row[21].isoformat() if row[21] is not None else None,
            "lag_years": row[22],
            "lag_precision": row[23],
            "source_confidence": float(row[24]),
        }
        output.append(payload)
    return output


def distinct_model_ids(*, db_path: Path | str) -> list[str]:
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"missing staging db: {db}")
    with _duckdb().connect(str(db), read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT model_id FROM universal_event_staging ORDER BY model_id"
        ).fetchall()
    return [str(row[0]) for row in rows]


__all__ = [
    "ACTOR_TYPES",
    "DATE_PRECISIONS",
    "LAG_PRECISIONS",
    "LOCATION_TYPES",
    "RELATION_TYPES",
    "REQUIRED_FIELDS",
    "STAGING_INSERT_COLUMNS",
    "STAGING_SCHEMA_COLUMNS",
    "UniversalEventRecord",
    "build_staging_row",
    "distinct_model_ids",
    "event_id_for",
    "init_universal_event_staging",
    "load_predictions",
    "normalize_subject_id",
    "upsert_universal_event_rows",
    "validate_universal_event_staging_schema",
]
