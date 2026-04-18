"""Load normalized event records from the DuckDB warehouse."""

from __future__ import annotations

from pathlib import Path

from ingest.event_tape import EventTapeRecord
from ingest.event_warehouse import query_records
from ingest.paths import resolve_data_root, warehouse_path


def load_event_records(
    *,
    warehouse_db_path: Path | str | None = None,
    data_root: Path | str | None = None,
    source_names: set[str] | None = None,
) -> list[EventTapeRecord]:
    """Return all events from the warehouse at ``warehouse_db_path`` or the default path for ``data_root``."""
    db = (
        Path(warehouse_db_path).expanduser().resolve()
        if warehouse_db_path is not None
        else warehouse_path(resolve_data_root(data_root))
    )
    return query_records(db_path=db, source_names=source_names)
