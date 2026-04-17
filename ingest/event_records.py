"""Load normalized event records from JSONL tape files or the DuckDB warehouse."""

from __future__ import annotations

from pathlib import Path

from ingest.event_tape import EventTapeRecord, load_event_tape
from ingest.event_warehouse import query_records
from ingest.paths import resolve_data_root, warehouse_path


def load_event_records(
    *,
    tape_path: Path | str | None = None,
    warehouse_db_path: Path | str | None = None,
    data_root: Path | str | None = None,
    source_names: set[str] | None = None,
) -> list[EventTapeRecord]:
    """Return event records from ``tape_path`` when set; otherwise query the warehouse.

    The warehouse path is ``warehouse_db_path`` if provided, else
    ``warehouse_path(resolve_data_root(data_root))``.
    """
    if tape_path is not None:
        return load_event_tape(Path(tape_path))
    db = (
        Path(warehouse_db_path).expanduser().resolve()
        if warehouse_db_path is not None
        else warehouse_path(resolve_data_root(data_root))
    )
    return query_records(db_path=db, source_names=source_names)
