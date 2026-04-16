"""Small IO helpers shared by ingestion and baseline runners."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, TextIO


def open_text_auto(path: Path, mode: str) -> TextIO:
    """Open plain text or gzip-compressed text based on the file suffix."""

    if "b" in mode:
        raise ValueError("open_text_auto only supports text modes")
    path = Path(path)
    if path.suffix == ".gz":
        gzip_mode = mode
        if "t" not in gzip_mode:
            gzip_mode = f"{gzip_mode}t"
        return gzip.open(path, gzip_mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically, using gzip when path ends in .gz."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes[-2:]) if path.suffix == ".gz" else path.suffix
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f"{suffix}.tmp",
        dir=str(path.parent),
        text=False,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open_text_auto(tmp_path if path.suffix != ".gz" else Path(f"{tmp_name}.gz"), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        actual_tmp = tmp_path if path.suffix != ".gz" else Path(f"{tmp_name}.gz")
        actual_tmp.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
        Path(f"{tmp_name}.gz").unlink(missing_ok=True)


def write_jsonl_records(path: Path, records: Iterable[Any]) -> int:
    """Write JSONL records, accepting dicts, pydantic models, or JSON strings."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text_auto(path, "w") as handle:
        for record in records:
            if isinstance(record, str):
                line = record
            elif hasattr(record, "model_dump_json"):
                line = record.model_dump_json()
            else:
                line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            handle.write(line.rstrip("\n") + "\n")
            count += 1
    return count
