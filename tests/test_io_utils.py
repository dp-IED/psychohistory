from __future__ import annotations

import json
from pathlib import Path

from ingest.io_utils import open_text_auto, write_json_atomic, write_jsonl_records


def test_open_text_auto_reads_and_writes_plain_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    with open_text_auto(path, "w") as handle:
        handle.write('{"a":1}\n')

    with open_text_auto(path, "r") as handle:
        assert handle.read() == '{"a":1}\n'


def test_open_text_auto_reads_and_writes_gzip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"

    count = write_jsonl_records(path, [{"a": 1}, {"b": 2}])

    assert count == 2
    with open_text_auto(path, "r") as handle:
        assert [json.loads(line) for line in handle] == [{"a": 1}, {"b": 2}]


def test_write_json_atomic_supports_gzip(tmp_path: Path) -> None:
    path = tmp_path / "audit.json.gz"

    write_json_atomic(path, {"ok": True})

    with open_text_auto(path, "r") as handle:
        assert json.load(handle) == {"ok": True}
