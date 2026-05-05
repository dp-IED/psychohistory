from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ingest.acled_raw import ARAB_SPRING_ACLED_FIELDS, fetch_arab_spring_acled


def _sample_row(suffix: str) -> dict[str, str]:
    return {
        "event_id_cnty": f"EGY{suffix}",
        "event_date": "2011-01-05",
        "country": "Egypt",
        "admin1": "Cairo",
        "actor1": "Protesters",
        "actor2": "",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "fatalities": "0",
        "notes": "n",
    }


def test_fetch_arab_spring_acled_oauth_two_pages_then_empty_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACLED_EMAIL", "test@example.com")
    monkeypatch.setenv("ACLED_PASSWORD", "test-password")

    def fake_fetch(
        *,
        access_token: str,
        api_url: str,
        params: dict[str, str],
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> dict[str, object]:
        assert access_token == "fake-token"
        assert params["fields"] == ARAB_SPRING_ACLED_FIELDS
        assert params["country_where"] == "|"
        assert params["event_date_where"] == "BETWEEN"
        assert "email" not in params and "key" not in params
        page = int(params["page"])
        if page == 1:
            return {"data": [_sample_row("a"), _sample_row("b"), _sample_row("c"), _sample_row("d"), _sample_row("e")]}
        if page == 2:
            return {"data": [_sample_row("f"), _sample_row("g")]}
        if page == 3:
            return {"data": []}
        raise AssertionError(f"unexpected page={page}")

    out = tmp_path / "raw"
    with (
        patch("ingest.acled_raw.get_access_token", return_value="fake-token"),
        patch("ingest.acled_raw._fetch_page", side_effect=fake_fetch),
    ):
        result = fetch_arab_spring_acled(
            out_dir=out,
            event_start=dt.date(2011, 1, 1),
            event_end=dt.date(2011, 1, 7),
            limit=5000,
            max_pages=10,
            progress=False,
        )

    fragments = sorted(out.glob("acled_arab_spring_page_*.jsonl"))
    assert [p.name for p in fragments] == [
        "acled_arab_spring_page_0001.jsonl",
        "acled_arab_spring_page_0002.jsonl",
    ]
    lines_1 = (out / "acled_arab_spring_page_0001.jsonl").read_text(encoding="utf-8").strip().splitlines()
    lines_2 = (out / "acled_arab_spring_page_0002.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_1) == 5
    assert len(lines_2) == 2
    first = json.loads(lines_1[0])
    assert first["_api_page"] == 1

    manifest = json.loads((out / "fetch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows_written"] == 7
    assert manifest["files_fetched"] == 3
    assert manifest["fetch_completed_at"] is not None
    assert manifest["acled_api"] == "oauth_read"
    assert result["row_count"] == 7
    assert result["page_count"] == 3
