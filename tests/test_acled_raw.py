from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingest.acled_raw import (
    AcledPageResult,
    _build_request_url,
    _extract_rows,
    _extract_total_count,
    fetch_france_protests,
    iter_acled_pages,
    load_acled_fragments,
    normalize_acled_row,
    parse_acled_date,
)
import datetime as dt


# ---------------------------------------------------------------------------
# parse_acled_date
# ---------------------------------------------------------------------------


def test_parse_acled_date_iso_format() -> None:
    assert parse_acled_date("2021-01-05") == dt.date(2021, 1, 5)


def test_parse_acled_date_acled_format() -> None:
    assert parse_acled_date("05 January 2021") == dt.date(2021, 1, 5)


def test_parse_acled_date_rejects_garbage() -> None:
    with pytest.raises((ValueError, KeyError)):
        parse_acled_date("not-a-date")


# ---------------------------------------------------------------------------
# _build_request_url
# ---------------------------------------------------------------------------


def test_build_request_url_encodes_params() -> None:
    url = _build_request_url(
        api_key="KEY",
        email="test@example.com",
        event_start=dt.date(2021, 1, 1),
        event_end=dt.date(2021, 12, 31),
        page=1,
        page_size=500,
        base_url="https://api.acleddata.com/acled/read",
    )
    assert "key=KEY" in url
    assert "country=France" in url
    assert "event_type=Protests" in url
    assert "event_date=2021-01-01%7C2021-12-31" in url or "event_date=2021-01-01|2021-12-31" in url
    assert "page=1" in url
    assert "limit=500" in url


# ---------------------------------------------------------------------------
# _extract_rows / _extract_total_count
# ---------------------------------------------------------------------------


def test_extract_rows_returns_data_list() -> None:
    assert _extract_rows({"data": [{"a": 1}], "count": 1}) == [{"a": 1}]


def test_extract_rows_raises_on_missing_key() -> None:
    with pytest.raises(ValueError, match="missing 'data' key"):
        _extract_rows({"count": 0})


def test_extract_rows_raises_on_non_list() -> None:
    with pytest.raises(ValueError, match="not a list"):
        _extract_rows({"data": "oops"})


def test_extract_total_count_zero_when_missing() -> None:
    assert _extract_total_count({}) == 0


def test_extract_total_count_parses_int() -> None:
    assert _extract_total_count({"count": "42"}) == 42


# ---------------------------------------------------------------------------
# normalize_acled_row
# ---------------------------------------------------------------------------


def _acled_row(**overrides: str) -> dict[str, str]:
    row: dict[str, str] = {
        "data_id": "1",
        "iso": "250",
        "event_id_cnty": "FRA1",
        "event_date": "2021-01-05",
        "year": "2021",
        "time_precision": "1",
        "disorder_type": "Political violence",
        "event_type": "Protests",
        "sub_event_type": "Peaceful protest",
        "actor1": "Protesters (France)",
        "actor2": "",
        "country": "France",
        "region": "Western Europe",
        "admin1": "Île-de-France",
        "admin2": "Paris",
        "admin3": "",
        "location": "Paris",
        "latitude": "48.8566",
        "longitude": "2.3522",
        "geo_precision": "1",
        "source": "AFP",
        "source_scale": "International",
        "notes": "Protest outside the National Assembly.",
        "fatalities": "0",
        "timestamp": "1609891200",
        "iso3": "FRA",
    }
    row.update(overrides)
    return row


def test_normalize_acled_row_accepts_valid_row() -> None:
    result = normalize_acled_row(_acled_row())
    assert result is not None
    assert result["country"] == "France"


def test_normalize_acled_row_rejects_wrong_country() -> None:
    assert normalize_acled_row(_acled_row(country="Germany")) is None


def test_normalize_acled_row_rejects_wrong_event_type() -> None:
    assert normalize_acled_row(_acled_row(event_type="Battles")) is None


def test_normalize_acled_row_rejects_missing_date() -> None:
    assert normalize_acled_row(_acled_row(event_date="")) is None


def test_normalize_acled_row_rejects_unparseable_date() -> None:
    assert normalize_acled_row(_acled_row(event_date="not-a-date")) is None


def test_normalize_acled_row_case_insensitive_country() -> None:
    assert normalize_acled_row(_acled_row(country="FRANCE")) is not None


def test_normalize_acled_row_case_insensitive_event_type() -> None:
    assert normalize_acled_row(_acled_row(event_type="PROTESTS")) is not None


# ---------------------------------------------------------------------------
# iter_acled_pages (mocked HTTP)
# ---------------------------------------------------------------------------


def _mock_response(rows: list[dict], total: int) -> dict:
    return {"data": rows, "count": str(total), "success": True}


def test_iter_acled_pages_single_page() -> None:
    row = _acled_row()
    responses = [_mock_response([row], total=1)]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        pages = list(
            iter_acled_pages(
                api_key="K",
                email="e@example.com",
                event_start=dt.date(2021, 1, 1),
                event_end=dt.date(2021, 12, 31),
            )
        )

    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].rows == [row]
    assert pages[0].total_count == 1


def test_iter_acled_pages_multiple_pages() -> None:
    row_a = _acled_row(data_id="1")
    row_b = _acled_row(data_id="2")
    responses = [
        _mock_response([row_a], total=2),
        _mock_response([row_b], total=2),
    ]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        pages = list(
            iter_acled_pages(
                api_key="K",
                email="e@example.com",
                event_start=dt.date(2021, 1, 1),
                event_end=dt.date(2021, 12, 31),
                page_size=1,
            )
        )

    assert len(pages) == 2
    assert pages[1].page == 2
    assert pages[1].rows == [row_b]


def test_iter_acled_pages_stops_on_empty_data() -> None:
    responses = [_mock_response([], total=0)]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        pages = list(
            iter_acled_pages(
                api_key="K",
                email="e@example.com",
                event_start=dt.date(2021, 1, 1),
                event_end=dt.date(2021, 12, 31),
            )
        )

    assert pages == []


# ---------------------------------------------------------------------------
# fetch_france_protests (integration: writes fragments + manifest)
# ---------------------------------------------------------------------------


def test_fetch_writes_fragment_and_manifest(tmp_path: Path) -> None:
    row = _acled_row()
    responses = [_mock_response([row], total=1)]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        metadata = fetch_france_protests(
            api_key="K",
            email="e@example.com",
            event_start=dt.date(2021, 1, 1),
            event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
        )

    assert metadata["status"] == "ok"
    assert metadata["total_rows"] == 1

    manifest_path = tmp_path / "fetch_manifest.jsonl"
    assert manifest_path.exists()
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["status"] == "ok"

    fragment = tmp_path / manifest_rows[0]["fragment_path"]
    assert fragment.exists()
    written_rows = [json.loads(line) for line in fragment.read_text().splitlines() if line.strip()]
    assert len(written_rows) == 1
    assert written_rows[0]["_source_name"] == "acled_api_v1"
    assert "_retrieved_at" in written_rows[0]
    assert "_run_id" in written_rows[0]


def test_fetch_skips_when_completed_run_exists(tmp_path: Path) -> None:
    """Second call with no --force should not issue any HTTP requests."""
    row = _acled_row()
    responses = [_mock_response([row], total=1)]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        fetch_france_protests(
            api_key="K",
            email="e@example.com",
            event_start=dt.date(2021, 1, 1),
            event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
        )

    with patch("ingest.acled_raw._fetch_page", side_effect=Exception("should not be called")) as mock_fetch:
        fetch_france_protests(
            api_key="K",
            email="e@example.com",
            event_start=dt.date(2021, 1, 1),
            event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
        )
        mock_fetch.assert_not_called()


def test_fetch_force_refetches(tmp_path: Path) -> None:
    row = _acled_row()
    responses = [_mock_response([row], total=1), _mock_response([row], total=1)]

    call_count = 0

    def fake_fetch(url: str, *, max_retries: int, retry_backoff_seconds: float) -> dict:
        nonlocal call_count
        call_count += 1
        return responses[call_count - 1]

    with patch("ingest.acled_raw._fetch_page", side_effect=fake_fetch):
        fetch_france_protests(
            api_key="K", email="e@example.com",
            event_start=dt.date(2021, 1, 1), event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
        )
        fetch_france_protests(
            api_key="K", email="e@example.com",
            event_start=dt.date(2021, 1, 1), event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
            force=True,
        )

    assert call_count == 2


def test_fetch_raises_on_api_error(tmp_path: Path) -> None:
    with patch("ingest.acled_raw._fetch_page", side_effect=OSError("network error")):
        with pytest.raises(RuntimeError, match="ACLED fetch failed"):
            fetch_france_protests(
                api_key="K",
                email="e@example.com",
                event_start=dt.date(2021, 1, 1),
                event_end=dt.date(2021, 12, 31),
                out_dir=tmp_path,
            )

    manifest_path = tmp_path / "fetch_manifest.jsonl"
    assert manifest_path.exists()
    manifest_row = json.loads(manifest_path.read_text().strip())
    assert manifest_row["status"] == "failed"
    assert "network error" in manifest_row["error"]


# ---------------------------------------------------------------------------
# load_acled_fragments
# ---------------------------------------------------------------------------


def test_load_acled_fragments_returns_rows(tmp_path: Path) -> None:
    row = _acled_row()
    responses = [_mock_response([row], total=1)]

    with patch("ingest.acled_raw._fetch_page", side_effect=responses):
        fetch_france_protests(
            api_key="K",
            email="e@example.com",
            event_start=dt.date(2021, 1, 1),
            event_end=dt.date(2021, 12, 31),
            out_dir=tmp_path,
        )

    loaded = load_acled_fragments(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["country"] == "France"


def test_load_acled_fragments_raises_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing ACLED fetch manifest"):
        load_acled_fragments(tmp_path)


def test_load_acled_fragments_raises_missing_metadata(tmp_path: Path) -> None:
    (tmp_path / "fetch_manifest.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="missing ACLED fetch metadata"):
        load_acled_fragments(tmp_path)
