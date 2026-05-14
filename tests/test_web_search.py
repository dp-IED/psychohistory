from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from harness.query_mapper import PITViolationError, WebSearchRequest
from harness.tools.web_search import AskNewsAPIError, AskNewsSearchTool


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _request() -> WebSearchRequest:
    return WebSearchRequest(
        query="coalition stability italy",
        as_of_date=date(2026, 5, 20),
        market_family="metaculus_binary",
        blind_spot_check="coalition_stability_check",
    )


def test_asknews_web_search_returns_tool_call_records_sorted_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "as_dicts": [
            {
                "eng_title": "Older",
                "summary": "s2",
                "article_url": "https://news.example/2",
                "pub_date": "2026-05-01T10:00:00Z",
                "source_id": "SourceB",
            },
            {
                "eng_title": "Newer",
                "summary": "s1",
                "article_url": "https://news.example/1",
                "pub_date": "2026-05-19T09:00:00Z",
                "source_id": "SourceA",
            },
        ]
    }

    def _fake_urlopen(_req):
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k", max_results=10)
    out = tool(_request())

    assert len(out) == 1
    assert out[0].tool_name == "web_search"
    assert out[0].evidence_count == 2
    assert "2026-05-19" in out[0].notes
    assert "https://news.example/1" in out[0].notes
    assert out[0].notes.index("2026-05-19") < out[0].notes.index("2026-05-01")


def test_asknews_web_search_raises_on_post_cutoff_result(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "as_dicts": [
            {
                "eng_title": "Leak",
                "summary": "future",
                "article_url": "https://news.example/leak",
                "pub_date": "2026-05-21T01:00:00Z",
                "source_id": "SourceX",
            }
        ]
    }

    def _fake_urlopen(_req):
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k")
    with pytest.raises(PITViolationError, match="post-cutoff"):
        tool(_request())


def test_asknews_web_search_returns_empty_list_for_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(_req):
        return _FakeHTTPResponse({"as_dicts": []})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k")
    out = tool(_request())
    assert out == []


def test_asknews_web_search_raises_domain_error_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(req):
        raise HTTPError(req.full_url, 500, "boom", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k")
    with pytest.raises(AskNewsAPIError, match="AskNews API error"):
        tool(_request())


def test_asknews_web_search_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    payload = {"as_dicts": []}

    def _fake_urlopen(req):
        calls["n"] += 1
        if calls["n"] < 3:
            raise HTTPError(req.full_url, 429, "rate limit", hdrs=None, fp=None)
        return _FakeHTTPResponse(payload)

    sleeps: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("harness.tools.web_search.time.sleep", _fake_sleep)

    tool = AskNewsSearchTool(api_key="k")
    out = tool(_request())

    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert out == []


def _utc_day_bounds(d_after: date, d_before: date) -> tuple[int, int]:
    start_dt = datetime(d_after.year, d_after.month, d_after.day, tzinfo=timezone.utc)
    end_dt = datetime(
        d_before.year,
        d_before.month,
        d_before.day,
        tzinfo=timezone.utc,
    ) + timedelta(days=1) - timedelta(seconds=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def test_asknews_web_search_passes_expected_date_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_urlopen(req):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["qs"] = parse_qs(urlparse(req.full_url).query)
        assert req.method == "GET"
        assert req.data is None
        return _FakeHTTPResponse({"as_dicts": []})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k", window_days=90)
    tool(_request())

    req = _request()
    published_after = req.as_of_date - timedelta(days=90)
    expected_start, expected_end = _utc_day_bounds(published_after, req.as_of_date)

    qs = captured["qs"]
    assert qs["start_timestamp"] == [str(expected_start)]
    assert qs["end_timestamp"] == [str(expected_end)]
    assert qs["historical"] == ["true"]
    assert qs["query"] == ["coalition stability italy"]
    assert qs["method"] == ["kw"]
    assert qs["return_type"] == ["dicts"]
    assert qs["n_articles"] == ["10"]

    assert captured["headers"].get("Authorization") == "Bearer k"
    path = urlparse(captured["url"]).path.rstrip("/")
    assert path.endswith("/v1/news/search")


def test_asknews_async_search_raw_runs_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    enters: list[float] = []

    def fake_search_raw(self: AskNewsSearchTool, _req):  # noqa: ARG001
        enters.append(time.perf_counter())
        time.sleep(0.06)
        return []

    monkeypatch.setattr(AskNewsSearchTool, "search_raw", fake_search_raw)

    tool = AskNewsSearchTool(api_key="k")
    reqs = [_request() for _ in range(3)]

    async def gather_all():
        out = await asyncio.gather(*(tool.async_search_raw(r) for r in reqs))
        return out

    wall0 = time.perf_counter()
    batches = asyncio.run(gather_all())
    wall = time.perf_counter() - wall0

    assert len(batches) == 3
    assert all(b == [] for b in batches)
    assert len(enters) == 3
    assert max(enters) - min(enters) < 0.05
    assert wall < 0.2
