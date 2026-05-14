from __future__ import annotations

import json
from datetime import date
from urllib.error import HTTPError

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
        "results": [
            {
                "title": "Older",
                "summary": "s2",
                "url": "https://news.example/2",
                "publishedAt": "2026-05-01T10:00:00Z",
                "source": "SourceB",
            },
            {
                "title": "Newer",
                "summary": "s1",
                "url": "https://news.example/1",
                "publishedAt": "2026-05-19T09:00:00Z",
                "source": "SourceA",
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
        "results": [
            {
                "title": "Leak",
                "summary": "future",
                "url": "https://news.example/leak",
                "publishedAt": "2026-05-21T01:00:00Z",
                "source": "SourceX",
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
        return _FakeHTTPResponse({"results": []})

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


def test_asknews_web_search_passes_expected_date_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_urlopen(req):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"results": []})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    tool = AskNewsSearchTool(api_key="k", window_days=90)
    tool(_request())

    assert captured["body"]["publishedBefore"] == "2026-05-20"
    assert captured["body"]["publishedAfter"] == "2026-02-19"
    assert captured["body"]["query"] == "coalition stability italy"
