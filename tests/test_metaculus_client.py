from __future__ import annotations

import io
import json
from datetime import date
from urllib.error import HTTPError

import pytest

from harness.metaculus_client import MetaculusAPIError, MetaculusClient, MetaculusQuestion


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_post_forecast_rejects_degenerate_probabilities_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MetaculusClient(api_token="token")

    called = {"count": 0}

    def _boom(_request):
        called["count"] += 1
        raise AssertionError("network should not be called")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    with pytest.raises(ValueError, match="p_yes"):
        client.post_forecast(question_id=123, p_yes=0.0, comment="reason")
    with pytest.raises(ValueError, match="p_yes"):
        client.post_forecast(question_id=123, p_yes=1.0, comment="reason")

    assert called["count"] == 0


def test_post_forecast_raises_domain_error_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MetaculusClient(api_token="token")

    def _raise(_request):
        raise HTTPError(
            url="https://www.metaculus.com/api/v2/questions/1/forecast/",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"detail": "bad token"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(MetaculusAPIError, match="401"):
        client.post_forecast(question_id=1, p_yes=0.51, comment="reason")


def test_post_forecast_raises_domain_error_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MetaculusClient(api_token="token")

    def _raise(_request):
        raise HTTPError(
            url="https://www.metaculus.com/api/v2/questions/1/forecast/",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"detail": "rate limited"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    with pytest.raises(MetaculusAPIError, match="429"):
        client.post_forecast(question_id=1, p_yes=0.51, comment="reason")


def test_get_resolution_maps_unresolved_and_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MetaculusClient(api_token="token")

    payloads = [
        {"question": {"resolution": None}},
        {"question": {"resolution": True}},
        {"question": {"resolution": False}},
    ]

    def _open(_request):
        return _FakeResponse(payloads.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", _open)

    assert client.get_resolution(question_id=10) is None
    assert client.get_resolution(question_id=10) is True
    assert client.get_resolution(question_id=10) is False


def test_get_open_questions_deserializes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MetaculusClient(api_token="token")

    payload = {
        "results": [
            {
                "id": 101,
                "title": "Will X happen?",
                "description": "desc",
                "resolution_criteria": "criteria",
                "resolve_time": "2026-07-01T00:00:00Z",
                "close_time": "2026-06-20T00:00:00Z",
            }
        ]
    }

    captured = {"url": ""}

    def _open(req):
        captured["url"] = req.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _open)

    got = client.get_open_questions(project_id=55)

    assert "project=55" in captured["url"]
    assert got == [
        MetaculusQuestion(
            question_id=101,
            title="Will X happen?",
            description="desc",
            resolution_criteria="criteria",
            resolution_date=date(2026, 7, 1),
            close_date=date(2026, 6, 20),
        )
    ]
