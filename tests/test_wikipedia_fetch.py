from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from ingest.wikipedia_fetch import fetch_article


def _page_payload(*, title: str, revisions: list[dict]) -> dict:
    return {
        "query": {
            "pages": [
                {
                    "title": title,
                    "pageid": 1,
                    "revisions": revisions,
                }
            ]
        }
    }


def test_overlap_fetch_falls_back_to_first_revision_after_as_of(monkeypatch) -> None:
    calls = []

    def fake_api_get(params, *, timeout=30):
        calls.append(dict(params))
        if params.get("rvdir") == "older":
            return _page_payload(title="Arab Spring", revisions=[])
        return _page_payload(
            title="Arab Spring",
            revisions=[
                {
                    "revid": 42,
                    "timestamp": "2011-01-15T00:00:00Z",
                    "slots": {"main": {"content": "Arab Spring article text"}},
                }
            ],
        )

    monkeypatch.setattr("ingest.wikipedia_fetch._api_get", fake_api_get)

    out = fetch_article(
        title="Arab Spring",
        url="https://en.wikipedia.org/wiki/Arab_Spring",
        pit_mode="arab_spring_overlap",
        as_of="2010-01-01",
    )

    assert out["revision_id"] == "42"
    assert out["pit_status"] == "pit_fallback_after_as_of"
    assert out["pit_warning"] is not None
    assert len(calls) == 2
    assert calls[0]["rvdir"] == "older"
    assert calls[1]["rvdir"] == "newer"


def test_static_fetch_uses_latest_revision(monkeypatch) -> None:
    def fake_api_get(params, *, timeout=30):
        return _page_payload(
            title="Black Death",
            revisions=[
                {
                    "revid": 99,
                    "timestamp": "2026-04-27T00:00:00Z",
                    "slots": {"main": {"content": "Black Death article text"}},
                }
            ],
        )

    monkeypatch.setattr("ingest.wikipedia_fetch._api_get", fake_api_get)

    out = fetch_article(
        title="Black Death",
        url="https://en.wikipedia.org/wiki/Black_Death",
        pit_mode="static",
        as_of=None,
    )

    assert out["revision_id"] == "99"
    assert out["pit_status"] == "static_latest"
    assert out["pit_warning"] is None


def test_static_fetch_retries_transient_api_failure(monkeypatch) -> None:
    calls = []

    def fake_api_get(params, *, timeout=30):
        calls.append(dict(params))
        if len(calls) == 1:
            raise TimeoutError("temporary wikipedia timeout")
        return _page_payload(
            title="Black Death",
            revisions=[
                {
                    "revid": 100,
                    "timestamp": "2026-04-27T00:00:00Z",
                    "slots": {"main": {"content": "Recovered Black Death article text"}},
                }
            ],
        )

    monkeypatch.setattr("ingest.wikipedia_fetch._api_get", fake_api_get)
    monkeypatch.setattr("ingest.wikipedia_fetch.time.sleep", lambda _: None)

    out = fetch_article(
        title="Black Death",
        url="https://en.wikipedia.org/wiki/Black_Death",
        pit_mode="static",
        as_of=None,
    )

    assert out["revision_id"] == "100"
    assert len(calls) == 2


def test_static_fetch_retries_http_429_rate_limit_after_retry_after_delay(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_api_get(params, *, timeout=30):
        calls.append(dict(params))
        if len(calls) == 1:
            raise HTTPError(
                url="https://en.wikipedia.org/w/api.php",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "2"},
                fp=BytesIO(b"rate limited"),
            )
        return _page_payload(
            title="Black Death",
            revisions=[
                {
                    "revid": 101,
                    "timestamp": "2026-04-27T00:00:00Z",
                    "slots": {"main": {"content": "Recovered after rate limit"}},
                }
            ],
        )

    monkeypatch.setattr("ingest.wikipedia_fetch._api_get", fake_api_get)
    monkeypatch.setattr("ingest.wikipedia_fetch.time.sleep", sleeps.append)

    out = fetch_article(
        title="Black Death",
        url="https://en.wikipedia.org/wiki/Black_Death",
        pit_mode="static",
        as_of=None,
    )

    assert out["revision_id"] == "101"
    assert len(calls) == 2
    assert sleeps == [2.0]
