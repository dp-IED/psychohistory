from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import ClassVar
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from harness.memory_schema import ToolCallRecord
from harness.query_mapper import PITViolationError, WebSearchRequest


class AskNewsAPIError(RuntimeError):
    """Raised when AskNews API calls fail or return malformed payloads."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    summary: str
    url: str
    published_at: date
    source: str


def _utc_start_of_day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _utc_end_of_day(d: date) -> datetime:
    return _utc_start_of_day(d + timedelta(days=1)) - timedelta(seconds=1)


class AskNewsSearchTool:
    _executor: ClassVar[ThreadPoolExecutor | None] = None

    def __init__(
        self,
        api_key: str,
        window_days: int = 90,
        max_results: int = 10,
        endpoint: str = "https://api.asknews.app/v1/news/search",
        method: str = "kw",
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        if method not in {"kw", "nl", "both"}:
            raise ValueError('method must be one of "kw", "nl", "both"')
        self._api_key = api_key
        self._window_days = window_days
        self._max_results = max_results
        self._endpoint = endpoint
        self._method = method

    def _fetch_parsed(self, req: WebSearchRequest) -> list[SearchResult]:
        published_after = req.as_of_date - timedelta(days=self._window_days)
        published_before = req.as_of_date
        start_ts = int(_utc_start_of_day(published_after).timestamp())
        end_ts = int(_utc_end_of_day(published_before).timestamp())

        raw_results = self._search(
            query=req.query,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )
        parsed = [self._parse_result(item) for item in raw_results]
        parsed.sort(key=lambda item: item.published_at, reverse=True)

        for item in parsed:
            if item.published_at > req.as_of_date:
                raise PITViolationError(
                    f"AskNews returned post-cutoff result: {item.url} "
                    f"published {item.published_at} > cutoff {req.as_of_date}"
                )

        return parsed

    def search_raw(self, req: WebSearchRequest) -> list[SearchResult]:
        """Return raw SearchResult rows without ToolCallRecord wrapping."""
        return self._fetch_parsed(req)

    async def async_search_raw(self, req: WebSearchRequest) -> list[SearchResult]:
        """Async wrapper around search_raw — runs the sync HTTP call in a thread pool."""
        loop = asyncio.get_running_loop()
        if AskNewsSearchTool._executor is None:
            AskNewsSearchTool._executor = ThreadPoolExecutor(max_workers=4)
        return await loop.run_in_executor(AskNewsSearchTool._executor, self.search_raw, req)

    def to_tool_records(self, req: WebSearchRequest, parsed: list[SearchResult]) -> list[ToolCallRecord]:
        """Build ToolCallRecords from an already-fetched result list (single API round-trip)."""
        if not parsed:
            return []
        return [
            ToolCallRecord(
                tool_name="web_search",
                query=req.query,
                as_of_time=f"{req.as_of_date.isoformat()}T00:00:00Z",
                evidence_count=len(parsed),
                notes="; ".join(
                    f"{item.published_at.isoformat()}|{item.title}|{item.source}|{item.url}|{item.summary[:200]}"
                    for item in parsed
                ),
            )
        ]

    def __call__(self, req: WebSearchRequest) -> list[ToolCallRecord]:
        return self.to_tool_records(req, self._fetch_parsed(req))

    def _search(
        self,
        *,
        query: str,
        start_timestamp: int,
        end_timestamp: int,
    ) -> list[dict]:
        params: dict[str, str | int] = {
            "query": query,
            "n_articles": min(self._max_results, 100),
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "historical": "true",
            "method": self._method,
            "return_type": "dicts",
        }
        url = f"{self._endpoint}?{urlencode(params)}"

        req_http = request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            method="GET",
        )
        data: dict
        for attempt in range(3):
            try:
                with request.urlopen(req_http) as resp:
                    if getattr(resp, "status", 200) >= 400:
                        raise AskNewsAPIError(f"AskNews API error: status={resp.status}")
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise AskNewsAPIError(f"AskNews API error: status={exc.code}") from exc
            except URLError as exc:
                raise AskNewsAPIError(f"AskNews API error: {exc.reason}") from exc
            except json.JSONDecodeError as exc:
                raise AskNewsAPIError("AskNews API error: invalid JSON response") from exc

        results = data.get("as_dicts")
        if not isinstance(results, list):
            raise AskNewsAPIError("AskNews API error: response missing list field 'as_dicts'")
        return results

    @staticmethod
    def _parse_result(item: dict) -> SearchResult:
        published_at_raw = item.get("pub_date") or item.get("publishedAt") or item.get("published_at")
        if not isinstance(published_at_raw, str):
            raise AskNewsAPIError("AskNews API error: result missing pub_date")

        try:
            dt = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AskNewsAPIError("AskNews API error: invalid pub_date timestamp") from exc

        return SearchResult(
            title=str(item.get("eng_title") or item.get("title") or ""),
            summary=str(item.get("summary", "")),
            url=str(item.get("article_url") or item.get("url") or ""),
            published_at=dt.date(),
            source=str(item.get("source_id") or item.get("source") or ""),
        )


_asknews_rate_lock = threading.Lock()
_asknews_last_mono: float = 0.0
ASKNEWS_MIN_INTERVAL_SEC = 12.0


def rate_limited_asknews_call(tool: AskNewsSearchTool, req: WebSearchRequest) -> list[ToolCallRecord]:
    """Serialize AskNews calls and enforce a minimum interval (free tier ~5/min)."""

    global _asknews_last_mono
    with _asknews_rate_lock:
        now = time.monotonic()
        wait = ASKNEWS_MIN_INTERVAL_SEC - (now - _asknews_last_mono)
        if wait > 0:
            time.sleep(wait)
        result = tool(req)
        _asknews_last_mono = time.monotonic()
        return result
