from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode

from harness.memory_schema import ToolCallRecord
from harness.query_mapper import PITViolationError, WebSearchRequest


class AskNewsAPIError(Exception):
    """Raised when the AskNews API returns a non-retryable error."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    summary: str
    url: str
    published_at: date
    source: str


class AskNewsSearchTool:
    """Web search via AskNews API with PIT-enforced date bounds."""

    BASE_URL = "https://api.asknews.app/v1/news/search"

    def __init__(
        self,
        api_key: str,
        max_results: int = 10,
        window_days: int = 90,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._window_days = window_days

    def __call__(self, request: WebSearchRequest) -> list[ToolCallRecord]:
        return self.search_raw(request)

    def search_raw(self, request: WebSearchRequest) -> list[ToolCallRecord]:
        """Synchronous search returning ToolCallRecord list."""
        results = self._fetch(request)
        if not results:
            return []

        # PIT check: reject any result published after the cutoff
        cutoff_dt = datetime(
            request.as_of_date.year,
            request.as_of_date.month,
            request.as_of_date.day,
            tzinfo=timezone.utc,
        ) + timedelta(days=1) - timedelta(seconds=1)

        for r in results:
            pub_dt = datetime(
                r.published_at.year,
                r.published_at.month,
                r.published_at.day,
                tzinfo=timezone.utc,
            )
            if pub_dt > cutoff_dt:
                raise PITViolationError(
                    f"AskNews returned post-cutoff result: "
                    f"{r.title} published {r.published_at} > cutoff {request.as_of_date}"
                )

        # Build notes: sorted desc by date, include URLs
        sorted_results = sorted(results, key=lambda r: r.published_at, reverse=True)
        urls = [r.url for r in sorted_results]
        dates = [r.published_at.isoformat() for r in sorted_results]
        notes_lines = []
        for d, u in zip(dates, urls):
            notes_lines.append(f"{d} {u}")
        notes = "\n".join(notes_lines)

        return [
            ToolCallRecord(
                tool_name="web_search",
                query=request.query,
                as_of_time=f"{request.as_of_date.isoformat()}T00:00:00Z",
                evidence_count=len(results),
                notes=notes,
            )
        ]

    async def async_search_raw(self, request: WebSearchRequest) -> list[ToolCallRecord]:
        """Async wrapper — runs search_raw in a thread (stub for contract)."""
        import asyncio

        return await asyncio.to_thread(self.search_raw, request)

    def _fetch(self, request: WebSearchRequest) -> list[SearchResult]:
        """Call AskNews API and parse results."""
        published_after = request.as_of_date - timedelta(days=self._window_days)

        start_dt = datetime(
            published_after.year,
            published_after.month,
            published_after.day,
            tzinfo=timezone.utc,
        )
        end_dt = (
            datetime(
                request.as_of_date.year,
                request.as_of_date.month,
                request.as_of_date.day,
                tzinfo=timezone.utc,
            )
            + timedelta(days=1)
            - timedelta(seconds=1)
        )

        params = {
            "query": request.query,
            "start_timestamp": str(int(start_dt.timestamp())),
            "end_timestamp": str(int(end_dt.timestamp())),
            "historical": "true",
            "method": "kw",
            "return_type": "dicts",
            "n_articles": str(self._max_results),
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"

        last_exc = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(req) as resp:
                    body = json.loads(resp.read())
                    as_dicts = body.get("as_dicts", [])
                    return [
                        SearchResult(
                            title=d.get("eng_title", d.get("title", "")),
                            summary=d.get("summary", ""),
                            url=d.get("article_url", ""),
                            published_at=_parse_date(d.get("pub_date", "")),
                            source=d.get("source_id", ""),
                        )
                        for d in as_dicts
                    ]
            except HTTPError as e:
                if e.code == 429:
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                raise AskNewsAPIError(
                    f"AskNews API error {e.code}: {e.msg}"
                ) from e

        raise AskNewsAPIError("Max retries exceeded")


def _parse_date(raw: str) -> date:
    if not raw:
        return date.today()
    # Handle ISO 8601 datetime strings
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
        return dt.date()
    except (ValueError, TypeError):
        return date.today()
