from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from urllib import request
from urllib.error import HTTPError, URLError

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


class AskNewsSearchTool:
    def __init__(
        self,
        api_key: str,
        window_days: int = 90,
        max_results: int = 10,
        endpoint: str = "https://api.asknews.app/v1/search",
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        self._api_key = api_key
        self._window_days = window_days
        self._max_results = max_results
        self._endpoint = endpoint

    def __call__(self, req: WebSearchRequest) -> list[ToolCallRecord]:
        payload = {
            "query": req.query,
            "limit": self._max_results,
            "publishedBefore": req.as_of_date.isoformat(),
            "publishedAfter": (req.as_of_date - timedelta(days=self._window_days)).isoformat(),
        }

        raw_results = self._search(payload)
        parsed = [self._parse_result(item) for item in raw_results]
        parsed.sort(key=lambda item: item.published_at, reverse=True)

        for item in parsed:
            if item.published_at > req.as_of_date:
                raise PITViolationError(
                    f"AskNews returned post-cutoff result: {item.url} "
                    f"published {item.published_at} > cutoff {req.as_of_date}"
                )

        if not parsed:
            return []

        return [
            ToolCallRecord(
                tool_name="web_search",
                query=req.query,
                as_of_time=f"{req.as_of_date.isoformat()}T00:00:00Z",
                evidence_count=len(parsed),
                notes="; ".join(
                    f"{item.published_at.isoformat()} {item.source} {item.url}" for item in parsed
                ),
            )
        ]

    def _search(self, payload: dict) -> list[dict]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                if getattr(resp, "status", 200) >= 400:
                    raise AskNewsAPIError(f"AskNews API error: status={resp.status}")
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise AskNewsAPIError(f"AskNews API error: status={exc.code}") from exc
        except URLError as exc:
            raise AskNewsAPIError(f"AskNews API error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise AskNewsAPIError("AskNews API error: invalid JSON response") from exc

        results = data.get("results")
        if not isinstance(results, list):
            raise AskNewsAPIError("AskNews API error: response missing list field 'results'")
        return results

    @staticmethod
    def _parse_result(item: dict) -> SearchResult:
        published_at_raw = item.get("publishedAt") or item.get("published_at")
        if not isinstance(published_at_raw, str):
            raise AskNewsAPIError("AskNews API error: result missing publishedAt")

        try:
            dt = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AskNewsAPIError("AskNews API error: invalid publishedAt timestamp") from exc

        return SearchResult(
            title=str(item.get("title", "")),
            summary=str(item.get("summary", "")),
            url=str(item.get("url", "")),
            published_at=dt.date(),
            source=str(item.get("source", "")),
        )
