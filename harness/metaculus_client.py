"""Thin Metaculus REST transport for competition adapters.

This module intentionally handles only HTTP transport + shape conversion.
Loop orchestration belongs in a separate runner layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


class MetaculusAPIError(RuntimeError):
    """Domain error for non-success Metaculus API responses."""


@dataclass(frozen=True)
class MetaculusQuestion:
    question_id: int
    title: str
    description: str
    resolution_criteria: str
    resolution_date: date
    close_date: date


class MetaculusClient:
    def __init__(self, api_token: str, base_url: str = "https://www.metaculus.com") -> None:
        if not isinstance(api_token, str) or not api_token.strip():
            raise ValueError("api_token must be a non-empty string")
        self._api_token = api_token
        self._base_url = base_url.rstrip("/")

    def get_open_questions(self, project_id: int | str) -> list[MetaculusQuestion]:
        query = urlencode({"project": project_id, "status": "open"})
        payload = self._request_json("GET", f"/api2/questions/?{query}")

        rows = payload.get("results", [])
        out: list[MetaculusQuestion] = []
        for row in rows:
            resolve_raw = row.get("resolve_time") or row.get("scheduled_resolve_time") or row.get("resolution_date")
            close_raw = row.get("close_time") or row.get("scheduled_close_time") or row.get("close_date")
            if resolve_raw is None or close_raw is None:
                raise MetaculusAPIError("Question payload missing resolve/close time fields")

            out.append(
                MetaculusQuestion(
                    question_id=int(row["id"]),
                    title=str(row.get("title", "")),
                    description=str(row.get("description", "")),
                    resolution_criteria=str(row.get("resolution_criteria", "")),
                    resolution_date=self._parse_api_date(str(resolve_raw)),
                    close_date=self._parse_api_date(str(close_raw)),
                )
            )
        return out

    def post_forecast(self, question_id: int, p_yes: float, comment: str) -> None:
        if not (0.0 < float(p_yes) < 1.0):
            raise ValueError("p_yes must satisfy 0.0 < p_yes < 1.0")
        if not isinstance(comment, str) or not comment.strip():
            raise ValueError("comment must be a non-empty string")

        body = {
            "prediction": float(p_yes),
            "explanation": comment,
        }

        # Current Metaculus posting path (accepts post-id slugs/ids).
        try:
            self._request_json("POST", f"/api2/questions/{question_id}/predict/", body)
            return
        except MetaculusAPIError:
            # Backward-compatible fallback for older endpoint contracts.
            legacy_body = {"prediction": float(p_yes), "comment": comment}
            self._request_json("POST", f"/api/v2/questions/{question_id}/forecast/", legacy_body)

    def get_resolution(self, question_id: int) -> bool | None:
        payload = self._request_json("GET", f"/api/v2/questions/{question_id}/")
        question = payload.get("question", payload)
        resolution = question.get("resolution")

        if resolution is None:
            return None
        if isinstance(resolution, bool):
            return resolution

        normalized = str(resolution).strip().lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
        return None

    def _request_json(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = request.Request(
            url=f"{self._base_url}{path}",
            method=method,
            data=data,
            headers={
                "Authorization": f"Token {self._api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with request.urlopen(req) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except HTTPError as exc:
            body_text = ""
            if exc.fp is not None:
                body_text = exc.fp.read().decode("utf-8", errors="replace")
            raise MetaculusAPIError(
                f"Metaculus API error {exc.code} for {method} {path}: {body_text or exc.reason}"
            ) from exc
        except URLError as exc:
            raise MetaculusAPIError(f"Metaculus API transport error for {method} {path}: {exc.reason}") from exc

    @staticmethod
    def _parse_api_date(value: str) -> date:
        # Handles ISO timestamps like 2026-07-01T00:00:00Z.
        return date.fromisoformat(value[:10])


__all__ = ["MetaculusAPIError", "MetaculusClient", "MetaculusQuestion"]
