from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError


class MetaculusAPIError(Exception):
    """Wraps HTTP errors from the Metaculus API."""


@dataclass(frozen=True)
class MetaculusQuestion:
    question_id: int
    title: str
    description: str
    resolution_criteria: str
    resolution_date: date
    close_date: date


class MetaculusClient:
    """Thin client for the Metaculus API.

    API base: https://www.metaculus.com/api/v2
    Auth: Token <token> via Authorization header.
    """

    BASE = "https://www.metaculus.com/api/v2"

    def __init__(self, api_token: str) -> None:
        self._token = api_token

    # ── Public API ───────────────────────────────────────────────────

    def get_open_questions(self, project_id: int) -> list[MetaculusQuestion]:
        """Fetch open questions for a tournament project."""
        url = f"{self.BASE}/questions/?project={project_id}&status=open&limit=100"
        body = self._get(url)
        results = body.get("results", [])
        return [self._parse_question(r) for r in results]

    def post_forecast(
        self, question_id: int, p_yes: float, comment: str
    ) -> None:
        """Post a binary forecast to a Metaculus question.

        Raises ValueError for degenerate probabilities (0.0 or 1.0)
        BEFORE any network call.
        """
        if p_yes <= 0.0 or p_yes >= 1.0:
            raise ValueError(
                f"p_yes must be strictly between 0 and 1, got {p_yes}"
            )

        url = f"{self.BASE}/questions/{question_id}/forecast/"
        body = {"prediction": p_yes}
        self._post(url, body)

    def get_resolution(self, question_id: int) -> bool | None:
        """Fetch resolution status for a question.

        Returns None if unresolved, True/False if resolved.
        """
        url = f"{self.BASE}/questions/{question_id}/"
        body = self._get(url)
        question = body.get("question", {})
        resolution = question.get("resolution")
        if resolution is None:
            return None
        return bool(resolution)

    # ── Internal ─────────────────────────────────────────────────────

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Token {self._token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except HTTPError as e:
            raise MetaculusAPIError(
                f"Metaculus API {e.code} on {url}: {e.msg}"
            ) from e

    def _post(self, url: str, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Token {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            # Successful post returns {} — no body needed
        except HTTPError as e:
            raise MetaculusAPIError(
                f"Metaculus API {e.code} on {url}: {e.msg}"
            ) from e

    @staticmethod
    def _parse_question(raw: dict) -> MetaculusQuestion:
        return MetaculusQuestion(
            question_id=raw["id"],
            title=raw.get("title", ""),
            description=raw.get("description", ""),
            resolution_criteria=raw.get("resolution_criteria", ""),
            resolution_date=_parse_iso(raw.get("resolve_time")),
            close_date=_parse_iso(raw.get("close_time")),
        )


def _parse_iso(raw: str | None) -> date:
    if not raw:
        return date.today()
    return date.fromisoformat(raw.replace("Z", "+00:00")[:10])
