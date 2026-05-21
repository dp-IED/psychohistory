"""Tournament watcher — polls Metaculus for question drops via /api/posts/.

Based on the official metaculus-bot template pattern.
Questions in the Summer Bot Tournament open for ~1.5hr windows.
This polls for new posts and triggers the forecast pipeline.

API endpoints (from main_with_no_framework.py):
  GET  /api/posts/         — list open posts by tournament
  GET  /api/posts/{id}/    — get post details (includes question data)
  POST /api/questions/forecast/ — submit forecast
  POST /api/comments/create/    — post rationale as comment
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ── Constants (from official bot) ────────────────────────────────────

API_BASE = "https://www.metaculus.com/api"
FORECAST_TYPES = "binary,multiple_choice,numeric,discrete"

KNOWN_TOURNAMENTS: dict[str, str | int] = {
    "summer-bot": 33022,
    "cup": 33021,
    "market-pulse": 33013,
    "minibench": "minibench",
    "q1-2025": 32627,
    "fall-2025": "fall-aib-2025",
}


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class WatcherConfig:
    tournament_id: str | int
    name: str
    poll_interval_seconds: int = 60
    """How often to check for new questions."""
    mode: str = "drop"
    """'drop' = detect new Qs, 'continuous' = re-forecast on schedule."""
    update_interval_hours: float | None = None
    """For continuous mode: re-forecast if last > N hours."""


# ── API helpers (following official bot patterns) ────────────────────


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def list_tournament_posts(
    tournament_id: str | int,
    token: str,
    *,
    status: str = "open",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """GET /api/posts/ — list posts for a tournament (official pattern)."""
    url = f"{API_BASE}/posts/"
    params = {
        "limit": str(limit),
        "offset": str(offset),
        "order_by": "-hotness",
        "forecast_type": FORECAST_TYPES,
        "tournaments": str(tournament_id),
        "statuses": status,
        "include_description": "true",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query}"

    req = Request(full_url, headers=_auth_headers(token))
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())  # type: ignore[no-any-return]
    except HTTPError as e:
        body = e.read().decode()[:300] if e.fp else ""
        logger.error("API error %d on %s: %s", e.code, url, body)
        raise


def get_post_details(post_id: int, token: str) -> dict[str, Any]:
    """GET /api/posts/{id}/ — full post details including question data."""
    url = f"{API_BASE}/posts/{post_id}/"
    req = Request(url, headers=_auth_headers(token))
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def post_forecast(
    question_id: int,
    forecast_payload: dict[str, Any],
    token: str,
) -> None:
    """POST /api/questions/forecast/ — submit a forecast."""
    url = f"{API_BASE}/questions/forecast/"
    body = json.dumps([{"question": question_id, **forecast_payload}]).encode()
    req = Request(url, data=body, headers=_auth_headers(token), method="POST")
    with urlopen(req, timeout=30) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Forecast post failed: {resp.status}")


def post_comment(post_id: int, comment_text: str, token: str) -> None:
    """POST /api/comments/create/ — post a private comment with rationale."""
    url = f"{API_BASE}/comments/create/"
    body = json.dumps(
        {
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        }
    ).encode()
    req = Request(url, data=body, headers=_auth_headers(token), method="POST")
    with urlopen(req, timeout=30) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Comment post failed: {resp.status}")


def binary_forecast_payload(p_yes: float) -> dict[str, Any]:
    """Build forecast payload for binary questions."""
    return {
        "probability_yes": p_yes,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }


# ── Question extraction (from post data) ─────────────────────────────


@dataclass(frozen=True)
class QuestionInfo:
    question_id: int
    post_id: int
    title: str
    description: str
    resolution_criteria: str
    fine_print: str
    question_type: str  # binary, numeric, multiple_choice, discrete
    close_time: str
    resolve_time: str
    status: str


def extract_open_questions(
    posts_data: dict[str, Any],
) -> list[tuple[int, int]]:
    """Extract (question_id, post_id) pairs from /api/posts/ response.

    Matches the official bot's get_open_question_ids_from_tournament().
    """
    post_dict: dict[int, list[dict]] = {}
    for post in posts_data.get("results", []):
        if question := post.get("question"):
            post_dict[post["id"]] = [question]
        # Group questions might have multiple
        if questions := post.get("questions"):
            post_dict[post["id"]] = questions

    open_pairs: list[tuple[int, int]] = []
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                open_pairs.append((question["id"], post_id))

    return open_pairs


# ── State ────────────────────────────────────────────────────────────


@dataclass
class WatcherState:
    tournament_id: str
    seen_post_ids: set[int] = field(default_factory=set)
    forecasted: dict[str, str] = field(default_factory=dict)
    """question_id → forecast_timestamp ISO 8601"""
    last_poll: str = ""
    drops_detected: int = 0
    forecasts_posted: int = 0

    def to_dict(self) -> dict:
        return {
            "tournament_id": self.tournament_id,
            "seen_post_ids": sorted(self.seen_post_ids),
            "forecasted": self.forecasted,
            "last_poll": self.last_poll,
            "drops_detected": self.drops_detected,
            "forecasts_posted": self.forecasts_posted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WatcherState":
        return cls(
            tournament_id=d["tournament_id"],
            seen_post_ids=set(d.get("seen_post_ids", [])),
            forecasted=d.get("forecasted", {}),
            last_poll=d.get("last_poll", ""),
            drops_detected=d.get("drops_detected", 0),
            forecasts_posted=d.get("forecasts_posted", 0),
        )


# ── Events ───────────────────────────────────────────────────────────


@dataclass
class DropEvent:
    tournament_id: str
    timestamp: str
    questions: list[QuestionInfo]
    total_open: int


OnDropCallback = Callable[[DropEvent], None]


# ── Watcher ──────────────────────────────────────────────────────────


class TournamentWatcher:
    """Polls /api/posts/ for new questions and triggers callbacks."""

    def __init__(
        self,
        token: str,
        config: WatcherConfig,
        state_dir: Path,
        *,
        on_drop: OnDropCallback | None = None,
    ) -> None:
        self._token = token
        self._config = config
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._on_drop = on_drop

    @property
    def state(self) -> WatcherState:
        return self._state

    def poll_once(self) -> DropEvent | None:
        """Single poll cycle. Returns DropEvent if new questions found."""
        now = datetime.now(timezone.utc).isoformat()
        self._state.last_poll = now

        try:
            posts = list_tournament_posts(self._config.tournament_id, self._token)
        except Exception as e:
            logger.error("Poll failed for %s: %s", self._config.name, e)
            return None

        open_pairs = extract_open_questions(posts)
        current_post_ids = {pid for _, pid in open_pairs}
        new_post_ids = current_post_ids - self._state.seen_post_ids

        # Update seen set
        self._state.seen_post_ids |= current_post_ids

        if not new_post_ids:
            return None

        # Fetch full details for new questions (with rate-limit safety)
        new_questions: list[QuestionInfo] = []
        for qid, pid in open_pairs:
            if pid not in new_post_ids:
                continue
            qdata = _fetch_post_details_safe(pid, self._token)
            if qdata is not None and qdata.get("status") == "open":
                new_questions.append(
                    QuestionInfo(
                        question_id=qid,
                        post_id=pid,
                        title=qdata.get("title", ""),
                        description=qdata.get("description", ""),
                        resolution_criteria=qdata.get(
                            "resolution_criteria", ""
                        ),
                        fine_print=qdata.get("fine_print", ""),
                        question_type=qdata.get("type", "binary"),
                        close_time=qdata.get(
                            "scheduled_close_time", ""
                        ),
                        resolve_time=qdata.get(
                            "scheduled_resolve_time", ""
                        ),
                        status=qdata.get("status", ""),
                    )
                )

        if not new_questions:
            return None

        self._state.drops_detected += 1

        logger.info(
            "📬 DROP: %s — %d new questions (total open: %d)",
            self._config.name,
            len(new_questions),
            len(open_pairs),
        )
        for q in new_questions:
            logger.info("  Q%d (post %d): %s", q.question_id, q.post_id, q.title[:100])

        event = DropEvent(
            tournament_id=str(self._config.tournament_id),
            timestamp=now,
            questions=new_questions,
            total_open=len(open_pairs),
        )

        if self._on_drop:
            self._on_drop(event)

        self._save_state()
        return event

    def record_forecast(self, question_id: int) -> None:
        """Mark a question as forecasted."""
        self._state.forecasted[str(question_id)] = (
            datetime.now(timezone.utc).isoformat()
        )
        self._state.forecasts_posted += 1
        self._save_state()

    def watch(self, max_cycles: int = 0) -> None:
        """Run poll loop. max_cycles=0 → run forever."""
        cycle = 0
        logger.info(
            "👀 Watching %s (mode=%s, interval=%ds)",
            self._config.name,
            self._config.mode,
            self._config.poll_interval_seconds,
        )
        while max_cycles == 0 or cycle < max_cycles:
            self.poll_once()
            cycle += 1
            if max_cycles > 0 and cycle >= max_cycles:
                break
            time.sleep(self._config.poll_interval_seconds)

    def _state_path(self) -> Path:
        safe = str(self._config.tournament_id).replace("/", "_")
        return self._state_dir / f"state_{safe}.json"

    def _load_state(self) -> WatcherState:
        path = self._state_path()
        if path.exists():
            try:
                return WatcherState.from_dict(json.loads(path.read_text()))
            except Exception:
                logger.warning("Corrupt state, starting fresh")
        return WatcherState(tournament_id=str(self._config.tournament_id))

    def _save_state(self) -> None:
        self._state_path().write_text(
            json.dumps(self._state.to_dict(), indent=2, default=str)
        )


# ── Rate-limit-safe fetch ─────────────────────────────────────────────


def _fetch_post_details_safe(
    post_id: int,
    token: str,
    *,
    _base_delay: float = 1.5,
    _max_retries: int = 3,
) -> dict[str, Any] | None:
    """Fetch post details with rate-limit awareness.

    - 1.5s delay between successive calls (regardless of caller).
    - On 429: respects ``Retry-After`` header; falls back to
      exponential backoff (2s → 4s → 8s).
    - Returns ``None`` if all retries are exhausted or a non-429
      error occurs.
    """
    import time as _time

    _time.sleep(_base_delay)

    for attempt in range(_max_retries + 1):
        try:
            details = get_post_details(post_id, token)
            return details.get("question", {})
        except HTTPError as e:
            if e.code != 429:
                logger.warning(
                    "Failed to get details for post %d: %s", post_id, e
                )
                return None

            # ── Rate-limited ──
            retry_after: float | None = None
            try:
                if e.headers is not None:
                    ra = e.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = float(ra)
            except (ValueError, TypeError):
                pass

            if attempt < _max_retries:
                delay = retry_after if retry_after else 2.0 ** (attempt + 1)
                logger.warning(
                    "429 on post %d (attempt %d/%d), sleeping %.1fs",
                    post_id, attempt + 1, _max_retries + 1, delay,
                )
                _time.sleep(delay)
            else:
                logger.error(
                    "Rate-limited on post %d after %d retries, skipping",
                    post_id, _max_retries,
                )
                return None

    return None


# ── Convenience: forecast handler ────────────────────────────────────


def _build_question_info(qid: int, pid: int, qdata: dict[str, Any]) -> QuestionInfo:
    return QuestionInfo(
        question_id=qid,
        post_id=pid,
        title=qdata.get("title", ""),
        description=qdata.get("description", ""),
        resolution_criteria=qdata.get("resolution_criteria", ""),
        fine_print=qdata.get("fine_print", ""),
        question_type=qdata.get("type", "binary"),
        close_time=qdata.get("scheduled_close_time", ""),
        resolve_time=qdata.get("scheduled_resolve_time", ""),
        status=qdata.get("status", ""),
    )


def create_drop_handler(
    token: str,
    *,
    dry_run: bool = True,
) -> OnDropCallback:
    """Factory for a drop handler that posts forecasts.

    Replace the inner forecast logic with your own pipeline
    (librarian → vault → forecaster → post).
    """

    def handle_drop(event: DropEvent) -> None:
        logger.info(
            "📝 Processing %d new questions for %s",
            len(event.questions),
            event.tournament_id,
        )
        for q in event.questions:
            if q.question_type != "binary":
                logger.info(
                    "  Q%d: skipping non-binary (%s)", q.question_id, q.question_type
                )
                continue

            # TODO: Replace with actual PIT librarian + vault forecast
            # For now, post a placeholder forecast
            logger.info("  Q%d: %s", q.question_id, q.title[:80])

            if not dry_run:
                try:
                    post_forecast(
                        q.question_id,
                        binary_forecast_payload(0.5),
                        token,
                    )
                    post_comment(
                        q.post_id,
                        f"Placeholder forecast for: {q.title}",
                        token,
                    )
                    logger.info("    ✅ Posted forecast + comment")
                except Exception as e:
                    logger.error("    ❌ Failed: %s", e)

    return handle_drop
