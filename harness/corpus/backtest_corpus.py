"""Build deterministic backtest corpora from resolved public prediction-market APIs."""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal, cast

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
MANIFOLD_MARKETS_URL = "https://api.manifold.markets/v0/markets"
MANIFOLD_BETS_URL = "https://api.manifold.markets/v0/bets"
MANIFOLD_MARKET_HISTORY_URL = "https://api.manifold.markets/v0/market/{market_id}/history"
USER_AGENT = "Mozilla/5.0 (compatible; psychohistory-harness/0.1)"

# Map Polymarket tag labels to forecast category vocabulary.
# Keys are the human-readable tag strings Polymarket attaches to markets.
# Update as new Polymarket tag conventions emerge.
TAG_TO_CATEGORY: dict[str, str] = {
    "politics": "politics",
    "crypto": "crypto",
    "sports": "sports",
    "economics": "economics",
    "science": "science",
    "technology": "technology",
    "entertainment": "culture",
    "culture": "culture",
    "world": "politics",
    "finance": "economics",
    "business": "economics",
    "health": "science",
    "weather": "science",
    "environment": "science",
}

# Optional UUID -> category for Polymarket `groupItemTagUUIDs` when `tags` is absent.
POLYMARKET_GROUP_TAG_UUID_TO_CATEGORY: dict[str, str] = {}

_QUESTION_KEYWORD_HINTS: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset(
            {
                "election",
                "president",
                "senate",
                "congress",
                "parliament",
                "minister",
                "government",
                "vote",
                "ballot",
                "governor",
            },
        ),
        "politics",
    ),
    (frozenset({"bitcoin", "ethereum", "crypto", "btc", "defi"}), "crypto"),
    (
        frozenset(
            {
                "nba",
                "nfl",
                "mlb",
                "soccer",
                "football",
                "super bowl",
                "olympics",
                "ucl",
                "f1",
                "formula 1",
            },
        ),
        "sports",
    ),
    (frozenset({"gdp", "inflation", "fed", "recession", "stock", "unemployment"}), "economics"),
    (
        frozenset(
            {
                "climate",
                "covid",
                "vaccine",
                "disease",
                "hurricane",
                "earthquake",
                "weather",
                "biology",
                "gene",
                "space",
                "nasa",
            },
        ),
        "science",
    ),
    (frozenset({"iphone", "ai model", "openai", "google", "tesla", "tech"}), "technology"),
    (frozenset({"oscar", "movie", "album", "celebr", "culture"}), "culture"),
)


def _load_json_field(value: object) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return cast(list[Any], value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _iso_date_only(value: object) -> date | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if len(text) >= 10:
            head = text[:10]
            return date.fromisoformat(head)
        return date.fromisoformat(text)
    except ValueError:
        return None


def _volume_float(raw: dict[str, object]) -> float | None:
    for key in ("volumeNum", "volume"):
        v = raw.get(key)
        if v in (None, ""):
            continue
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _parse_resolution_price_scalar(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    try:
        if isinstance(value, str):
            stripped = value.strip()
            try:
                parsed_json = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_json = None
            if isinstance(parsed_json, (int, float)):
                return float(parsed_json)
            return float(stripped)
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _resolution_bool_from_gamma_price(price: float) -> bool | None:
    if math.isclose(price, 1.0):
        return True
    if math.isclose(price, 0.0):
        return False
    return None


def _primary_group_tag_uuid(raw: dict[str, object]) -> str | None:
    for key in ("groupItemTagUUIDs", "groupItemTagUuids"):
        items = raw.get(key)
        if items is None or items == "":
            continue
        if isinstance(items, list) and items:
            return str(items[0])
        if isinstance(items, str):
            try:
                parsed = json.loads(items)
            except json.JSONDecodeError:
                parsed = items
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
    return None


def _polymarket_tag_token(tag: object) -> str | None:
    """Normalize a Polymarket tag entry from the API to a lowercase lookup key."""

    if isinstance(tag, str):
        text_cell = tag.strip().lower()
        return text_cell or None
    if isinstance(tag, dict):
        for key in ("label", "slug", "name"):
            inner_cell = tag.get(key)
            if isinstance(inner_cell, str) and inner_cell.strip():
                return inner_cell.strip().lower()
        return None
    return None


def _category_from_polymarket_question(question: str) -> str | None:
    lower = question.lower()
    for keywords, category_cell in _QUESTION_KEYWORD_HINTS:
        for phrase in keywords:
            if phrase in lower:
                return category_cell
    return None


def _extract_polymarket_category(raw: dict[str, object]) -> str:
    """Extract a readable category from Polymarket market metadata."""

    tags_blob = raw.get("tags")
    if isinstance(tags_blob, list):
        for tag in tags_blob:
            tag_str = _polymarket_tag_token(tag)
            if tag_str is None:
                continue
            if tag_str in TAG_TO_CATEGORY:
                return TAG_TO_CATEGORY[tag_str]
            hyphen_parts = tag_str.replace("/", "-").split("-")
            for part in hyphen_parts:
                token = part.strip()
                if token in TAG_TO_CATEGORY:
                    return TAG_TO_CATEGORY[token]

    category_blob = raw.get("category")
    if isinstance(category_blob, str) and category_blob.strip():
        normalized = category_blob.strip().lower()
        if normalized in TAG_TO_CATEGORY:
            return TAG_TO_CATEGORY[normalized]

    mapped_uuid = _primary_group_tag_uuid(raw)
    if mapped_uuid and mapped_uuid in POLYMARKET_GROUP_TAG_UUID_TO_CATEGORY:
        return POLYMARKET_GROUP_TAG_UUID_TO_CATEGORY[mapped_uuid]

    question_blob = raw.get("question")
    if isinstance(question_blob, str):
        hinted = _category_from_polymarket_question(question_blob)
        if hinted is not None:
            return hinted

    return "general"


def _is_strict_yes_no_outcomes(raw: dict[str, object]) -> bool:
    parsed = _load_json_field(raw.get("outcomes"))
    return [str(label) for label in parsed] == ["Yes", "No"]


def _outcome_prices_list(raw: dict[str, object]) -> list[float] | None:
    prices_raw = _load_json_field(raw.get("outcomePrices"))
    out: list[float] = []
    for item in prices_raw:
        try:
            out.append(float(item))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return out


@dataclass(frozen=True)
class BacktestQuestion:
    question_id: str
    source: Literal["polymarket", "kalshi", "manifold", "metaculus"]
    question_text: str
    open_date: date
    close_date: date
    resolution: bool
    market_price_at_open: float | None
    category: str | None  # human-readable taxonomy (Polymarket: mapped labels; Manifold: first tag slug)
    volume: float = 0.0  # total volume in USDC (higher = more liquid)

    def __post_init__(self) -> None:
        if not (self.open_date < self.close_date):
            raise ValueError("open_date must be strictly before close_date")

        price = self.market_price_at_open
        if price is not None:
            pr = float(price)
            if not (0.0 <= pr <= 1.0):
                raise ValueError("market_price_at_open must be in [0.0, 1.0]")


def normalize_polymarket_market(
    raw: dict[str, object],
    *,
    min_close_date: date,
) -> BacktestQuestion | None:
    # Use outcomePrices (not resolutionPrice) to detect terminal resolution,
    # matching the existing ingest/polymarket_resolved.py convention.
    prices = _outcome_prices_list(raw)
    if prices is None or len(prices) != 2:
        return None
    if {round(p, 6) for p in prices} != {0.0, 1.0}:
        return None
    resolution_flag = prices[0] > prices[1]  # True if YES price > NO price

    volumes = _volume_float(raw)
    if volumes is None:
        return None
    question_volume = volumes

    close_date_value = _iso_date_only(raw.get("endDate"))
    if close_date_value is None:
        return None
    if close_date_value < min_close_date:
        return None

    open_date_candidate = _iso_date_only(raw.get("startDate"))
    if open_date_candidate is None:
        open_date_candidate = _iso_date_only(raw.get("createdAt"))
    if open_date_candidate is None:
        return None

    if not _is_strict_yes_no_outcomes(raw):
        return None

    question_id = str(raw.get("conditionId") or raw.get("id") or "")
    if not question_id:
        return None

    question_text = str(raw.get("question") or "")
    if not question_text.strip():
        return None

    market_open: float | None = None
    # outcomePrices on closed/resolved markets are terminal [0,1] or [1,0],
    # not opening prices.  Set to None; opening prices require historical
    # trade data from a different endpoint.

    category = _extract_polymarket_category(raw)

    if not (open_date_candidate < close_date_value):
        return None

    return BacktestQuestion(
        question_id=question_id,
        source="polymarket",
        question_text=question_text,
        open_date=open_date_candidate,
        close_date=close_date_value,
        resolution=resolution_flag,
        market_price_at_open=market_open,
        category=category,
        volume=question_volume,
    )


def normalize_manifold_market(
    raw: dict[str, object],
    *,
    min_close_time_ms: int,
    opening_probability: float | None = None,
) -> BacktestQuestion | None:
    if raw.get("isResolved") is not True:
        return None

    outcome_type_value = raw.get("outcomeType")
    if str(outcome_type_value) != "BINARY":
        return None

    resolution_value = raw.get("resolution")
    if isinstance(resolution_value, str):
        resolution_upper = resolution_value.upper()
        if resolution_upper == "YES":
            resolution_flag = True
        elif resolution_upper == "NO":
            resolution_flag = False
        else:
            return None
    else:
        return None

    close_raw = raw.get("closeTime")
    if isinstance(close_raw, bool) or close_raw in (None, ""):
        return None
    try:
        close_ms = int(close_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if close_ms < min_close_time_ms:
        return None

    created_raw = raw.get("createdTime")
    if isinstance(created_raw, bool) or created_raw in (None, ""):
        return None
    try:
        created_ms = int(created_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    question_id_value = raw.get("id")
    question_id_text = str(question_id_value).strip()
    question_text_raw = raw.get("question") or ""

    question_text_text = str(question_text_raw).strip()
    if not question_id_text or not question_text_text:
        return None

    open_date_candidate = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).date()
    close_date_candidate = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc).date()

    if not (open_date_candidate < close_date_candidate):
        return None

    market_open_price: float | None = None
    if opening_probability is not None:
        try:
            coerced_open = float(opening_probability)
        except (TypeError, ValueError):
            market_open_price = None
        else:
            if 0.0 <= coerced_open <= 1.0:
                market_open_price = coerced_open
    elif "probability" in raw:
        probability_obj = raw.get("probability")
        try:
            market_open_price = float(probability_obj)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            market_open_price = None
        else:
            if market_open_price < 0.0 or market_open_price > 1.0:
                market_open_price = None

    category_first = _manifold_category_tag(raw.get("tags"))

    return BacktestQuestion(
        question_id=question_id_text,
        source="manifold",
        question_text=question_text_text,
        open_date=open_date_candidate,
        close_date=close_date_candidate,
        resolution=resolution_flag,
        market_price_at_open=market_open_price,
        category=category_first,
    )


def _manifold_category_tag(raw_tags: object) -> str | None:
    if isinstance(raw_tags, list):
        return str(raw_tags[0]) if raw_tags else None
    if isinstance(raw_tags, str):
        stripped = raw_tags.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            entries = _load_json_field(raw_tags)
            return str(entries[0]) if entries else None
        return stripped
    return None


def _min_close_datetime_to_utc_epoch_ms(day: date) -> int:
    dt = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _http_json_get(url: str) -> object:
    request_obj = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request_obj, timeout=120) as response_body:
        raw_bytes = response_body.read().decode("utf-8")
    return json.loads(raw_bytes)


def _opening_probability_from_manifold_history(payload: dict[str, object]) -> float | None:
    history_cell = payload.get("history")
    if not isinstance(history_cell, list) or not history_cell:
        return None

    earliest: tuple[int, float] | None = None
    for point in history_cell:
        if not isinstance(point, dict):
            continue
        t_cell = point.get("t")
        p_cell = point.get("p")
        if isinstance(t_cell, bool) or not isinstance(t_cell, (int, float)):
            continue
        if isinstance(p_cell, bool) or not isinstance(p_cell, (int, float)):
            continue
        prob = float(p_cell)
        if not (0.0 <= prob <= 1.0):
            continue
        ts = int(t_cell)
        if earliest is None or ts < earliest[0]:
            earliest = (ts, prob)

    return None if earliest is None else earliest[1]


def _opening_probability_from_manifold_first_bet(market_id: str) -> float | None:
    """Infer opening YES probability from the first trade's probBefore."""

    bets_params = urllib.parse.urlencode({"contractId": market_id, "limit": "1", "order": "asc"})
    bets_url = f"{MANIFOLD_BETS_URL}?{bets_params}"
    try:
        bets_cell = _http_json_get(bets_url)
    except Exception:
        return None
    if not isinstance(bets_cell, list) or not bets_cell:
        return None
    first_cell = bets_cell[0]
    if not isinstance(first_cell, dict):
        return None
    prob_before = first_cell.get("probBefore")
    if isinstance(prob_before, bool) or not isinstance(prob_before, (int, float)):
        return None
    result = float(prob_before)
    if 0.0 <= result <= 1.0:
        return result
    return None


def _fetch_manifold_opening_probability(market_id: str) -> float | None:
    """Opening YES probability from market history API, falling back to first-bet probBefore.

    Manifold previously documented a `/v0/market/{id}/history` series (`t`, `p`); some deployments
    return 404. When history is absent or unusable we use `/v0/bets?contractId=&order=asc`:
    ``probBefore`` on the chronologically first bet equals the YES probability immediately before any
    trade (i.e. the effective opening probability for activity-based markets).
    """

    url = MANIFOLD_MARKET_HISTORY_URL.format(market_id=urllib.parse.quote(market_id))
    payload: dict[str, object] | None = None
    try:
        unpacked = _http_json_get(url)
        if isinstance(unpacked, dict):
            typed = dict[str, object]((str(inner_k), inner_v) for inner_k, inner_v in unpacked.items())
            payload = typed
    except Exception:
        payload = None

    if payload is not None:
        hist_open = _opening_probability_from_manifold_history(payload)
        if hist_open is not None:
            return hist_open

    return _opening_probability_from_manifold_first_bet(market_id)


def _typed_polymarket_rows(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        return []
    typed_rows: list[dict[str, object]] = []
    for row in payload:
        if isinstance(row, dict):
            typed_rows.append(
                dict[str, object]((str(key_cell), cell) for key_cell, cell in row.items()),
            )
    return typed_rows


def _coerce_manifold_markets_page(
    payload: object,
    *,
    requested_limit: int,
) -> tuple[list[dict[str, object]], str | None, bool]:
    """Unpack Manifold v0 /markets response — returns plain array per API docs."""

    # Manifold v0 API returns plain array, not {"markets": [...]}.
    if not isinstance(payload, list) or not payload:
        return [], None, True

    parsed_rows: list[dict[str, object]] = []
    for candidate in payload:
        if isinstance(candidate, dict):
            parsed_rows.append(
                dict[str, object](
                    (str(inner_key), inner_val) for inner_key, inner_val in candidate.items()
                ),
            )

    trailing = payload[-1]
    trailing_id_cell: object | None
    if isinstance(trailing, dict):
        trailing_id_cell = dict[str, object](trailing).get("id")
    else:
        trailing_id_cell = None

    cursor_candidate = str(trailing_id_cell) if trailing_id_cell not in (None, "") else None
    halted = len(payload) < requested_limit or cursor_candidate is None
    return parsed_rows, cursor_candidate, halted


def build_polymarket_corpus(
    min_date: date,
    max_questions: int,
    *,
    allowed_categories: frozenset[str] | None = None,
) -> list[BacktestQuestion]:
    """
    Build a backtest corpus from resolved Polymarket markets.

    Parameters
    ----------
    min_date:
        Earliest close date to include.
    max_questions:
        Maximum number of questions to return.
    allowed_categories:
        If set, only include questions whose category is in this set.
        Useful for filtering out sports/culture/weather markets.
        Set to ``None`` to include all categories (default).
    """
    if max_questions < 1:
        return []

    corpus: list[BacktestQuestion] = []
    offset = 0
    seen_ids: set[str] = set()
    max_offset = 1000  # cap pagination to prevent infinite loops

    while len(corpus) < max_questions:
        if offset >= max_offset:
            break
        page_limit = min(100, max(max_questions - len(corpus), 1))

        params = urllib.parse.urlencode(
            {"closed": "true", "limit": str(page_limit), "offset": str(offset),
             "order": "volume", "ascending": "false"},
        )
        url = f"{GAMMA_MARKETS_URL}?{params}"

        decoded = _http_json_get(url)
        if not isinstance(decoded, list):
            break

        if len(decoded) == 0:
            break

        typed_rows = _typed_polymarket_rows(decoded)

        stride = len(decoded)
        offset += stride

        for typed_row in typed_rows:
            record = normalize_polymarket_market(typed_row, min_close_date=min_date)

            if record is None:
                continue
            if record.question_id in seen_ids:
                continue
            if allowed_categories is not None and (record.category is None or record.category not in allowed_categories):
                continue
            corpus.append(record)
            seen_ids.add(record.question_id)
            if len(corpus) >= max_questions:
                break

        time.sleep(0.05)

        if stride == 0:
            break

        if len(corpus) >= max_questions:
            break

    return corpus[:max_questions]


def build_manifold_corpus(min_date: date, max_questions: int) -> list[BacktestQuestion]:
    if max_questions < 1:
        return []

    min_ms = _min_close_datetime_to_utc_epoch_ms(min_date)
    corpus: list[BacktestQuestion] = []
    seen_ids: set[str] = set()
    cursor_before: str | None = None

    while len(corpus) < max_questions:
        page_limit = min(500, max(max_questions - len(corpus), 1))

        params: list[tuple[str, str]] = [("limit", str(page_limit))]
        if cursor_before is not None:
            params.append(("before", cursor_before))

        encoded = urllib.parse.urlencode(params)
        url = f"{MANIFOLD_MARKETS_URL}?{encoded}"
        decoded_payload = _http_json_get(url)
        markets_page, explicit_last, exhausted = _coerce_manifold_markets_page(
            decoded_payload,
            requested_limit=page_limit,
        )

        if not markets_page:
            break

        for typed_row in markets_page:
            record = normalize_manifold_market(typed_row, min_close_time_ms=min_ms)

            if record is None:
                continue
            if record.question_id in seen_ids:
                continue

            opening_candidate = _fetch_manifold_opening_probability(record.question_id)
            time.sleep(0.01)

            if opening_candidate is not None:
                record = BacktestQuestion(
                    question_id=record.question_id,
                    source=record.source,
                    question_text=record.question_text,
                    open_date=record.open_date,
                    close_date=record.close_date,
                    resolution=record.resolution,
                    market_price_at_open=opening_candidate,
                    category=record.category,
                )

            corpus.append(record)
            seen_ids.add(record.question_id)

            if len(corpus) >= max_questions:
                break

        if len(corpus) >= max_questions:
            break

        if exhausted:
            break

        cursor_before = explicit_last

        if cursor_before is None:
            break

    return corpus[:max_questions]


__all__ = [
    "BacktestQuestion",
    "GAMMA_MARKETS_URL",
    "MANIFOLD_MARKETS_URL",
    "build_manifold_corpus",
    "build_polymarket_corpus",
    "normalize_manifold_market",
    "normalize_polymarket_market",
]
