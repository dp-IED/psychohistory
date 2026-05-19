"""Fetch Polymarket YES-token price at a point in time (CLOB prices-history)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_USER_AGENT = "Mozilla/5.0 (compatible; psychohistory-polymarket-price/0.1)"


def _http_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_json_list(raw: object) -> list[object]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return []


def yes_token_id_from_market(market: dict[str, object]) -> str | None:
    outcomes = [str(x).lower() for x in _parse_json_list(market.get("outcomes"))]
    tokens = [str(x) for x in _parse_json_list(market.get("clobTokenIds"))]
    if len(outcomes) < 1 or len(tokens) < 1:
        return None
    if outcomes[0] == "yes":
        return tokens[0]
    if len(outcomes) > 1 and outcomes[1] == "yes":
        return tokens[1]
    return tokens[0]


def fetch_market_by_slug(slug: str) -> dict[str, object] | None:
    url = f"{GAMMA_MARKETS_URL}?{urllib.parse.urlencode({'slug': slug})}"
    payload = _http_json(url)
    if not isinstance(payload, list) or not payload:
        return None
    row = payload[0]
    return row if isinstance(row, dict) else None


def fetch_price_history(token_id: str, *, fidelity_minutes: int = 1440) -> list[tuple[int, float]]:
    params = urllib.parse.urlencode(
        {"market": token_id, "interval": "max", "fidelity": str(fidelity_minutes)}
    )
    payload = _http_json(f"{CLOB_PRICES_URL}?{params}")
    if not isinstance(payload, dict):
        return []
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    out: list[tuple[int, float]] = []
    for point in history:
        if not isinstance(point, dict):
            continue
        try:
            out.append((int(point["t"]), float(point["p"])))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def cutoff_to_unix(cutoff: date) -> int:
    dt = datetime(cutoff.year, cutoff.month, cutoff.day, 23, 59, 59, tzinfo=timezone.utc)
    return int(dt.timestamp())


def price_at_timestamp(
    history: list[tuple[int, float]],
    *,
    as_of_unix: int,
) -> float | None:
    """Last trade price at or before ``as_of_unix``."""
    if not history:
        return None
    best: tuple[int, float] | None = None
    for ts, price in history:
        if ts <= as_of_unix:
            best = (ts, price)
        else:
            break
    return best[1] if best else None


def yes_price_at_cutoff(
    *,
    token_id: str | None = None,
    slug: str | None = None,
    cutoff: date,
    fidelity_minutes: int = 1440,
) -> tuple[float | None, str | None]:
    """Return (yes_price, token_id_used)."""
    tid = token_id
    if tid is None and slug:
        market = fetch_market_by_slug(slug)
        if market is None:
            return None, None
        tid = yes_token_id_from_market(market)
    if not tid:
        return None, None
    history = fetch_price_history(tid, fidelity_minutes=fidelity_minutes)
    price = price_at_timestamp(history, as_of_unix=cutoff_to_unix(cutoff))
    return price, tid


def yes_price_at_cutoff_with_retry(
    *,
    token_id: str | None = None,
    slug: str | None = None,
    cutoff: date,
    retries: int = 2,
    sleep_s: float = 0.15,
) -> tuple[float | None, str | None]:
    last: tuple[float | None, str | None] = (None, None)
    for attempt in range(retries + 1):
        last = yes_price_at_cutoff(token_id=token_id, slug=slug, cutoff=cutoff)
        if last[0] is not None:
            return last
        if attempt < retries:
            time.sleep(sleep_s)
    return last
