"""Fetch resolved binary Polymarket metadata from the public Gamma API.

The fetcher intentionally stores market metadata and terminal resolution only.
It does not treat market text as PIT evidence for a forecast cutoff; downstream
harness runs must attach PIT-safe evidence separately.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
USER_AGENT = "Mozilla/5.0 (compatible; psychohistory-polymarket-fetcher/0.1)"


@dataclass(frozen=True)
class ResolvedMarketRecord:
    id: str
    slug: str
    question: str
    description: str
    category: str | None
    market_type: str | None
    condition_id: str
    outcomes: list[str]
    terminal_outcome_prices: list[float]
    resolved_outcome: str
    volume: float | None
    liquidity: float | None
    start_date: str | None
    end_date: str | None
    closed_time: str | None
    created_at: str | None
    updated_at: str | None
    clob_token_ids: list[str]
    url: str
    gamma_url: str


def _load_json_field(value: object) -> list[object]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _resolved_outcome(outcomes: list[str], prices: list[float]) -> str | None:
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    if {round(p, 6) for p in prices} != {0.0, 1.0}:
        return None
    winner_idx = 0 if prices[0] > prices[1] else 1
    return outcomes[winner_idx]


def normalize_market(raw: dict[str, object]) -> ResolvedMarketRecord | None:
    outcomes = [str(x) for x in _load_json_field(raw.get("outcomes"))]
    prices_raw = _load_json_field(raw.get("outcomePrices"))
    try:
        prices = [float(x) for x in prices_raw]
    except (TypeError, ValueError):
        return None
    clob_token_ids = [str(x) for x in _load_json_field(raw.get("clobTokenIds"))]
    resolved = _resolved_outcome(outcomes, prices)
    if resolved is None:
        return None
    if tuple(o.lower() for o in outcomes) != ("yes", "no"):
        return None
    market_id = str(raw.get("id") or "")
    slug = str(raw.get("slug") or "")
    if not market_id or not slug:
        return None
    return ResolvedMarketRecord(
        id=market_id,
        slug=slug,
        question=str(raw.get("question") or ""),
        description=str(raw.get("description") or ""),
        category=str(raw.get("category")) if raw.get("category") is not None else None,
        market_type=str(raw.get("marketType")) if raw.get("marketType") is not None else None,
        condition_id=str(raw.get("conditionId") or ""),
        outcomes=outcomes,
        terminal_outcome_prices=prices,
        resolved_outcome=resolved,
        volume=_float_or_none(raw.get("volumeNum", raw.get("volume"))),
        liquidity=_float_or_none(raw.get("liquidityNum", raw.get("liquidity"))),
        start_date=str(raw.get("startDate")) if raw.get("startDate") else None,
        end_date=str(raw.get("endDate")) if raw.get("endDate") else None,
        closed_time=str(raw.get("closedTime")) if raw.get("closedTime") else None,
        created_at=str(raw.get("createdAt")) if raw.get("createdAt") else None,
        updated_at=str(raw.get("updatedAt")) if raw.get("updatedAt") else None,
        clob_token_ids=clob_token_ids,
        url=f"https://polymarket.com/market/{slug}",
        gamma_url=f"{GAMMA_MARKETS_URL}?slug={urllib.parse.quote(slug)}",
    )


def fetch_closed_markets(*, limit: int, page_size: int = 100, sleep_s: float = 0.05) -> list[ResolvedMarketRecord]:
    records: list[ResolvedMarketRecord] = []
    seen: set[str] = set()
    offset = 0
    while len(records) < limit:
        params = {
            "limit": str(min(page_size, max(limit - len(records), 1))),
            "offset": str(offset),
            "closed": "true",
            "order": "volume",
            "ascending": "false",
        }
        url = f"{GAMMA_MARKETS_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        if not isinstance(payload, list) or not payload:
            break
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            record = normalize_market(raw)
            if record is None or record.id in seen:
                continue
            records.append(record)
            seen.add(record.id)
            if len(records) >= limit:
                break
        offset += len(payload)
        time.sleep(sleep_s)
    return records


def write_json(records: list[ResolvedMarketRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": GAMMA_MARKETS_URL,
        "description": "Resolved binary Yes/No Polymarket metadata from Gamma API, sorted by volume.",
        "count": len(records),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(records: list[ResolvedMarketRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "slug",
        "question",
        "category",
        "resolved_outcome",
        "volume",
        "end_date",
        "closed_time",
        "condition_id",
        "url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({key: row.get(key) for key in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json-output", type=Path, default=Path("data/polymarket/resolved_binary_markets.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("data/polymarket/resolved_binary_markets.csv"))
    args = parser.parse_args(argv)

    records = fetch_closed_markets(limit=args.limit)
    write_json(records, args.json_output)
    write_csv(records, args.csv_output)
    print(f"wrote {len(records)} resolved binary markets")
    print(f"json: {args.json_output}")
    print(f"csv:  {args.csv_output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
