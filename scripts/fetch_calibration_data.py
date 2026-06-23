#!/usr/bin/env python3
"""Fetch all resolved Polymarket markets via pmxt.

Dumps every resolved non-sports binary market with its tags — no regex
classification.  The tag-based calibration model consumes this output;
the agent maps novel questions to tag clusters at query time.

Formerly used The Graph subgraph + regex mechanism families.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import pmxt

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
OUTPUT = TESTBED / "data" / "polymarket" / "resolved_markets.jsonl"

# ── Exclusion filters ──────────────────────────────────────────────────
# Goal: keep every market that could inform political/economic/conflict
# calibration.  Drop obvious sports, crypto prices, meme bets.

EXCLUDE_TITLE = re.compile(
    r"(price of|above \$|below \$|dip to|BTC|ETH|SOL|XRP|NFT)"
    r"|(NBA|NFL|NHL|MLB|UFC|NCAAB|F1|PGA|Super Bowl|World Cup|Wimbledon|Masters)"
    r"|(Will [A-Z][a-z]+ say|Will [A-Z][a-z]+ mention|Will [A-Z][a-z]+ wear)"
    r"|(Games Total|Map Handicap|Counter-Strike|Moneyline|O/U|Spread:)"
    r"|(Completed Match|Total Kills|Coin Toss|Anytime TD|Receiving Yards)"
    r"|(Champions League|UEFA|Premier League|La Liga|Bundesliga|Serie A|Ligue 1)"
    r"|(Stanley Cup|Grand Prix|Tour de France|Daytona|Indy 500)",
    re.IGNORECASE,
)

EXCLUDE_TAGS = {
    "Sports", "Soccer", "Basketball", "Baseball", "Football",
    "Tennis", "Golf", "MMA", "Boxing", "Cricket", "Rugby",
    "Hockey", "NASCAR", "Formula 1", "Olympics",
}

# Tags that are structural/metadata, not topical
META_TAGS = {
    "All", "Recurring", "Monthly", "Parent For Derivative",
    "Hit Price", "Best of 2025",
}

PAGE_SIZE = 100
FETCH_DELAY = 0.10


def is_sports(market: pmxt.models.UnifiedMarket) -> bool:
    if market.tags and EXCLUDE_TAGS.intersection(market.tags):
        return True
    if EXCLUDE_TITLE.search(market.title):
        return True
    return False


def clean_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    return sorted(t for t in tags if t not in META_TAGS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch all resolved Polymarket markets via pmxt"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max records (0 = fetch all available)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== Fetching resolved Polymarket markets via pmxt ===\n")
    client = pmxt.Polymarket()

    records: list[dict] = []
    seen_ids: set[str] = set()
    offset = 0
    empty_pages = 0

    while args.limit == 0 or len(records) < args.limit:
        try:
            markets = client.fetch_markets(params={
                "status": "closed",
                "limit": PAGE_SIZE,
                "offset": offset,
            })
        except Exception as e:
            print(f"  Error at offset={offset}: {e}")
            break

        if not markets:
            break

        new_in_page = 0
        for m in markets:
            if m.market_id in seen_ids:
                continue
            seen_ids.add(m.market_id)

            if is_sports(m):
                continue

            resolution = _resolve_outcome(m)
            if resolution is None:
                continue

            end_date = ""
            if m.resolution_date:
                end_date = (m.resolution_date.isoformat()
                            if hasattr(m.resolution_date, "isoformat")
                            else str(m.resolution_date))

            records.append({
                "question": m.title,
                "resolution": resolution,
                "end_date": end_date,
                "tags": clean_tags(m.tags),
                "slug": m.slug or "",
                "market_id": m.market_id,
                "volume": m.volume or 0,
            })
            new_in_page += 1

        offset += PAGE_SIZE
        pct = (len(records) / max(offset, 1)) * 100

        if new_in_page == 0:
            empty_pages += 1
            if empty_pages >= 3:
                print(f"  {offset:>5} markets checked, {len(records):>4} kept "
                      f"({pct:.0f}%) — 3 empty pages, stopping.")
                break
            print(f"  {offset:>5} markets checked, {len(records):>4} kept "
                  f"({pct:.0f}%)")
        else:
            empty_pages = 0
            print(f"  {offset:>5} markets checked, {len(records):>4} kept "
                  f"({pct:.0f}%)  +{new_in_page}")

        time.sleep(FETCH_DELAY)

    # ── Write ──────────────────────────────────────────────────────────
    if not args.dry_run:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nWrote {len(records)} records to {OUTPUT}")

    # ── Summary ────────────────────────────────────────────────────────
    yes = sum(1 for r in records if r["resolution"])
    no = len(records) - yes
    print(f"\nResolved: {yes} YES  /  {no} NO  ({yes/(yes+no)*100:.1f}% YES)")

    # Tag distribution
    tag_counts = Counter()
    for r in records:
        for t in r["tags"]:
            tag_counts[t] += 1

    print(f"\nTop 30 tags:")
    for tag, count in tag_counts.most_common(30):
        # Show yes/no split per tag
        tag_yes = sum(1 for r in records
                      if tag in r["tags"] and r["resolution"])
        tag_total = sum(1 for r in records if tag in r["tags"])
        post = (1 + tag_yes) / (2 + tag_total)
        print(f"  {tag:<30} {tag_total:>4} markets  "
              f"posterior={post:.3f}")

    return 0


def _resolve_outcome(market: pmxt.models.UnifiedMarket) -> bool | None:
    yes = market.yes
    if yes is None:
        return None
    price = yes.price
    if price >= 0.99:
        return True
    if price <= 0.01:
        return False
    return None


if __name__ == "__main__":
    raise SystemExit(main())
