#!/usr/bin/env python3
"""Fetch resolved Polymarket markets from The Graph subgraph.

The Polymarket CLOB subgraph tracks all markets on-chain with proper filtering.
Endpoint: https://api.studio.thegraph.com/query/{id}/polymarket-clob/version/latest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
OUTPUT = TESTBED / "data" / "polymarket" / "mechanism_calibration.jsonl"

# Polymarket subgraph endpoint (CLOB — the active/live subgraph)
# Alternative: the older polymarket-matic subgraph
SUBGRAPH_ENDPOINTS = [
    "https://api.studio.thegraph.com/query/83978/polymarket-clob/version/latest",
    "https://api.studio.thegraph.com/query/83978/polymarket-clob/v0.0.1",
    "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/...",  # needs key
]

# Fallback: direct subgraph ID if studio endpoint doesn't work
SUBGRAPH_IDS = [
    "polymarket/polymarket-clob",  # common naming convention
    "polymarket/polymarket-matic",  # legacy
]

FAMILY_PATTERNS = {
    "structural-lock-in": [
        (r"(win|most seats|holds?.+majority).+(election|presidential|midterm)", "FPTP front-runner"),
        (r"(announce|declares?.+(ceasefire|truce))", "ceasefire announcement"),
        (r"(begins? trading|launch|lists? on).+(ETF|exchange)", "ETF listing"),
        (r"(uphold|strike down|enjoin|block).+(law|ban|executive order)", "court ruling on law"),
        (r"(confirm|approve).+(nominee|justice|judge|appointment)", "confirmation lock-in"),
    ],
    "structural-ceiling": [
        (r"(third.?party|minor.?party|independent).+(win|majority|most seats)", "third-party ceiling"),
        (r"(never|less than 1%|under 1%).+(win|elected)", "near-zero probability"),
        (r"(nuclear|nuke).+(detonat|use|strike|launch)", "nuclear use"),
        (r"(shutdown|default|debt ceiling).+(government|federal)", "shutdown ceiling"),
    ],
    "discretionary": [
        (r"(Fed|Federal Reserve).+(increase|decrease|cut|hike|hold).+(rate|bps)", "Fed decision"),
        (r"(win|elected).+(president|senate|congress|governor|primary)", "election outcome"),
        (r"(VP|vice president).+(nominee|selected|chosen)", "VP selection"),
        (r"(ceasefire|peace|truce|negotiation).+(Israel|Hamas|Ukraine|Russia|Gaza)", "ceasefire"),
        (r"(drop.?out|withdraw|resign|step.?down).+(president|candidate)", "withdrawal"),
    ],
}

EXCLUDE = re.compile(
    r"(price of|above \$|below \$|dip to|BTC|ETH|SOL|XRP|NFT)"
    r"|(NBA|NFL|NHL|MLB|UFC|NCAAB|F1|PGA|Super Bowl|World Cup|Wimbledon|Masters)"
    r"|(Will [A-Z][a-z]+ say|Will [A-Z][a-z]+ mention|Will [A-Z][a-z]+ wear)"
    r"|(Games Total|Map Handicap|Counter-Strike|Moneyline|O/U|Spread:)"
    r"|(Completed Match|Total Kills|Coin Toss)",
    re.IGNORECASE
)


def graphql_query(endpoint: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against a subgraph endpoint."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "psychohistory/0.1"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def discover_endpoint() -> str | None:
    """Try known endpoints and subgraph IDs to find a working one."""
    # Simple introspection query
    test_query = "{ _meta { block { number } } }"

    # Try studio endpoints first
    for ep in SUBGRAPH_ENDPOINTS:
        try:
            result = graphql_query(ep, test_query)
            block = result.get("data", {}).get("_meta", {}).get("block", {}).get("number")
            if block:
                print(f"  Found working endpoint: {ep} (block {block})")
                return ep
        except Exception as e:
            print(f"  {ep[:50]}... → {str(e)[:60]}")

    # Try decentralized network endpoints
    for subgraph_id in SUBGRAPH_IDS:
        ep = f"https://api.thegraph.com/subgraphs/name/{subgraph_id}"
        try:
            result = graphql_query(ep, test_query)
            if result.get("data"):
                print(f"  Found: {ep}")
                return ep
        except Exception:
            pass

    return None


def fetch_resolved_markets(endpoint: str, limit: int = 500) -> list[dict]:
    """Fetch closed markets with resolution from subgraph."""
    query = """
    query($first: Int!, $skip: Int!) {
      conditions(first: $first, skip: $skip, 
        where: { 
          resolved: true,
        },
        orderBy: resolutionTimestamp,
        orderDirection: desc
      ) {
        id
        question
        outcomes
        payouts
        resolutionTimestamp
        creationTimestamp
        category
      }
    }
    """

    markets = []
    skip = 0
    batch_size = min(100, limit)

    while len(markets) < limit:
        try:
            result = graphql_query(endpoint, query, {"first": batch_size, "skip": skip})
            conditions = result.get("data", {}).get("conditions", [])
        except Exception as e:
            print(f"  Error at skip={skip}: {e}")
            break

        if not conditions:
            break

        markets.extend(conditions)
        skip += batch_size
        print(f"  Fetched {len(markets)} resolved markets...")
        time.sleep(0.5)

    return markets[:limit]


def resolve_outcome(condition: dict) -> bool | None:
    """Determine resolution from subgraph condition data."""
    # payouts is an array of [payout_for_outcome_0, payout_for_outcome_1]
    # For binary: [1.0, 0.0] = YES, [0.0, 1.0] = NO
    payouts = condition.get("payouts", [])
    if payouts and len(payouts) >= 2:
        payout_0 = float(payouts[0])
        if payout_0 >= 0.99:
            return True  # YES won
        elif payout_0 <= 0.01:
            return False  # NO won

    # Check outcomes directly
    outcomes = condition.get("outcomes", [])
    if outcomes:
        return outcomes[0].lower() if isinstance(outcomes[0], str) else None

    return None


def classify(question: str) -> tuple[str | None, str]:
    q = question.lower()
    for family in ["structural-lock-in", "structural-ceiling", "discretionary"]:
        for pat, subcat in FAMILY_PATTERNS[family]:
            if re.search(pat, q):
                return family, subcat
    return None, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== Finding Polymarket subgraph endpoint ===\n")
    endpoint = discover_endpoint()

    if not endpoint:
        print("\n❌ No working subgraph endpoint found.")
        print("   Options:")
        print("   1. Get an API key from https://thegraph.com/studio/")
        print("   2. Use the decentralized network: https://api.thegraph.com/subgraphs/name/polymarket/polymarket-clob")
        print("   3. Try: https://gateway.thegraph.com/api/[key]/subgraphs/id/...")
        return 1

    print(f"\n=== Fetching resolved markets ===\n")
    conditions = fetch_resolved_markets(endpoint, args.limit)
    print(f"Fetched {len(conditions)} total conditions")

    # Filter and classify
    records = []
    stats = defaultdict(lambda: {"found": 0, "classified": 0})

    for c in conditions:
        q = c.get("question", "")
        if not q or EXCLUDE.search(q):
            continue

        resolution = resolve_outcome(c)
        if resolution is None:
            continue

        family, subcat = classify(q)
        if not family:
            continue

        records.append({
            "question": q,
            "resolution": resolution,
            "family": family,
            "subcategory": subcat,
            "end_date": c.get("resolutionTimestamp", ""),
            "category": c.get("category", ""),
        })
        stats[family]["classified"] += 1

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} mechanism-relevant markets to {OUTPUT}")
    print(f"\nBy family:")
    for family in ["structural-lock-in", "structural-ceiling", "discretionary"]:
        yes = sum(1 for r in records if r["family"] == family and r["resolution"])
        no = sum(1 for r in records if r["family"] == family and not r["resolution"])
        total = yes + no
        if total > 0:
            alpha = 1 + yes
            beta = 1 + no
            post = alpha / (alpha + beta)
            print(f"  {family}: {total} ({yes}Y/{no}N) → posterior {post:.3f}")

    if args.dry_run:
        return 0

    # Build calibration tables
    print(f"\n=== Building calibration tables ===\n")
    calib = defaultdict(lambda: defaultdict(lambda: {"yes": 0, "no": 0}))
    for rec in records:
        if rec["resolution"]:
            calib[rec["family"]][rec["subcategory"]]["yes"] += 1
        else:
            calib[rec["family"]][rec["subcategory"]]["no"] += 1

    for family in ["structural-lock-in", "structural-ceiling", "discretionary"]:
        subs = calib[family]
        total_yes = sum(s["yes"] for s in subs.values())
        total_no = sum(s["no"] for s in subs.values())
        total = total_yes + total_no
        if total == 0:
            continue

        alpha = 1 + total_yes
        beta = 1 + total_no
        post = alpha / (alpha + beta)

        print(f"## {family} (pooled: {total} markets, posterior {post:.3f})")
        for subcat in sorted(subs.keys()):
            s = subs[subcat]
            n = s["yes"] + s["no"]
            if n == 0: continue
            a, b = 1 + s["yes"], 1 + s["no"]
            print(f"   {subcat}: {s['yes']}Y/{s['no']}N ({n}) → posterior {a/(a+b):.3f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
