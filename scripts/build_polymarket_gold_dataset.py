#!/usr/bin/env python
"""Build a 30-case content-oriented Polymarket gold set with Wikipedia context.

The dataset is deliberately structural/content gold, not a terminal-leak oracle:
market outcomes are labels, while branch orientation terms are drawn from market
text plus current Wikipedia summaries for the event/entity surface form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

GAMMA = "https://gamma-api.polymarket.com/markets"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
UA = "Mozilla/5.0 (compatible; psychohistory-polymarket-gold/0.1)"

QUERY_TERMS = [
    "Ukraine Russia war", "Israel Gaza ceasefire", "Taiwan election", "Trump trial", "Biden nominee",
    "Federal Reserve rate cut", "CPI inflation", "recession", "government shutdown", "debt ceiling",
    "OpenAI", "Elon Musk Tesla", "TikTok ban", "Supreme Court", "NATO",
    "Iran nuclear", "Argentina election", "French election", "UK election", "Venezuela election",
    "Bitcoin ETF SEC", "Ethereum ETF SEC", "World Health Organization", "climate summit", "OPEC",
    "BRICS",    "China GDP", "US unemployment", "interest rates", "central bank", "Biden drop out", "Biden nominee",
    "Venezuela presidential election", "Argentina legislative election", "TikTok Supreme Court", "Israel Hamas ceasefire",
    "Trump sentencing", "Trump election interference", "SEC Ethereum ETF", "Bitcoin ETF approval", "government shutdown",
    "NATO Ukraine", "Russia Ukraine ceasefire", "Iran Israel", "French legislative election", "UK general election",
]

EXCLUDE = re.compile(
    r"\b(above|below|dip to|o/u|goalscorer|win on \d|pole\?|penta kill|destroy inhibitors|highest temperature|say [\"“]|tweet|tweets|price of bitcoin|price of ethereum|price of solana|nba|nfl|ufc|mlb|f1)\b",
    re.I,
)
FAMILY_RULES = [
    ("event_negotiation", re.compile(r"ceasefire|truce|hostage|deal|agreement|treaty|negotiation|peace|war|strike|summit|nato|iran|ukraine|gaza|russia|taiwan|hamas|israel", re.I), ["local", "analogue", "disruptor"]),
    ("institutional_process", re.compile(r"election|inaugurat|congress|senate|parliament|court|supreme court|bill |veto|impeach|nomination|confirmation|ballot|sec|ban|trial|shutdown|sentenced|sentencing|deputies|presidential race", re.I), ["local", "disruptor"]),
    ("macro_policy_print", re.compile(r"fed|fomc|rate cut|cut rates|interest rate|target range|cpi|inflation|unemployment|jobs report|gdp|recession|treasury|central bank|opec", re.I), ["local", "analogue"]),
]

STOP = {"will","there","before","after","during","this","that","with","from","into","have","been","being","market","resolves","resolution","criteria","shall","question","whether","price","above","below","yes","no","the","and","for","of","to","in","on","by","at","a","an"}


def get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable retry fallthrough")


def parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def resolved_binary(raw: dict[str, object]) -> dict[str, object] | None:
    outcomes = [str(x) for x in parse_json_list(raw.get("outcomes"))]
    prices = [float(x) for x in parse_json_list(raw.get("outcomePrices")) or []]
    if [o.lower() for o in outcomes] != ["yes", "no"] or {round(p, 6) for p in prices} != {0.0, 1.0}:
        return None
    question = str(raw.get("question") or "")
    desc = str(raw.get("description") or "")
    slug = str(raw.get("slug") or "")
    if not question or not slug or EXCLUDE.search(question):
        return None
    if len(question) < 35 or len(desc) < 80:
        return None
    winner = outcomes[0] if prices[0] > prices[1] else outcomes[1]
    volume = float(raw.get("volumeNum") or raw.get("volume") or 0)
    # Favor larger, less toy-like questions but avoid pure high-volume sports/crypto lines.
    hard_score = min(volume, 1_000_000) / 1_000_000 + min(len(desc), 2500) / 2500
    return {
        "id": str(raw.get("id") or ""),
        "slug": slug,
        "question": question,
        "description": desc,
        "category": raw.get("category"),
        "market_type": raw.get("marketType"),
        "condition_id": str(raw.get("conditionId") or ""),
        "outcomes": outcomes,
        "terminal_outcome_prices": prices,
        "resolved_outcome": winner,
        "volume": volume,
        "liquidity": float(raw.get("liquidityNum") or raw.get("liquidity") or 0),
        "start_date": raw.get("startDate"),
        "end_date": raw.get("endDate"),
        "closed_time": raw.get("closedTime"),
        "created_at": raw.get("createdAt"),
        "updated_at": raw.get("updatedAt"),
        "clob_token_ids": [str(x) for x in parse_json_list(raw.get("clobTokenIds"))],
        "url": f"https://polymarket.com/market/{slug}",
        "gamma_url": f"{GAMMA}?slug={urllib.parse.quote(slug)}",
        "event_slug": raw.get("eventSlug") or raw.get("event_slug") or "",
        "hardness_score": round(hard_score, 4),
    }


def infer_family(text: str) -> tuple[str, list[str]]:
    for family, rx, branches in FAMILY_RULES:
        if rx.search(text):
            return family, branches
    return "event_negotiation", ["local", "analogue", "disruptor"]


def wiki_context(query: str) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"action":"query","list":"search","srsearch":query,"format":"json","srlimit":"1"})
    try:
        search = get_json(f"{WIKI_SEARCH}?{params}")
    except urllib.error.HTTPError:
        return []
    out = []
    for item in search.get("query", {}).get("search", [])[:1]:
        title = item["title"]
        sparams = urllib.parse.urlencode({"action":"query","prop":"extracts","exintro":"1","explaintext":"1","titles":title,"format":"json","redirects":"1"})
        try:
            data = get_json(f"{WIKI_SEARCH}?{sparams}")
        except urllib.error.HTTPError:
            continue
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = str(page.get("extract") or "").strip().replace("\n", " ")
            if extract:
                out.append({"title": str(page.get("title") or title), "extract": extract[:1400], "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(str(page.get("title") or title).replace(" ", "_"))})
        time.sleep(1.0)
    return out


def keywords(text: str, n: int = 24) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]{3,}", text)
    counts = {}
    for w in words:
        lw = w.lower().strip("'-")
        if lw in STOP or len(lw) < 4:
            continue
        counts[lw] = counts.get(lw, 0) + 1
    ranked = sorted(counts, key=lambda k: (-counts[k], len(k), k))
    return ranked[:n]


def fetch_candidates() -> list[dict[str, object]]:
    seen = {}
    for term in QUERY_TERMS:
        # public-search has materially better semantic recall than /markets?q=.
        params = urllib.parse.urlencode({"q": term})
        try:
            payload = get_json(f"https://gamma-api.polymarket.com/public-search?{params}")
        except Exception:
            continue
        events = payload.get("events", []) if isinstance(payload, dict) else []
        for event in events:
            if not isinstance(event, dict) or not event.get("closed"):
                continue
            event_text = "\n\n".join(str(event.get(k) or "") for k in ("title", "description", "subtitle"))
            for raw in event.get("markets") or []:
                if not isinstance(raw, dict):
                    continue
                merged = dict(raw)
                # Keep full event text in the resolution text so the gold set has
                # richer, non-placeholder context from the actual Polymarket event.
                merged["description"] = "\n\n".join(
                    part for part in [str(raw.get("description") or ""), event_text] if part.strip()
                )
                merged.setdefault("category", event.get("category"))
                merged["eventSlug"] = event.get("slug") or event.get("id") or ""
                rec = resolved_binary(merged)
                if rec:
                    seen[rec["id"]] = rec
        time.sleep(0.5)
    return sorted(seen.values(), key=lambda r: (-float(r["hardness_score"]), str(r["question"])))


def semantic_requirements(family: str, branches: list[str]) -> dict[str, dict[str, object]]:
    """Gold obligations by semantic role/direction, not lexical overlap.

    These are branch-orientation contracts: a builder should expose the kinds of
    mechanisms a forecaster/GNN needs (drivers, blockers, gates, spoilers,
    analogues), while the full text/Wikipedia fields provide the human-readable
    event-specific content behind those obligations.
    """

    by_family = {
        "event_negotiation": {
            "local": {"roles": ["driver", "spoiler", "constraint", "signal"], "directions": ["FOR", "AGAINST", "MIXED"], "min_elements": 5, "orientation": "direct bargaining channel, local spoiler capacity, and deadline reachability"},
            "analogue": {"roles": ["signal", "constraint", "driver"], "directions": ["MIXED", "AGAINST"], "min_elements": 4, "orientation": "prior comparable deals and failed-deal analogues"},
            "disruptor": {"roles": ["spoiler", "constraint", "signal"], "directions": ["AGAINST", "MIXED"], "min_elements": 4, "orientation": "external escalation, sponsor vetoes, and domestic constraints"},
        },
        "institutional_process": {
            "local": {"roles": ["institutional_gate", "constraint", "signal"], "directions": ["FOR", "AGAINST", "MIXED"], "min_elements": 5, "orientation": "formal authority, procedural completion, legal blockers, and deadline pressure"},
            "disruptor": {"roles": ["spoiler", "constraint", "signal"], "directions": ["AGAINST", "MIXED"], "min_elements": 4, "orientation": "agenda shocks, coalition fractures, and late procedural disruption"},
        },
        "macro_policy_print": {
            "local": {"roles": ["signal", "constraint", "institutional_gate"], "directions": ["FOR", "AGAINST", "MIXED"], "min_elements": 5, "orientation": "indicator path, official communication, release-calendar gate, and cutoff safety"},
            "analogue": {"roles": ["driver", "constraint", "signal"], "directions": ["MIXED", "AGAINST"], "min_elements": 4, "orientation": "historical regime comparison plus methodology/revision risk"},
        },
    }
    return {branch: by_family.get(family, by_family["event_negotiation"])[branch] for branch in branches}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/polymarket/gold_branch_dataset.json"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args(argv)
    candidates = fetch_candidates()
    cases = []
    used_questions = set()
    event_counts = {}
    for rec in candidates:
        event_key = str(rec.get("event_slug") or rec["slug"]).lower()
        if event_counts.get(event_key, 0) >= 2:
            continue
        qkey = re.sub(r"\W+", " ", str(rec["question"]).lower()).strip()[:90]
        if qkey in used_questions:
            continue
        full_text_seed = f"{rec['question']}\n\nPolymarket resolution text:\n{rec['description']}"
        wctx = wiki_context(str(rec["question"]))
        full_context = full_text_seed + "\n\nWikipedia current context:\n" + "\n\n".join(f"{w['title']}: {w['extract']}" for w in wctx)
        family, branches = infer_family(full_text_seed)
        cases.append({
            "case_id": f"gold_{len(cases)+1:02d}_{rec['slug'][:48].strip('-')}",
            "record": {k: rec[k] for k in ["id","slug","question","description","category","market_type","condition_id","outcomes","terminal_outcome_prices","resolved_outcome","volume","liquidity","start_date","end_date","closed_time","created_at","updated_at","clob_token_ids","url","gamma_url"]},
            "selection_notes": "Resolved binary market selected for non-trivial text, non-sports/price-line form, volume/text hardness, and Wikipedia-enrichable context.",
            "hardness_score": rec["hardness_score"],
            "wikipedia_context_as_of": "2026-05-09",
            "wikipedia_context": wctx,
            "full_text": full_context[:6000],
            "expected_family": family,
            "expected_required_branches": branches,
            "expected_target_value": 1.0 if str(rec["resolved_outcome"]).lower() == "yes" else 0.0,
            "expected_min_prerequisites": 3 if family == "macro_policy_print" else 4,
            "expected_node_types": ["market", "outcome_hypothesis", "branch", "portfolio_element", "prerequisite"],
            "gold_orientation": {
                "evaluation_mode": "semantic_role_direction_contract_not_word_overlap",
                "semantic_requirements": semantic_requirements(family, branches),
                "branch_orientation_notes": {b: semantic_requirements(family, branches)[b]["orientation"] for b in branches},
                "expressiveness_min_elements": 8,
                "expressiveness_min_distinct_roles": 4,
                "expressiveness_min_rationale_chars": 500,
            },
        })
        used_questions.add(qkey)
        event_counts[event_key] = event_counts.get(event_key, 0) + 1
        if len(cases) >= args.limit:
            break
    if len(cases) < args.limit:
        raise SystemExit(f"only built {len(cases)} cases")
    payload = {"description":"30-case content gold standard for resolved Polymarket branch construction. Full text uses Polymarket resolution text plus current Wikipedia summaries as of 2026-05-09; terminal outcomes are labels only.", "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)+"\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
