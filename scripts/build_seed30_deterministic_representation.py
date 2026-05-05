#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

UTC = dt.timezone.utc


def parse_ts(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        return t
    except Exception:
        return None


def to_z(ts: dt.datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def compact_event_text(row: dict[str, Any]) -> str:
    keys = [
        "question_text",
        "event_title",
        "actor1_name",
        "actor2_name",
        "raw_action_geo_country",
        "sourceurl",
        "series_id",
        "metric",
        "period",
        "award",
        "category",
        "film_title",
        "work_title",
        "artist_name",
        "person_name",
        "nominee_text",
        "chart_name",
        "song",
        "artist",
    ]
    return " | ".join(str(row.get(k, "")) for k in keys)


def source_timestamp(row: dict[str, Any]) -> dt.datetime | None:
    for k in ("publication_time", "release_time", "observed_time", "event_time"):
        t = parse_ts(row.get(k))
        if t:
            return t
    # GDELT event_time is yyyymmdd
    et = str(row.get("event_time", ""))
    if re.fullmatch(r"\d{8}", et):
        try:
            return dt.datetime.strptime(et, "%Y%m%d").replace(tzinfo=UTC)
        except Exception:
            return None
    return None


def extract_keywords(query: dict[str, Any]) -> list[str]:
    kw: set[str] = set()
    for src in [query.get("missing_critical_entities", ""), query.get("question_text", ""), query.get("event_title", "")]:
        for chunk in re.split(r"[,;]", str(src)):
            c = chunk.strip().strip("\"'“”")
            if len(c) >= 3:
                kw.add(c.lower())
    # normalize known alias
    if "joker 2" in kw:
        kw.add("joker: folie à deux")
    if any("inside out 2" in k for k in kw):
        kw.add("inside out 2")
    return sorted(kw)


def matches_keywords(text: str, keywords: list[str]) -> bool:
    t = normalize_text(text)
    return any(k in t for k in keywords)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_gdelt(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def make_ref_id(tier: str, idx: int) -> str:
    return f"{tier}:{idx}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=".context/polymarket_30_seed_coverage_audit.json")
    ap.add_argument("--wikidata", default=".context/seed30_wikidata_backbone_entities.json")
    ap.add_argument("--gdelt", default="data/gdelt/raw/seed_global_news/seed_global_gdelt.jsonl.gz")
    ap.add_argument("--macro", default="data/macro/raw/seed_official_macro/seed_official_macro_events.jsonl")
    ap.add_argument("--culture", default="data/culture/raw/seed_culture_tier/seed_culture_events.jsonl")
    ap.add_argument("--out-dir", default="data/representations/seed30/day4_deterministic")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    root = Path.cwd()
    seed = json.loads((root / args.seed).read_text(encoding="utf-8"))
    wikidata_obj = json.loads((root / args.wikidata).read_text(encoding="utf-8"))
    wikidata_entities = wikidata_obj.get("entities", [])
    gdelt_rows = load_gdelt(root / args.gdelt)
    macro_rows = load_jsonl(root / args.macro)
    culture_rows = load_jsonl(root / args.culture)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_path = out_dir / "seed30_deterministic_representations.jsonl"
    metrics_path = out_dir / "axis1_metrics.jsonl"
    summary_path = out_dir / "axis1_summary.json"

    reps: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    # Pre-index by domain for cheap filtering
    wd_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in wikidata_entities:
        for d in e.get("domains", []):
            wd_by_domain[d].append(e)

    macro_by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in macro_rows:
        macro_by_metric[m.get("metric", "unknown")].append(m)

    for q in seed:
        qid = q["query_id"]
        domain = q["domain"]
        cutoff = parse_ts(q.get("cutoff_t"))
        keywords = extract_keywords(q)

        refs: list[dict[str, Any]] = []

        # Tier 1: Wikidata
        for i, e in enumerate(wd_by_domain.get(domain, [])):
            label = str(e.get("entity_label", ""))
            if matches_keywords(label, keywords) or matches_keywords(str(e.get("wikidata_label", "")), keywords):
                refs.append(
                    {
                        "ref_id": make_ref_id("wikidata", i),
                        "tier": "wikipedia_wikidata",
                        "timestamp": None,
                        "qid": e.get("qid"),
                        "snippet": f"{e.get('entity_label')} -> {e.get('wikidata_label')} ({e.get('qid')})",
                        "source": "seed30_wikidata_backbone_entities.json",
                    }
                )

        # Tier 2: GDELT/news (bounded lexical retrieve)
        gdelt_hits = 0
        for i, r in enumerate(gdelt_rows):
            text = compact_event_text(r)
            if matches_keywords(text, keywords):
                ts = source_timestamp(r)
                if cutoff and ts and ts > cutoff:
                    continue
                refs.append(
                    {
                        "ref_id": make_ref_id("gdelt", i),
                        "tier": "gdelt_global_news",
                        "timestamp": to_z(ts),
                        "snippet": text[:240],
                        "source": r.get("source_url"),
                    }
                )
                gdelt_hits += 1
                if gdelt_hits >= 8:
                    break

        # Tier 3: domain-specific official/culture
        if domain == "economics":
            # most recent <= cutoff per metric
            for metric, rows in macro_by_metric.items():
                best = None
                best_ts = None
                for r in rows:
                    ts = source_timestamp(r)
                    if not ts:
                        continue
                    if cutoff and ts > cutoff:
                        continue
                    if best_ts is None or ts > best_ts:
                        best, best_ts = r, ts
                if best:
                    refs.append(
                        {
                            "ref_id": f"macro:{metric}",
                            "tier": "official_macro",
                            "timestamp": to_z(best_ts),
                            "snippet": f"{best.get('metric')}={best.get('value')} ({best.get('period')})",
                            "source": best.get("source_name"),
                        }
                    )
        if domain == "culture":
            c_hits = 0
            for i, r in enumerate(culture_rows):
                text = compact_event_text(r)
                if matches_keywords(text, keywords):
                    ts = source_timestamp(r)
                    if cutoff and ts and ts > cutoff:
                        continue
                    refs.append(
                        {
                            "ref_id": make_ref_id("culture", i),
                            "tier": "awards_boxoffice_chart",
                            "timestamp": to_z(ts),
                            "snippet": text[:240],
                            "source": r.get("source_url"),
                        }
                    )
                    c_hits += 1
                    if c_hits >= 10:
                        break

        # Keep deterministic top-k by tier priority then timestamp recency
        tier_rank = {
            "wikipedia_wikidata": 0,
            "official_macro": 1,
            "awards_boxoffice_chart": 1,
            "gdelt_global_news": 2,
        }

        def sort_key(x: dict[str, Any]) -> tuple[int, str]:
            return (tier_rank.get(x.get("tier", "gdelt_global_news"), 9), x.get("timestamp") or "")

        refs = sorted(refs, key=sort_key)[: args.k]

        tier_counts: dict[str, int] = defaultdict(int)
        for r in refs:
            tier_counts[r["tier"]] += 1

        missing_tiers = []
        if tier_counts.get("wikipedia_wikidata", 0) == 0:
            missing_tiers.append("slow_backbone_missing")
        if tier_counts.get("gdelt_global_news", 0) == 0:
            missing_tiers.append("fast_signal_missing")
        if domain == "economics" and tier_counts.get("official_macro", 0) == 0:
            missing_tiers.append("official_macro_missing")
        if domain == "culture" and tier_counts.get("awards_boxoffice_chart", 0) == 0:
            missing_tiers.append("culture_tier_missing")

        assumptions = []
        if missing_tiers:
            assumptions.append("Some required source tiers absent at cutoff; probability forecasts should widen uncertainty.")

        rep = {
            "query_id": qid,
            "cutoff_t": q.get("cutoff_t"),
            "question_text": q.get("question_text"),
            "market_source": q.get("polymarket_url"),
            "resolution_source": q.get("resolution_source"),
            "schema_fields": [
                "outcome_hypothesis",
                "key_entities",
                "supporting_evidence",
                "counter_evidence",
                "uncertainty_drivers",
            ],
            "evidence_refs": refs,
            "assumption_states": assumptions,
            "coverage_flags": missing_tiers,
            "reasoning_trace_id": f"trace_stub::{qid}",
            "forecast_output_id": f"forecast_stub::{qid}",
        }
        reps.append(rep)

        # Axis 1 metric proxies
        gold_terms = [t.strip() for t in str(q.get("missing_critical_entities", "")).split(",") if t.strip()]
        gold_l = [g.lower() for g in gold_terms]
        covered = 0
        concat_refs = normalize_text(" ".join(r.get("snippet", "") for r in refs))
        for g in gold_l:
            if g in concat_refs:
                covered += 1
        evidence_recall = covered / len(gold_l) if gold_l else 0.0

        stamped = [r for r in refs if r.get("timestamp")]
        if stamped and cutoff:
            temporal_precision = sum(1 for r in stamped if parse_ts(r["timestamp"]) and parse_ts(r["timestamp"]) <= cutoff) / len(stamped)
        else:
            temporal_precision = 1.0

        query_specificity = min(1.0, len(refs) / max(1, args.k))
        assumption_coverage = 1.0 if (missing_tiers and assumptions) or (not missing_tiers) else 0.0

        metrics.append(
            {
                "query_id": qid,
                "domain": domain,
                "evidence_recall_at_k": round(evidence_recall, 4),
                "temporal_precision": round(temporal_precision, 4),
                "query_specificity": round(query_specificity, 4),
                "assumption_coverage": round(assumption_coverage, 4),
                "evidence_ref_count": len(refs),
                "tier_counts": dict(tier_counts),
                "coverage_flags": missing_tiers,
            }
        )

    with rep_path.open("w", encoding="utf-8") as f:
        for r in reps:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with metrics_path.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        by_domain[m["domain"]].append(m)

    summary: dict[str, Any] = {
        "created_at": to_z(dt.datetime.now(tz=UTC)),
        "queries": len(metrics),
        "out_dir": str(out_dir.resolve()),
        "domain_means": {},
    }
    for d, rows in by_domain.items():
        def avg(k: str) -> float:
            return round(sum(float(x[k]) for x in rows) / max(1, len(rows)), 4)

        summary["domain_means"][d] = {
            "evidence_recall_at_k": avg("evidence_recall_at_k"),
            "temporal_precision": avg("temporal_precision"),
            "query_specificity": avg("query_specificity"),
            "assumption_coverage": avg("assumption_coverage"),
            "mean_evidence_ref_count": round(sum(int(x["evidence_ref_count"]) for x in rows) / len(rows), 2),
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
