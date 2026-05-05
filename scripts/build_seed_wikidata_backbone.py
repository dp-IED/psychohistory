#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.wikidata_linking import normalize_entity_label, search_wikidata_entity, qid_from_value

REPO = Path(__file__).resolve().parents[1]
SEED_PATH = REPO / ".context" / "polymarket_30_seed_coverage_audit.json"
OUT_JSON = REPO / ".context" / "seed30_wikidata_backbone_entities.json"
OUT_CSV = REPO / ".context" / "seed30_wikidata_backbone_entities.csv"
OUT_CHECKPOINT_MD = REPO / ".context" / "seed30_priority1_checkpoint.md"

_SPLIT = re.compile(r"\s*,\s*")


def parse_entities(row: dict[str, Any]) -> list[str]:
    raw = row.get("missing_critical_entities")
    if not isinstance(raw, str) or not raw.strip():
        return []
    vals = [v.strip() for v in _SPLIT.split(raw.strip()) if v.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        key = normalize_entity_label(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def main() -> None:
    rows: list[dict[str, Any]] = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    by_domain: dict[str, set[str]] = defaultdict(set)
    all_entities: dict[str, dict[str, Any]] = {}

    for row in rows:
        domain = str(row.get("domain") or "unknown")
        for ent in parse_entities(row):
            key = normalize_entity_label(ent)
            by_domain[domain].add(key)
            if key not in all_entities:
                all_entities[key] = {
                    "entity_label": ent,
                    "entity_key": key,
                    "domains": {domain},
                    "qid": None,
                    "wikidata_label": None,
                    "wikidata_description": None,
                    "resolution_method": "unresolved",
                }
            else:
                all_entities[key]["domains"].add(domain)

    resolved = 0
    unresolved = 0
    for key, rec in all_entities.items():
        label = rec["entity_label"]
        hit = search_wikidata_entity(label)
        qid = qid_from_value(hit.get("id")) if isinstance(hit, dict) else None
        if qid:
            rec["qid"] = qid
            rec["wikidata_label"] = hit.get("label")
            rec["wikidata_description"] = hit.get("description")
            rec["resolution_method"] = "search_api"
            resolved += 1
        else:
            unresolved += 1

    entity_rows = sorted(all_entities.values(), key=lambda r: (r["qid"] is None, r["entity_label"].lower()))

    OUT_JSON.write_text(
        json.dumps(
            {
                "source": str(SEED_PATH),
                "total_questions": len(rows),
                "unique_entities": len(entity_rows),
                "resolved_qids": resolved,
                "unresolved_entities": unresolved,
                "entities": [
                    {
                        **{k: v for k, v in r.items() if k != "domains"},
                        "domains": sorted(r["domains"]),
                    }
                    for r in entity_rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entity_label",
                "entity_key",
                "domains",
                "qid",
                "wikidata_label",
                "wikidata_description",
                "resolution_method",
            ],
        )
        writer.writeheader()
        for r in entity_rows:
            row = dict(r)
            row["domains"] = ";".join(sorted(r["domains"]))
            writer.writerow(row)

    # Priority-1 checkpoint summary for iterative ungate diagnosis
    by_domain_counts = {d: len(v) for d, v in by_domain.items()}
    coverage_lines = []
    for d in sorted(by_domain_counts):
        domain_total = by_domain_counts[d]
        domain_resolved = sum(1 for r in entity_rows if d in r["domains"] and r["qid"])
        coverage = domain_resolved / domain_total if domain_total else 0.0
        coverage_lines.append((d, domain_total, domain_resolved, coverage))

    md = [
        "# Priority 1 checkpoint — Wikidata backbone seeding",
        "",
        f"Seed file: `{SEED_PATH}`",
        f"Unique extracted entities: {len(entity_rows)}",
        f"Resolved Wikidata QIDs: {resolved}",
        f"Unresolved entities: {unresolved}",
        "",
        "## Domain checkpoint (intermediate ungate diagnostic)",
        "| domain | unique_entities | resolved_qids | resolved_rate |",
        "|---|---:|---:|---:|",
    ]
    for d, total, done, rate in coverage_lines:
        md.append(f"| {d} | {total} | {done} | {rate:.2%} |")
    md += [
        "",
        "Interpretation:",
        "- This is the Step-1 partial audit checkpoint requested before moving to later tiers.",
        "- If economics remains weak after this step, Priority 3 (official macro tier) is the binding ungate path.",
        "- If culture remains weak after Priority 2, accelerate Priority 4 outcome-tier ingestion.",
    ]
    OUT_CHECKPOINT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "out_json": str(OUT_JSON),
        "out_csv": str(OUT_CSV),
        "out_checkpoint": str(OUT_CHECKPOINT_MD),
        "unique_entities": len(entity_rows),
        "resolved_qids": resolved,
        "unresolved_entities": unresolved,
    }, indent=2))


if __name__ == "__main__":
    main()
