#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ISO2_RE = re.compile(r"^[A-Z]{2}$")
ISO2_PREFIX_RE = re.compile(r"^[A-Z]{2}-")

# Deterministic post-hoc alias map for known mixed-format admin labels.
ADMIN1_LABEL_TO_COUNTRY = {
    # Egypt
    "cairo": "EG",
    "giza": "EG",
    "alexandria": "EG",
    "suez": "EG",
    "gharbia": "EG",
    "sharqia": "EG",
    "dakahlia": "EG",
    "assiut": "EG",
    "menia": "EG",
    "port said": "EG",
    "qalyubia": "EG",
    "beheira": "EG",
    "fayoum": "EG",
    "ismailia": "EG",
    "damietta": "EG",
    "menoufia": "EG",
    "kafr el-sheikh": "EG",
    "north sinai": "EG",
    "south sinai": "EG",
    "aswan": "EG",
    "qena": "EG",
    "beni suef": "EG",
    "luxor": "EG",
    "sohag": "EG",
    "red sea": "EG",
    "new valley": "EG",
    # Libya coarse regions
    "west": "LY",
    "east": "LY",
    "south": "LY",
}


def infer_country_from_admin1(admin1_code: str | None) -> tuple[str | None, str]:
    if not admin1_code:
        return None, "missing"
    s = admin1_code.strip()
    if not s:
        return None, "blank"
    if ISO2_RE.fullmatch(s):
        return s, "native_iso2"
    if ISO2_PREFIX_RE.match(s):
        return s.split("-", 1)[0], "native_prefixed"
    mapped = ADMIN1_LABEL_TO_COUNTRY.get(s.lower())
    if mapped:
        return mapped, "alias_map"
    return None, "unmapped_label"


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = obj.get("rows", [])

    country_source = Counter()
    inferred_countries = Counter()
    invalid_country_rows = 0
    missing_first_seen_rows = 0
    missing_entity_hint_rows = 0
    unmapped_admin1_rows = 0
    mixed_format_rows = 0
    noncode_examples: list[str] = []

    for r in rows:
        admin1 = r.get("admin1_code")
        inferred_country, source = infer_country_from_admin1(admin1)
        country_source[source] += 1
        if inferred_country is not None:
            inferred_countries[inferred_country] += 1
        else:
            invalid_country_rows += 1
        if source in {"alias_map", "unmapped_label"}:
            mixed_format_rows += 1
            if admin1 and admin1 not in noncode_examples and len(noncode_examples) < 25:
                noncode_examples.append(admin1)
        if source == "unmapped_label":
            unmapped_admin1_rows += 1

        if not r.get("first_seen"):
            missing_first_seen_rows += 1

        ext = r.get("extensions") or {}
        hints = ext.get("entity_hint_keys")
        if not isinstance(hints, list) or len(hints) == 0:
            missing_entity_hint_rows += 1

    row_count = len(rows)

    checks = {
        "country_inference_complete": invalid_country_rows == 0,
        "no_unmapped_admin1_labels": unmapped_admin1_rows == 0,
        "first_seen_complete": missing_first_seen_rows == 0,
        "entity_hint_keys_complete": missing_entity_hint_rows == 0,
    }

    passed = all(checks.values())

    return {
        "manifest_path": str(manifest_path),
        "row_count": row_count,
        "passed": passed,
        "checks": checks,
        "metrics": {
            "invalid_country_rows": invalid_country_rows,
            "unmapped_admin1_rows": unmapped_admin1_rows,
            "mixed_format_rows": mixed_format_rows,
            "missing_first_seen_rows": missing_first_seen_rows,
            "missing_entity_hint_rows": missing_entity_hint_rows,
            "mixed_format_rate": (mixed_format_rows / row_count) if row_count else None,
        },
        "country_source_counts": dict(country_source),
        "inferred_country_distribution": dict(inferred_countries),
        "noncode_admin1_examples": noncode_examples,
        "policy_note": (
            "If this gate passes consistently in production runs, this validator+normalization contract "
            "should be re-implemented as the de facto warehouse build mechanism."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Warehouse quality gate: geo/time/entity contract checks for manifest rows.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero when checks fail.")
    args = p.parse_args()

    result = validate_manifest(args.manifest)
    payload = json.dumps(result, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(str(args.output))
    else:
        print(payload)

    if args.strict and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
