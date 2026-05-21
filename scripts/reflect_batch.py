#!/usr/bin/env python3
"""Batch reflection: review past forecasts, diagnose systematic errors, patch vault rules.

Reads forecast JSONs from data/metaculus/forecasts/, groups by domain, runs a single
hermes call to identify patterns, then applies patches to vault files.

Usage:
  # Review all forecasts, output diagnosis only (no patching)
  python scripts/reflect_batch.py --diagnose

  # Review + apply rule patches to vault
  python scripts/reflect_batch.py --patch

  # Only review forecasts from the last N days
  python scripts/reflect_batch.py --since 7 --diagnose
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FORECASTS_DIR = ROOT / "data" / "metaculus" / "forecasts"
GRAPH_VAULT = ROOT / "graph-vault"
FORECAST_RULES = GRAPH_VAULT / "_forecast_instructions.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reflect-batch")


# ── Forecast loading ──────────────────────────────────────────────────


def load_forecasts(since_days: int | None = None) -> list[dict[str, Any]]:
    """Load forecast records, optionally filtered by recency."""
    if not FORECASTS_DIR.exists():
        logger.warning("No forecasts directory at %s", FORECASTS_DIR)
        return []

    records: list[dict[str, Any]] = []
    cutoff_date = None
    if since_days:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)

    for fpath in sorted(FORECASTS_DIR.glob("forecast-*.json")):
        try:
            record = json.loads(fpath.read_text())
            if cutoff_date:
                ts = record.get("forecasted_at", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt < cutoff_date:
                            continue
                    except ValueError:
                        pass
            records.append(record)
        except Exception:
            logger.warning("Skipping corrupt forecast: %s", fpath.name)

    return records


def group_by_domain(records: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """Group forecasts by domain using keyword matching on question titles."""
    DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "health": ["outbreak", "case", "disease", "virus", "infection", "epidemic",
                    "hospital", "who", "cdc", "vaccine", "pandemic", "hantavirus",
                    "ebola", "covid", "public health", "quarantine", "hondius"],
        "politics": ["election", "president", "vote", "parliament", "congress",
                      "senate", "minister", "party", "government", "impeach",
                      "duterte", "pope", "sanction"],
        "economics": ["gdp", "inflation", "cpi", "rate", "market", "price",
                       "tariff", "trade", "recession", "fed", "stock", "bond",
                       "currency", "debt"],
        "geopolitics": ["war", "conflict", "invasion", "military", "nuclear",
                         "missile", "sanction", "nato", "threat", "ukraine",
                         "russia", "china", "iran"],
        "culture": ["film", "movie", "oscar", "grammy", "album", "song",
                     "sport", "olympic", "fifa", "world cup", "tournament",
                     "game", "box office"],
    }

    groups: dict[str, list[dict]] = {}
    for rec in records:
        title = (rec.get("title", "") or "").lower()
        matched = None
        best_score = 0
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in title)
            if score > best_score:
                best_score = score
                matched = domain
        if matched and best_score > 0:
            groups.setdefault(matched, []).append(rec)
    return groups


# ── Diagnosis ─────────────────────────────────────────────────────────


def build_diagnosis_prompt(records: list[dict[str, Any]]) -> str:
    """Build a prompt summarizing forecast performance for batch diagnosis."""
    domain_groups = group_by_domain(records)
    total = len(records)
    errors = [r for r in records if r.get("forecast_error")]

    lines = [
        "=== BATCH REFLECTION: FORECAST DIAGNOSIS ===",
        "",
        f"Total forecasts reviewed: {total}",
        f"Forecasts with errors: {len(errors)}",
        f"Domains represented: {list(domain_groups.keys())}",
        "",
        "=== FORECAST SUMMARIES ===",
    ]

    for domain, recs in sorted(domain_groups.items()):
        lines.append(f"\n## Domain: {domain} ({len(recs)} forecasts)")
        for r in recs[:5]:  # Limit per domain to keep prompt manageable
            p_yes = r.get("p_yes", "N/A")
            title = (r.get("title", "") or "")[:80]
            reasoning = (r.get("reasoning", "") or "")[:150]
            refl = r.get("orchestrator", {}).get("reflection", {})
            diagnosis = refl.get("diagnosis", "")
            adj_p = refl.get("adjusted_p_yes")

            lines.append(f"  Q: {title}")
            lines.append(f"     p_yes={p_yes}" +
                         (f" → adjusted={adj_p}" if adj_p is not None else ""))
            if reasoning:
                lines.append(f"     reasoning: {reasoning}")
            if diagnosis:
                lines.append(f"     reflection: {diagnosis[:150]}")

    lines += [
        "",
        "=== YOUR TASK ===",
        "Review these forecasts and identify what went wrong. You have full autonomy over",
        "how to improve the vault. You can create rules, write procedures, edit concepts,",
        "add threads — whatever you think will prevent these errors from recurring.",
        "",
        "Don't just list errors. Diagnose WHY they cluster, and decide what structural",
        "change to the vault would address the root cause. If multiple errors share a theme,",
        "a single procedure may be better than several rules.",
        "",
        "=== TARGET VAULT FILES ===",
        f"You may modify anything under {GRAPH_VAULT}/. Common targets:",
        f"  {FORECAST_RULES} (forecast rules)",
        f"  {GRAPH_VAULT}/procedures/ (forecast procedures)",
        f"  {GRAPH_VAULT}/domains/<domain>/concepts/ (domain playbooks)",
        f"  {GRAPH_VAULT}/domains/<domain>/threads/ (narrative threads)",
        "",
        "=== OUTPUT FORMAT (MANDATORY) ===",
        "Respond with ONLY a single JSON object:",
        "{",
        '    "diagnosis": "what patterns of error you observed and their root causes",',
        '    "vault_changes": [',
        "        {",
        '            "action": "add_rule|edit_procedure|new_concept|modify_thread|other",',
        "            \"target\": \"path to vault file or 'new'\",",
        '            "what": "what to change",',
        '            "why": "how this prevents recurrence"',
        "        },",
        "        ...",
        "    ],",
        '    "priority": "which change to apply first and why"',
        "}",
    ]

    return "\n".join(lines)


def run_diagnosis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run batch diagnosis via hermes -z."""
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    if not records:
        return {"diagnosis": "no forecasts to review", "vault_changes": [], "priority": ""}

    prompt = build_diagnosis_prompt(records)
    logger.info("Running batch diagnosis on %d forecasts...", len(records))

    result = subprocess.run(
        ["hermes", "-z", prompt, "--profile", "forecasting", "--yolo"],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"hermes exit {result.returncode}: {(result.stderr or '')[:300]}")

    raw = (result.stdout or "").strip()

    # Extract JSON
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        logger.error("No JSON found in diagnosis output. Raw: %s", raw[:500])
        return {"error": "no JSON in output", "raw": raw[:1000]}

    try:
        return json.loads(m.group(0))  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s", e)
        return {"error": str(e), "raw": raw[:1000]}


# ── Patching ──────────────────────────────────────────────────────────


def apply_patches(diagnosis: dict[str, Any], *, dry_run: bool = True) -> list[str]:
    """Apply vault changes suggested by the diagnosis agent. Returns list of actions taken."""
    actions: list[str] = []

    for change in diagnosis.get("vault_changes", []):
        action = change.get("action", "other")
        target = change.get("target", "")
        what = change.get("what", "")
        why = change.get("why", "")

        if not what:
            continue

        if dry_run:
            logger.info("  [DRY-RUN] %s → %s: %s", action, target, why[:80])
            actions.append(f"DRY-RUN: {target} — {why[:80]}")
            continue

        # Resolve target path
        if target == "new" or not target:
            # New file — figure out where to put it based on action
            if action == "add_rule":
                target_path = FORECAST_RULES
                current = target_path.read_text(encoding="utf-8")
                target_path.write_text(current.rstrip() + f"\n\n{what}\n", encoding="utf-8")
            elif action == "edit_procedure":
                target_path = GRAPH_VAULT / "procedures" / f"{_slugify(why[:40])}.md"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(what, encoding="utf-8")
            elif action == "new_concept":
                target_path = GRAPH_VAULT / "domains" / "_concepts" / f"{_slugify(why[:40])}.md"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(what, encoding="utf-8")
            else:
                logger.warning("  Cannot create new file for action '%s' without target", action)
                actions.append(f"SKIP: {action} — no target path")
                continue
            logger.info("  ✅ Created %s", target_path)
            actions.append(f"CREATED: {target_path}")
        else:
            target_path = ROOT / target if not target.startswith("/") else Path(target)
            if not target_path.exists():
                logger.warning("  Target does not exist: %s", target_path)
                actions.append(f"SKIP (not found): {target}")
                continue

            if action == "add_rule":
                current = target_path.read_text(encoding="utf-8")
                target_path.write_text(current.rstrip() + f"\n\n{what}\n", encoding="utf-8")
                logger.info("  ✅ Appended to %s", target)
                actions.append(f"APPENDED: {target}")
            else:
                # For edits, overwrite or use patch
                target_path.write_text(what, encoding="utf-8")
                logger.info("  ✅ Wrote %s", target)
                actions.append(f"WRITTEN: {target}")

    return actions


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:50]
    return slug.strip("-")

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch reflection on past forecasts")
    parser.add_argument("--diagnose", action="store_true", help="Run diagnosis (default if no action specified)")
    parser.add_argument("--patch", action="store_true", help="Apply suggested rule patches to vault")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show patches without applying (default)")
    parser.add_argument("--apply", dest="dry_run", action="store_false", help="Actually apply patches")
    parser.add_argument("--since", type=int, default=None, help="Only review forecasts from last N days")
    args = parser.parse_args()

    if not args.diagnose and not args.patch:
        args.diagnose = True

    records = load_forecasts(since_days=args.since)
    if not records:
        logger.info("No forecasts found")
        return

    logger.info("Loaded %d forecasts", len(records))

    diagnosis = run_diagnosis(records)

    vault_changes = diagnosis.get("vault_changes", [])
    priority = diagnosis.get("priority", "")
    diag_text = diagnosis.get("diagnosis", "")

    print("\n" + "=" * 60)
    print("BATCH REFLECTION RESULTS")
    print("=" * 60)

    if diag_text:
        print(f"\nDiagnosis:\n  {diag_text[:500]}")
    
    print(f"\nVault changes suggested ({len(vault_changes)}):")
    for vc in vault_changes[:15]:
        print(f"  • [{vc.get('action', '?')}] {vc.get('target', '?')}")
        print(f"    {vc.get('why', '')[:120]}")

    if args.patch:
        print(f"\nPriority: {priority}")
        actions = apply_patches(diagnosis, dry_run=args.dry_run)
        for a in actions:
            print(f"  {a}")

    # Save diagnosis to file for future reference
    out_dir = ROOT / "data" / "metaculus" / "reflections"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"batch-reflection-{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": ts,
        "forecasts_reviewed": len(records),
        "diagnosis": diagnosis,
    }, indent=2))
    logger.info("Saved batch reflection to %s", out_path)


if __name__ == "__main__":
    main()
