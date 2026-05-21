#!/usr/bin/env python3
"""Wire tournament watcher to the PIT forecast pipeline.

Usage:
  # Dry-run: detect but don't post
  python scripts/watch_tournaments.py --tournament cup --dry-run
  
  # Live: forecast and post to Metaculus
  python scripts/watch_tournaments.py --tournament cup --live
  
  # Watch for drops on Summer Bot (1.5hr windows)
  python scripts/watch_tournaments.py --tournament summer-bot --live --interval 30
  
  # Single poll (no loop)
  python scripts/watch_tournaments.py --tournament cup --once

Tournaments: cup (33021), summer-bot (33022), market-pulse (33013), minibench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.tournament_watcher import (
    KNOWN_TOURNAMENTS,
    DropEvent,
    QuestionInfo,
    TournamentWatcher,
    WatcherConfig,
    binary_forecast_payload,
    post_comment,
    post_forecast,
)

# ── Setup ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data" / "metaculus" / "watcher"
FORECASTS_DIR = ROOT / "data" / "metaculus" / "forecasts"
GRAPH_VAULT = ROOT / "graph-vault"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("watch")


# ── Forecast pipeline ────────────────────────────────────────────────


def run_forecast_pipeline(
    question: QuestionInfo,
    token: str,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run the full PIT forecast pipeline for one question.

    1. PIT librarian: research against vault at current date cutoff
    2. Forecaster: produce p_yes + reasoning
    3. Post to Metaculus (unless dry_run)
    4. Save forecast locally
    """
    cutoff = date.today()
    logger.info("🔬 Q%d: %s", question.question_id, question.title[:80])
    logger.info("   type=%s closes=%s", question.question_type, question.close_time[:20])

    # ── Step 1: Vault research ──────────────────────────────────
    vault_context = _read_vault_for_question(question)
    logger.info("   vault: %d files read, %d chars", 
                 vault_context.get("files_read", 0),
                 len(vault_context.get("content", "")))

    # ── Step 2: Forecast ────────────────────────────────────────
    # Calls hermes orchestrator (hermes -z --profile forecasting)
    # with PIT enforcement, vault research, and structured JSON output.
    try:
        p_yes, reasoning, forecast_meta = _make_forecast(question, vault_context, cutoff)
    except Exception as e:
        logger.error("   ❌ Forecast failed: %s", e)
        # Save failure record and skip — don't crash the watcher
        record = {
            "question_id": question.question_id,
            "post_id": question.post_id,
            "title": question.title,
            "question_type": question.question_type,
            "cutoff_date": cutoff.isoformat(),
            "close_time": question.close_time,
            "p_yes": None,
            "reasoning": "",
            "forecast_error": str(e)[:500],
            "vault_files_read": vault_context.get("files_read", 0),
            "forecasted_at": datetime.now(timezone.utc).isoformat(),
            "posted": False,
        }
        _save_forecast_record(record)
        return record

    logger.info("   p_yes=%.3f reasoning=%d chars", p_yes, len(reasoning))

    # ── Step 2.5: Reflection ────────────────────────────────────
    # Check forecast quality: resolution criteria alignment, temporal decay,
    # base rates, feedback loops. Adjusts if large errors detected.
    try:
        p_yes, reasoning, reflection_meta = _reflect_on_forecast(
            question, p_yes, reasoning, cutoff, vault_context
        )
        forecast_meta["reflection"] = reflection_meta
    except Exception as e:
        logger.warning("   ⚠️ Reflection failed (non-fatal): %s", e)
        reflection_meta = {"error": str(e)[:500]}

    # ── Step 3: Persist locally ─────────────────────────────────
    record = {
        "question_id": question.question_id,
        "post_id": question.post_id,
        "title": question.title,
        "question_type": question.question_type,
        "cutoff_date": cutoff.isoformat(),
        "close_time": question.close_time,
        "p_yes": p_yes,
        "reasoning": reasoning[:500],
        "vault_files_read": vault_context.get("files_read", 0),
        "vault_chars": len(vault_context.get("content", "")),
        "orchestrator": forecast_meta,
        "forecasted_at": datetime.now(timezone.utc).isoformat(),
        "posted": not dry_run,
    }
    _save_forecast_record(record)

    # ── Step 4: Post to Metaculus ───────────────────────────────
    if not dry_run:
        try:
            post_forecast(
                question.question_id,
                binary_forecast_payload(p_yes),
                token,
            )
            comment = (
                f"## Vault Forecast\n\n"
                f"**Probability:** {p_yes:.1%}\n\n"
                f"**Vault files consulted:** {vault_context.get('files_read', 0)}\n\n"
                f"**Reasoning:**\n{reasoning[:2000]}"
            )
            post_comment(question.post_id, comment, token)
            logger.info("   ✅ Posted forecast + comment")
            record["posted"] = True
        except Exception as e:
            logger.error("   ❌ Post failed: %s", e)
            record["post_error"] = str(e)

    return record


# ── Vault research ───────────────────────────────────────────────────


def _read_vault_for_question(question: QuestionInfo) -> dict[str, Any]:
    """Read relevant vault files for a question.

    Uses keyword matching across domains, threads, concepts, timeline.
    Returns content and metadata.
    """
    if not GRAPH_VAULT.exists():
        return {"files_read": 0, "content": ""}

    # Extract keywords from question
    keywords = _extract_keywords(question)
    logger.debug("   keywords: %s", keywords[:10])

    # Search for matching vault files
    matches: list[Path] = []
    search_dirs = [
        GRAPH_VAULT / "domains",
        GRAPH_VAULT / "timeline",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for md_file in search_dir.rglob("*.md"):
            if md_file.name.startswith("."):
                continue
            try:
                text = md_file.read_text(encoding="utf-8")[:2000].lower()
                score = sum(1 for kw in keywords if kw.lower() in text)
                if score >= 2:
                    matches.append(md_file)
            except Exception:
                pass

    # Sort by relevance, take top 10
    matches.sort(key=lambda p: _score_file(p, keywords), reverse=True)
    top_matches = matches[:10]

    # Build context
    content_parts: list[str] = []
    for mf in top_matches:
        rel_path = mf.relative_to(GRAPH_VAULT)
        try:
            content_parts.append(f"### {rel_path}\n{mf.read_text(encoding='utf-8')[:800]}")
        except Exception:
            pass

    return {
        "files_read": len(top_matches),
        "files": [str(m.relative_to(GRAPH_VAULT)) for m in top_matches],
        "content": "\n\n".join(content_parts),
    }


def _extract_keywords(question: QuestionInfo) -> list[str]:
    """Extract search keywords from question text."""
    import re

    text = f"{question.title} {question.description} {question.resolution_criteria}"
    # Extract capitalized phrases, entities, key terms
    words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\b[a-z]{4,}\b", text)
    # Dedupe, filter noise
    noise = {"will", "the", "before", "after", "what", "that", "this", "with", "from",
             "have", "been", "they", "their", "there", "which", "would", "could", "should"}
    return list(dict.fromkeys(w for w in words if w.lower() not in noise))[:20]


def _score_file(path: Path, keywords: list[str]) -> int:
    """Score a vault file by keyword match density."""
    try:
        text = path.read_text(encoding="utf-8").lower()
        return sum(1 for kw in keywords if kw.lower() in text)
    except Exception:
        return 0


# ── Forecast logic ───────────────────────────────────────────────────


def _build_question_text(question: QuestionInfo) -> str:
    """Assemble full question text for the orchestrator prompt."""
    parts = [question.title]
    if question.description:
        parts.append(f"\n{question.description}")
    if question.resolution_criteria:
        parts.append(f"\nResolution criteria: {question.resolution_criteria}")
    if question.fine_print:
        parts.append(f"\nFine print: {question.fine_print}")
    return "\n".join(parts)


def _make_forecast(
    question: QuestionInfo,
    vault_context: dict[str, Any],
    cutoff: date,
) -> tuple[float, str, dict[str, Any]]:
    """Produce a probability estimate using hermes orchestrator with PIT profile.

    Calls ``hermes -z --profile forecasting`` via harness.orchestrator.run_structured.
    The orchestrator handles vault research, PIT enforcement, and JSON output parsing.
    Timeout: 20 min (orchestrator default). Suitable for 1.5hr question windows.

    Returns (p_yes, reasoning, metadata).
    """
    from harness.orchestrator import run_structured

    question_text = _build_question_text(question)

    logger.info("   🧠 Calling hermes orchestrator (may take 5-20 min)...")
    p_yes, reasoning, metadata = run_structured(
        question_text=question_text,
        cutoff=cutoff,
        vault_dir=GRAPH_VAULT,
        question_id=str(question.question_id),
        source="metaculus-tournament",
    )

    logger.info(
        "   orchestrator metadata: vault_files=%d rules_checked=%d rules_triggered=%d",
        len(metadata.get("vault_files_read", [])),
        len(metadata.get("rules_checked", [])),
        len(metadata.get("rules_triggered", [])),
    )

    return p_yes, reasoning, metadata


# ── Reflection ────────────────────────────────────────────────────────


def _reflect_on_forecast(
    question: QuestionInfo,
    p_yes: float,
    reasoning: str,
    cutoff: date,
    vault_context: dict[str, Any],
) -> tuple[float, str, dict[str, Any]]:
    """Reflect on a forecast: check resolution criteria alignment, temporal decay,
    base rates, and containment/feedback loops. Adjusts if errors detected.

    Uses a focused hermes -z call (short timeout: 5 min).
    Non-fatal — failures log a warning and return the original forecast.
    """
    import shutil
    import subprocess

    if not shutil.which("hermes"):
        logger.warning("   hermes CLI not found, skipping reflection")
        return p_yes, reasoning, {"skipped": "hermes not found"}

    question_text = _build_question_text(question)
    close_dt = _parse_close_datetime(question.close_time)
    days_elapsed = (cutoff - _extract_open_date(question)).days if question.close_time else 0
    days_remaining = (close_dt - cutoff).days if close_dt else 0

    prompt = f"""=== REFLECTION ===

You are a reflection agent. Your job is to catch errors in a forecast before it's posted,
and to improve the vault so similar errors don't recur.

QUESTION:
{question_text}

OUR FORECAST: p_yes = {p_yes:.3f} ({p_yes:.1%})
OUR REASONING: {reasoning[:2000]}

CONTEXT:
- Cutoff date: {cutoff}
- Days elapsed since open: {days_elapsed}
- Days remaining until close: {days_remaining}

=== YOUR TASK ===

Examine the forecast and decide if it's wrong. You have full autonomy over what to check.
Common failure modes include misreading the resolution criteria, treating probability as
static when it should decay, ignoring event-structure (chains multiply, they don't add),
and failing to anchor against external signals.

If the forecast is correct, say so and explain why.

If it's wrong, adjust p_yes and explain why. Then decide what to change in the vault to
prevent this class of error. You can suggest new rules, edit existing procedures, add
concepts, modify threads — whatever you think will help. You are not limited to a fixed
set of actions.

Respond with ONLY a single JSON object:
{{
    "diagnosis": "what (if anything) is wrong",
    "adjusted_p_yes": 0.XX,
    "adjusted_reasoning": "revised reasoning if changed",
    "confidence_in_adjustment": "high|medium|low",
    "vault_changes": [
        {{
            "action": "add_rule|edit_procedure|new_concept|modify_thread|other",
            "target": "path to vault file or 'new'",
            "what": "what to change",
            "why": "how this prevents recurrence"
        }}
    ]
}}

If the forecast is correct, set adjusted_p_yes = {p_yes:.3f}."""

    try:
        result = subprocess.run(
            ["hermes", "-z", prompt, "--profile", "forecasting", "--yolo"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"hermes exit {result.returncode}: {(result.stderr or '')[:200]}")

        raw = (result.stdout or "").strip()
        reflection = _extract_json_reflection(raw)

        diagnosis = reflection.get("diagnosis", "")
        adj_p = reflection.get("adjusted_p_yes", p_yes)
        adj_reasoning = reflection.get("adjusted_reasoning", reasoning)
        confidence = reflection.get("confidence_in_adjustment", "medium")
        vault_changes = reflection.get("vault_changes", [])

        # Validate adjustment
        if not isinstance(adj_p, (int, float)) or adj_p < 0 or adj_p > 1:
            logger.warning("   ⚠️ Reflection returned invalid p_yes=%s, keeping original", adj_p)
            adj_p = p_yes
            adj_reasoning = reasoning

        if abs(adj_p - p_yes) > 0.01:
            logger.info(
                "   🔍 Reflection adjusted p_yes %.3f→%.3f (%s confidence)",
                p_yes, adj_p, confidence,
            )
            logger.info("   diagnosis: %s", diagnosis[:200])
        else:
            logger.info("   ✅ Reflection: forecast unchanged")

        if vault_changes:
            logger.info("   vault changes suggested: %d", len(vault_changes))
            for vc in vault_changes[:5]:
                logger.info("     → %s: %s", vc.get("action", "?"), vc.get("target", "?")[:60])

        return (
            float(adj_p),
            str(adj_reasoning),
            {
                "diagnosis": diagnosis,
                "original_p_yes": p_yes,
                "confidence": confidence,
                "vault_changes": vault_changes,
            },
        )

    except Exception as e:
        logger.warning("   ⚠️ Reflection call failed: %s", e)
        return p_yes, reasoning, {"error": str(e)[:500]}


def _extract_json_reflection(text: str) -> dict[str, Any]:
    """Extract JSON from reflection output, with fallback parsing."""
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Partial recovery — extract key fields
        result: dict[str, Any] = {}
        for key in ["diagnosis", "adjusted_reasoning", "confidence_in_adjustment"]:
            pat = _re.search(rf'"{key}"\s*:\s*"([^"]*)"', m.group(0))
            if pat:
                result[key] = pat.group(1)
        py_match = _re.search(r'"adjusted_p_yes"\s*:\s*([\d.]+)', m.group(0))
        if py_match:
            result["adjusted_p_yes"] = float(py_match.group(1))
        return result


def _parse_close_datetime(close_time: str) -> date | None:
    """Parse Metaculus close time string to date."""
    if not close_time:
        return None
    try:
        return datetime.fromisoformat(close_time.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def _extract_open_date(question: QuestionInfo) -> date:
    """Extract approximate open date from question metadata."""
    # Metaculus questions typically open a few days before first close
    today = date.today()
    close_dt = _parse_close_datetime(question.close_time)
    if close_dt:
        # Rough heuristic: questions are typically open ~2 weeks before close
        return close_dt.replace(day=max(1, close_dt.day - 14))
    return today


# ── Persistence ──────────────────────────────────────────────────────


def _save_forecast_record(record: dict[str, Any]) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"forecast-{record['question_id']}-{ts}.json"
    (FORECASTS_DIR / fname).write_text(json.dumps(record, indent=2))


# ── Drop handler ─────────────────────────────────────────────────────


def make_handler(token: str, dry_run: bool = True):
    """Create a drop handler wired to the forecast pipeline."""

    def handle_drop(event: DropEvent) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "📬 PROCESSING DROP: %d new questions for %s",
            len(event.questions),
            event.tournament_id,
        )
        logger.info("=" * 60)
        for q in event.questions:
            if q.question_type == "binary":
                run_forecast_pipeline(q, token, dry_run=dry_run)
            else:
                logger.info("  Q%d: skipping %s question", q.question_id, q.question_type)

    return handle_drop


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Tournament watcher + forecaster")
    parser.add_argument(
        "--tournament",
        type=str,
        default="cup",
        choices=list(KNOWN_TOURNAMENTS),
        help="Tournament to watch",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds",
    )
    parser.add_argument(
        "--once", action="store_true", help="Single poll, then exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Detect but don't post (default)",
    )
    parser.add_argument(
        "--live", dest="dry_run", action="store_false",
        help="Post forecasts to Metaculus",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=0,
        help="Max poll cycles (0=forever)",
    )
    args = parser.parse_args()

    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.metaculus_token")
        if os.path.exists(token_path):
            token = Path(token_path).read_text().strip()
    if not token:
        print("METACULUS_TOKEN not set and ~/.metaculus_token not found", file=sys.stderr)
        sys.exit(1)

    tid = KNOWN_TOURNAMENTS[args.tournament]
    name = args.tournament

    config = WatcherConfig(
        tournament_id=tid,
        name=name,
        poll_interval_seconds=args.interval,
    )

    watcher = TournamentWatcher(
        token=token,
        config=config,
        state_dir=STATE_DIR,
        on_drop=make_handler(token, dry_run=args.dry_run),
    )

    if args.once:
        logger.info("Single poll for %s (%s)", name, tid)
        watcher.poll_once()  # Handler fires internally via on_drop
        # Note: event already processed by watcher's internal callback
    else:
        logger.info("Starting watcher for %s — interval=%ds, dry_run=%s",
                     name, args.interval, args.dry_run)
        try:
            watcher.watch(max_cycles=args.max_cycles)
        except KeyboardInterrupt:
            logger.info("Stopped by user")


if __name__ == "__main__":
    main()
