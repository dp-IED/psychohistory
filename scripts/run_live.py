"""Submit live forecasts to Metaculus tournaments via the harness.

Usage:
    # Fetch 10 open binary questions and forecast them
    python -m scripts.run_live --max-questions 10

    # Forecast questions in a specific tournament (Summer 2026 AIB = 33022)
    python -m scripts.run_live --project-id 33022 --max-questions 5

    # Dry run — run the harness but don't submit
    python -m scripts.run_live --dry-run

Requires METACULUS_API_TOKEN in .env or environment.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from harness.agent_loop import run_agent_loop
from harness.memory_store import JsonlMemoryStore
from harness.policy_loader import load_policy
from harness.tools.real_toolset import build_real_toolset
from harness.metaculus_client import MetaculusClient

FORECAST_LOG = Path(".hermes/live_forecasts.json")
USER_AGENT = "PsychohistoryBot/1.0"


def _fetch_open_questions(
    api_token: str,
    project_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch open binary questions from Metaculus."""
    params = f"limit={limit}&status=open&order_by=-activity&type=binary"
    if project_id is not None:
        params += f"&project={project_id}"
    url = f"https://www.metaculus.com/api2/questions/?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Authorization": f"Token {api_token}",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    rows: list[dict[str, Any]] = data.get("results", [])
    # Filter to binary only
    binary = []
    for row in rows:
        q_meta = row.get("question") or {}
        if q_meta.get("type") != "binary":
            continue
        binary.append(row)
    return binary


def _load_forecast_log() -> dict[str, dict[str, Any]]:
    p = FORECAST_LOG.expanduser().resolve()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_forecast_log(log: dict[str, dict[str, Any]]) -> None:
    p = FORECAST_LOG.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit live Metaculus forecasts")
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Tournament project ID (default: all open binary questions)",
    )
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run harness but don't submit to Metaculus",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    api_token = os.environ.get("METACULUS_API_TOKEN", "").strip()
    if not api_token:
        print("METACULUS_API_TOKEN not set in .env or environment")
        return 1

    client = MetaculusClient(api_token)
    memory = JsonlMemoryStore(Path(".harness_memory"))
    policy = load_policy()
    tools = build_real_toolset(memory, policy, vault_dir=None)  # no vault for now

    # Load forecast log to avoid re-submitting
    forecast_log = _load_forecast_log()

    # Fetch open binary questions
    print("Fetching open binary questions...")
    raw_questions = _fetch_open_questions(api_token, args.project_id)
    print(f"  Found {len(raw_questions)} open binary questions")

    today = date.today()
    submitted = 0
    results: defaultdict[str, list[float]] = defaultdict(list)

    for row in raw_questions:
        if submitted >= args.max_questions:
            break

        qid = str(row["id"])
        if qid in forecast_log:
            continue

        title = str(row.get("title", ""))
        close_raw = row.get("scheduled_close_time") or row.get("close_time")
        if not close_raw:
            print(f"  Q{qid}: skip — no close time")
            continue
        try:
            close_date = date.fromisoformat(str(close_raw)[:10])
        except ValueError:
            print(f"  Q{qid}: skip — unparseable close time {close_raw}")
            continue

        print(f"\nQ{qid}: {title[:80]}...")
        print(f"  Close: {close_date}  Cutoff (today): {today}")

        # Run the harness
        try:
            result = run_agent_loop(
                question=title,
                cutoff_date=today,
                resolution_date=close_date,
                policy=policy,
                memory=memory,
                tools=tools,
                vault_dir=None,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        # Clamp to avoid strict-inequality rejection from Metaculus API
        p_clamped = max(0.001, min(0.999, result.final_p_yes))
        reasoning = result.reasoning_summary[:800]

        print(f"  p={result.final_p_yes:.4f} → clamped={p_clamped:.4f}")
        print(f"  reasoning: {reasoning[:100]}...")

        if not args.dry_run:
            try:
                client.post_forecast(int(qid), p_clamped, reasoning)
                print(f"  ✓ Submitted to Metaculus")
            except Exception as exc:
                print(f"  ✗ Submit failed: {exc}")
                continue

        # Log locally
        forecast_log[qid] = {
            "question_id": int(qid),
            "question": title,
            "close_date": close_date.isoformat(),
            "cutoff_date": today.isoformat(),
            "p_yes": result.final_p_yes,
            "p_yes_submitted": p_clamped,
            "reasoning": reasoning,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": args.dry_run,
        }
        _save_forecast_log(forecast_log)
        results["p_yes"].append(result.final_p_yes)
        submitted += 1

    print(f"\n{'='*50}")
    print(f"Done. {submitted} questions forecasted.")
    if results["p_yes"]:
        p_vals = results["p_yes"]
        print(f"Mean p_yes: {sum(p_vals)/len(p_vals):.4f}  Range: {min(p_vals):.4f}–{max(p_vals):.4f}")
    if args.dry_run:
        print("(DRY RUN — no forecasts submitted to Metaculus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
