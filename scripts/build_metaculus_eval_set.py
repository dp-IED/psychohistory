#!/usr/bin/env python3
"""Fetch Metaculus tournament questions and save as gold-set-compatible eval set.

Usage:
    python scripts/build_metaculus_eval_set.py --project-id 33021 --output data/metaculus/cup_eval.json
    python scripts/build_metaculus_eval_set.py --project-id 33022 --output data/metaculus/aib_eval.json
    python scripts/build_metaculus_eval_set.py --all --output data/metaculus/all_tournaments.json

Auth: reads token from ~/.metaculus_token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_BASE = "https://www.metaculus.com/api2"


def _headers() -> dict[str, str]:
    token_path = os.path.expanduser("~/.metaculus_token")
    if not os.path.exists(token_path):
        print("~/.metaculus_token not found. Create it with your Metaculus API key.", file=sys.stderr)
        sys.exit(1)
    token = Path(token_path).read_text().strip()
    return {
        "Authorization": f"Token {token}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Origin": "https://www.metaculus.com",
        "Referer": "https://www.metaculus.com/",
    }


def fetch_questions(project_id: int, *, status: str = "open") -> list[dict[str, Any]]:
    """Fetch all questions for a Metaculus project."""
    all_questions: list[dict[str, Any]] = []
    url = f"{API_BASE}/questions/?project={project_id}&status={status}&limit=100"

    while url:
        req = Request(url, headers=_headers())
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except HTTPError as e:
            print(f"HTTP {e.code} fetching project {project_id}: {e.msg}", file=sys.stderr)
            return []

        all_questions.extend(data.get("results", []))
        url = data.get("next")  # Pagination
        if url and not url.startswith("http"):
            url = f"https://www.metaculus.com{url}"

    return all_questions


KNOWN_TOURNAMENTS: dict[str, int] = {
    "summer-2026-cup": 33021,
    "summer-2026-aib": 33022,
}


def discover_tournaments() -> list[dict[str, Any]]:
    """Discover active tournaments by scanning known project IDs."""
    projects: list[dict[str, Any]] = []
    for name, pid in KNOWN_TOURNAMENTS.items():
        questions = fetch_questions(pid)
        projects.append({
            "name": name,
            "project_id": pid,
            "question_count": len(questions),
            "questions": questions,
        })
    return projects


def to_gold_schema(q: dict[str, Any], project_name: str) -> dict[str, Any]:
    """Convert a Metaculus question to gold-set-compatible schema."""
    return {
        "question_id": f"metaculus-{q['id']}",
        "question_text": q.get("title", ""),
        "description": q.get("description", ""),
        "resolution_criteria": q.get("resolution_criteria", ""),
        "source": "metaculus",
        "project": project_name,
        "project_id": q.get("project", {}).get("id") if isinstance(q.get("project"), dict) else None,
        "close_date": q.get("close_time"),
        "resolve_date": q.get("resolve_time"),
        "created_date": q.get("publish_time"),
        "community_prediction": q.get("community_prediction"),
        "my_forecasts": q.get("my_forecasts"),
        "expected_target_value": None,  # Unresolved — set to None
        "url": f"https://www.metaculus.com/questions/{q['id']}/",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Metaculus eval set")
    parser.add_argument("--project-id", type=int, help="Metaculus project ID")
    parser.add_argument("--project-name", type=str, default="custom",
                        help="Project name for schema tagging")
    parser.add_argument("--all", action="store_true",
                        help="Discover and export all known tournaments")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--status", type=str, default="open",
                        help="Question status filter (open/closed/all)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_set: dict[str, Any] = {
        "generated_at": date.today().isoformat(),
        "source": "metaculus",
        "questions": [],
    }

    if args.all:
        projects = discover_tournaments()
        for proj in projects:
            for q in proj["questions"]:
                eval_set["questions"].append(to_gold_schema(q, proj["name"]))
            print(f"  {proj['name']}: {len(proj['questions'])} questions")
    elif args.project_id:
        questions = fetch_questions(args.project_id, status=args.status)
        for q in questions:
            eval_set["questions"].append(to_gold_schema(q, args.project_name))
        print(f"  Project {args.project_id}: {len(questions)} questions")
    else:
        parser.error("Must provide --project-id or --all")

    output_path.write_text(json.dumps(eval_set, indent=2))
    print(f"✓ Written {len(eval_set['questions'])} questions to {output_path}")


if __name__ == "__main__":
    main()
