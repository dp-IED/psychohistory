"""Similar-question retrieval from episodic memory and Obsidian run exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.memory_store import MemoryStore
from harness.tools.vault_markdown import read_vault_note

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)
_PERSONAL_Q_RE = re.compile(
    r"^(will i |should i |can i |did i |am i |would i |could i |do i |have i )",
    re.I,
)


def _is_personal_question(text: str) -> bool:
    """True if the question is about the agent/user's own actions (inherently unforecastable)."""
    return bool(_PERSONAL_Q_RE.match(text.strip()))


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def find_analogues(
    question: str,
    market_family: str,
    memory: MemoryStore,
    max_results: int = 5,
    *,
    exclude_exact_question: bool = True,
) -> list[dict[str, Any]]:
    """Rank prior episodes in the same family by token overlap with the question.

    Args:
        exclude_exact_question: If True, skip episodes whose question text matches exactly.
            Prevents the LLM from seeing its own prior forecast for the same question (PIT leak).
    """

    q_tokens = _tokens(question)
    episodes = [ep for ep in memory.read_all_episodes() if ep.market_family == market_family]
    scored: list[tuple[float, Any]] = []
    for ep in episodes:
        if exclude_exact_question and ep.question.strip().lower() == question.strip().lower():
            continue
        if _is_personal_question(ep.question):
            continue
        t_tokens = _tokens(ep.question)
        if not q_tokens or not t_tokens:
            continue
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens) or 1
        jacc = inter / union
        if jacc <= 0:
            continue
        scored.append((jacc, ep))

    # Deduplicate: keep only the most recent episode per unique question text
    seen_q: dict[str, tuple[float, Any]] = {}
    for jacc, ep in sorted(scored, key=lambda x: (-x[0], -x[1].cutoff_date.toordinal() if hasattr(x[1], 'cutoff_date') and x[1].cutoff_date else 0)):
        key = ep.question.strip().lower()
        if key not in seen_q:
            seen_q[key] = (jacc, ep)
    deduped = sorted(seen_q.values(), key=lambda x: -x[0])

    out: list[dict[str, Any]] = []
    for jacc, ep in deduped[:max_results]:
        out.append(
            {
                "question": ep.question,
                "final_p_yes": ep.final_p_yes,
                "brier": ep.brier_score,
                "similarity": round(jacc, 4),
                "cutoff_date": ep.cutoff_date.isoformat(),
            }
        )
    return out


def analogues_to_tool_strings(rows: list[dict[str, Any]]) -> list[str]:
    """Format analogue dicts as concise strings for the AgentToolset analogues slot."""

    lines: list[str] = []
    for row in rows:
        lines.append(
            f"p={row.get('final_p_yes')} brier={row.get('brier')} ~sim={row.get('similarity')} :: {row.get('question','')[:220]}"
        )
    return lines


def find_vault_run_analogues(
    question: str,
    category: str,
    vault_dir: Path,
    *,
    runs_subdir: str = "runs",
    max_results: int = 5,
    exclude_exact_question: bool = True,
) -> list[dict[str, Any]]:
    """Rank exported run notes in ``vault_dir/runs`` by token overlap when category matches.

    Args:
        exclude_exact_question: If True, skip notes whose question text matches exactly.
            Prevents the LLM from seeing its own prior forecast for the same question (PIT leak).
    """

    root = vault_dir.expanduser().resolve() / runs_subdir
    if not root.is_dir():
        return []

    q_tokens = _tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []

    for path in sorted(root.glob("*.md")):
        try:
            fm, _ = read_vault_note(path)
        except (OSError, ValueError):
            continue
        if str(fm.get("category", "")).lower() != category.lower():
            continue
        qn = str(fm.get("question", ""))
        if exclude_exact_question and qn.strip().lower() == question.strip().lower():
            continue
        t_tokens = _tokens(qn)
        if not q_tokens or not t_tokens:
            continue
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens) or 1
        jacc = inter / union
        if jacc <= 0:
            continue
        scored.append(
            (
                jacc,
                {
                    "question": qn,
                    "final_p_yes": fm.get("p_yes"),
                    "brier": fm.get("brier"),
                    "similarity": round(jacc, 4),
                    "cutoff_date": str(fm.get("date", "")),
                    "source": f"vault:{path.name}",
                },
            )
        )

    scored.sort(key=lambda x: -x[0])
    return [row for _, row in scored[:max_results]]
