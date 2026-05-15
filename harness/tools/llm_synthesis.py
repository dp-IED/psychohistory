"""LLM-driven forecast synthesis via cursor-agent (`agent` / `cursor-agent` CLI)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from harness.tools.evidence_graph import EvidenceGraph
from harness.tools.loop_context import get_research_market_family, get_research_resolution
from harness.tools.pit_search import search as pit_search, results_to_prompt_block

DEFAULT_CURSOR_MODEL = "composer-2-fast"


def _agent_available() -> bool:
    return shutil.which("agent") is not None or shutil.which("cursor-agent") is not None


def _agent_bin_path() -> str | None:
    return shutil.which("agent") or shutil.which("cursor-agent")


def _call_cursor_agent(prompt: str, *, model: str, workspace: Path) -> str:
    agent_bin = _agent_bin_path()
    if not agent_bin:
        raise RuntimeError("cursor-agent CLI not found on PATH")
    ws = workspace.resolve()
    cmd = [
        agent_bin,
        "--model",
        model,
        "--print",
        "--output-format",
        "json",
        "--trust",
        "--workspace",
        str(ws),
        "--force",
        prompt,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ws))
    if result.returncode != 0:
        err = (result.stderr or "")[:500]
        raise RuntimeError(f"cursor-agent call failed (exit {result.returncode}): {err}")
    out = result.stdout or ""
    if not out.strip():
        raise RuntimeError("cursor-agent returned empty stdout")
    return out


def _assistant_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _extract_text_from_agent_json_stdout(stdout: str) -> str:
    raw = stdout.strip()
    if not raw:
        raise RuntimeError("cursor-agent returned empty stdout")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    # cursor-agent native format: {"type":"result","result":"..."}
    if payload.get("type") == "result" and isinstance(payload.get("result"), str):
        return payload["result"]

    # OpenAI messages format: {"messages":[{"role":"assistant","content":"..."}]}
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return raw
    for msg in reversed(msgs):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        text = _assistant_content_to_text(msg.get("content"))
        if text.strip():
            return text
    return raw


def _forecast_workspace() -> Path:
    w = os.environ.get("CURSOR_FORECAST_WORKSPACE", "").strip()
    if w:
        return Path(w).expanduser().resolve()
    return Path.cwd().resolve()


def _call_forecast_agent(prompt: str, *, model: str, workspace: Path) -> str:
    raw_json = _call_cursor_agent(prompt, model=model, workspace=workspace)
    return _extract_text_from_agent_json_stdout(raw_json)


_JSON_FENCE = re.compile(r"\{[\s\S]*\}")


def _parse_p_yes(raw: str) -> tuple[float, str] | None:
    raw = raw.strip()
    try:
        m = _JSON_FENCE.search(raw)
        payload = json.loads(m.group(0) if m else raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    py = payload.get("p_yes")
    if py is None:
        return None
    try:
        p = float(py)
    except (TypeError, ValueError):
        return None
    if p < 0.0 or p > 1.0:
        return None
    reason = str(payload.get("reasoning", "")).strip() or "synthesis"
    return p, reason


def llm_synthesize_forecast(
    question: str,
    cutoff_date: date,
    evidence_graph: EvidenceGraph,
    past_analogues: list[dict[str, Any]],
    policy_body: str,
    calibration_shrinkage: float | None,
    *,
    step: int,
    evidence_count: int,
    node_count: int,
    vault_context: str = "",
) -> tuple[float, str]:
    """Synthesize p_yes via cursor-agent. Raises RuntimeError if the CLI is missing or the call fails."""

    if not _agent_available():
        raise RuntimeError(
            "cursor-agent CLI not found on PATH. "
            "Install cursor-agent (agent / cursor-agent) for forecast synthesis."
        )

    lines = [
        f"- {a.get('question','')[:200]} p_yes={a.get('final_p_yes')} brier={a.get('brier')}"
        for a in past_analogues
    ]
    analog_txt = "\n".join(lines) if lines else "(none)"

    article_lines: list[str] = []
    for url, title, snip in evidence_graph.articles[:12]:
        article_lines.append(f"- {title[:160]} | {url[:120]} | {snip[:120]}")
    articles_txt = "\n".join(article_lines) if article_lines else "(no parsed articles)"

    shrink_note = (
        f"Apply mental calibration: shrink extreme forecasts toward 0.5; shrinkage_hint={calibration_shrinkage}."
        if calibration_shrinkage is not None
        else "No shrinkage hint."
    )

    res_date = get_research_resolution()
    if res_date is not None:
        horizon_days = max(1, abs((res_date - cutoff_date).days))
    else:
        horizon_days = 30
    category = get_research_market_family() or "general"

    workspace = _forecast_workspace()

    # Auto PIT search — gather point-in-time filtered evidence cursor-agent can use
    try:
        pit_resp = pit_search(question, cutoff_date, max_results=5)
        pit_context = results_to_prompt_block(pit_resp.results, cutoff_date) if not pit_resp.error else f"(PIT search unavailable: {pit_resp.error})"
    except Exception:
        pit_context = "(PIT search failed)"

    prompt = f"""You are a forecasting agent. Produce a calibrated probability.

Question: {question}
Cutoff date: {cutoff_date.isoformat()}

## PIT-filtered Research (knowable as of cutoff)

{pit_context}

## Evidence Graph

{evidence_graph.summary}

Articles:
{articles_txt}

Entities: {", ".join(n.label for n in evidence_graph.nodes if n.node_type == "entity")[:800]}

## Past Similar Questions

{analog_txt}

## Vault Context

{vault_context.strip()[:12000] if vault_context.strip() else "(no vault context loaded)"}

## Policy Notes

{policy_body[:2500] if policy_body.strip() else "(read .harness/policy.md for the machine policy)"}

{shrink_note}

Be creative about how you gather evidence. Try angles others wouldn't think of.
Output one JSON line:
{{"p_yes": 0.XX, "reasoning": "one sentence"}}"""

    model = (
        os.environ.get("CURSOR_FORECAST_MODEL", DEFAULT_CURSOR_MODEL).strip()
        or DEFAULT_CURSOR_MODEL
    )

    try:
        raw = _call_forecast_agent(prompt, model=model, workspace=workspace)
        parsed = _parse_p_yes(raw)
        if parsed is None:
            raise RuntimeError(
                "LLM synthesis returned unparseable output — no valid JSON with p_yes found. "
                f"Raw output (first 500 chars): {raw[:500]}"
            )
        return parsed[0], parsed[1]
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        raise


def synthesis_policy_hint(policy: object) -> str:
    body = getattr(policy, "body", "") or ""
    return str(body).strip()
