"""Minimal orchestration: load questions, prompt hermes, store runs in graph-vault.

Three modes:
- **Structured** (Polymarket / Metaculus):  question + cutoff -> JSON p_yes
  Single-agent: the agent researches autonomously using its tools.
- **Orchestrated** (multi-agent): the agent spawns sub-agents via delegate_task
  for diverse perspectives (actor simulations, game theorists, regional specialists).
- **Free-form** (direct analysis):  any text -> natural-language response

Usage:
  from harness.orchestrator import run_structured, run_orchestrated, run_freeform
  p_yes, reasoning, meta = run_orchestrated("Will X happen?", cutoff=...)
  # meta contains agents_used, vault_edits_summary, key_assumptions, roster_gaps
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

from harness.runs import RunNote, write_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200
_ORCHESTRATOR_TIMEOUT = 2400  # longer — delegate_task calls for each sub-agent
_PROFILE = "forecasting"


# ---------------------------------------------------------------------------
# Hermes one-shot
# ---------------------------------------------------------------------------

def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes (profile={_PROFILE}) failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


# ---------------------------------------------------------------------------
# Output parsers and validators
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _JSON_FENCE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def validate_structured(raw: str) -> tuple[float, str, list[str]]:
    """Validate structured output: must contain JSON with p_yes. Returns (p_yes, reasoning, errors)."""
    payload = _extract_json(raw)
    if payload is None:
        return (0.5, "", ["No valid JSON object found in output."])

    errors: list[str] = []
    py = payload.get("p_yes")
    if py is None:
        errors.append("Missing key 'p_yes'.")
        py = 0.5
    try:
        p = float(py)
    except (TypeError, ValueError):
        errors.append(f"'p_yes' is not a float: {py!r}.")
        p = 0.5
    if p < 0.0 or p > 1.0:
        errors.append(f"'p_yes'={p} outside [0, 1].")
        p = max(0.0, min(1.0, p))

    reason = str(payload.get("reasoning", "")).strip() or "synthesis"
    return (p, reason, errors)


def validate_freeform(raw: str) -> tuple[str, list[str]]:
    text = raw.strip()
    if not text:
        return ("", ["Empty response."])
    return (text, [])


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_structured_prompt(
    question_text: str,
    cutoff: date | None,
    *,
    vault_dir: str | Path | None = None,
    volume: float | None = None,
) -> str:
    """Build a prompt that requires the agent to research using its own tools.

    No context is pre-baked. The agent must read the vault, search the web,
    and consult past runs before emitting its forecast.
    """
    cutoff_str = cutoff.isoformat() if cutoff else "unknown"

    lines = [
        "=== FORECAST TASK ===",
        f"Question: {question_text}",
        f"Cutoff date: {cutoff_str}",
    ]
    if volume is not None and volume > 0:
        lines.append(f"Polymarket volume: ${volume:,.0f} USDC (higher = more liquid market signal)")
    lines += [
        "",
        "=== YOUR RESEARCH PROCESS ===",
        "You MUST complete the following research steps before forecasting.",
        "Use your available tools (read_file, search_files, web_search).",
        "",
        'Step 1 - FORECAST RULES: Read graph-vault/_forecast_instructions.md. This contains the evidence gate,',
        "  calibration data, and domain rules. Follow its instructions.",
        "",
        "Step 2 - PROCEDURE: Read graph-vault/_procedure.md for the research workflow.",
        "",
        "Step 3 - CONCEPTS: Search graph-vault/concepts/ for playbooks relevant",
        "  to this question type (politics, macro, culture, etc.).",
        '',
        "Step 4 - PAST RUNS: Use search_files to find analogous questions in",
        "  graph-vault/runs/. Compare predictions, outcomes, and what worked.",
        "",
        "Step 5 - KNOWLEDGE GRAPH: Read graph-vault/timeline/ for temporal context,",
        "  and graph-vault/entities/ for entity nodes relevant to this question (people, places,",
        "  organizations, domains). Search graph-vault/ for threads and related nodes.",
        "  Only read nodes relevant to the question — don't browse the whole vault.",
        "",
        "Step 6 - WEB RESEARCH: Use web_search and browser_navigate to research",
        f"  current context. You MUST respect the cutoff date ({cutoff_str})",
        "  and never use information from after that date.",
        "",
        "=== OUTPUT FORMAT (MANDATORY) ===",
        "After completing your research, respond with ONLY a single JSON object.",
        "No explanations, no markdown, no other text before or after.",
        '{"p_yes": 0.XX, "reasoning": "one-sentence summary"}',
        "p_yes must be a float between 0.0 and 1.0.",
        "",
    ]

    if vault_dir:
        lines.append(f"Vault directory: {vault_dir}")

    return "\n".join(lines)


def _build_orchestrator_prompt(
    question_text: str,
    cutoff: date | None,
    *,
    vault_dir: str | Path | None = None,
    volume: float | None = None,
) -> str:
    """Build a prompt that instructs the agent to orchestrate sub-agents via delegate_task.

    The agent reads _orchestrator_prerogatives.md, scans agent-roles/, selects
    relevant agents, spawns them in parallel, spawns the contrarian, then synthesizes.
    """
    cutoff_str = cutoff.isoformat() if cutoff else "unknown"

    lines = [
        "=== ORCHESTRATED FORECAST TASK ===",
        f"Question: {question_text}",
        f"Cutoff date: {cutoff_str}",
    ]
    if volume is not None and volume > 0:
        lines.append(f"Polymarket volume: ${volume:,.0f} USDC (higher = more liquid market signal)")
    lines += [
        "",
        "=== YOUR ROLE: ORCHESTRATOR ===",
        "You are not a standard forecaster — you are an ORCHESTRATOR.",
        "Your job is to MANAGE a team of sub-agents, each with a specialized perspective.",
        "",
        "=== YOUR WORKFLOW ===",
        "Execute these steps in order. Use your available tools (read_file, search_files,",
        "delegate_task, write_file, patch) to complete each step.",
        "",
        "Step 1 - READ YOUR PREROGATIVES: Read graph-vault/agent-roles/_orchestrator_prerogatives.md",
        "  This defines your meta-workflow. Follow it.",
        "",
        "Step 2 - SCAN THE AGENT ROSTER: Use search_files to list graph-vault/agent-roles/*.md",
        "  Read each role file's frontmatter (domain tags, region tags, trigger conditions).",
        "  Select 2-4 agents whose domain/region/triggers match the question.",
        "  CRITICAL: At minimum include an actor-simulation or regional specialist for the",
        "  relevant geography AND an analyst/theorist for the question's domain.",
        "",
        "Step 3 - SPAWN SUB-AGENTS (parallel):",
        "  Use delegate_task for each selected agent.",
        "  Each sub-agent receives: the full role file content, the question, the cutoff,",
        "  and vault access. Sub-agents are READ-WRITE — they create entity stubs, threads,",
        "  and concept files as their methodology prescribes.",
        "  Run them IN PARALLEL by passing all tasks in a single delegate_task call.",
        "",
        "Step 4 - COLLECT OUTPUTS: Each sub-agent returns its analysis (p_yes, reasoning)",
        "  and a log of vault edits (files created/modified).",
        "",
        "Step 5 - SPAWN CONTRARIAN: If you used 3+ sub-agents, spawn the contrarian-debater",
        "  with the sub-agents' outputs as context. The contrarian stress-tests assumptions",
        "  but does NOT produce its own forecast.",
        "",
        "Step 6 - SYNTHESIZE: Extract p_yes values, identify consensus/divergence,",
        "  apply the contrarian critique. Produce your final weighted estimate.",
        "",
        "Step 7 - REVIEW VAULT EDITS: Check what each sub-agent wrote to the vault.",
        "  Resolve conflicts (overlapping entity stubs, conflicing claims).",
        "  Approve good edits, fix bad ones.",
        "",
        "Step 8 - CHECK ROSTER GAPS: If no agent role exists for the question's domain",
        "  or region, create a minimal agent-role stub file.",
        "",
        "=== OUTPUT FORMAT (MANDATORY) ===",
        "After completing all 8 steps, respond with ONLY a single JSON object.",
        "No explanations, no markdown, no other text before or after.",
        """{
    "p_yes": 0.XX,
    "reasoning": "2-3 sentence summary including agents consulted and consensus",
    "agents_used": ["agent-name-1", "agent-name-2", ...],
    "vault_edits_summary": "comma-separated list of files created/modified by sub-agents",
    "key_assumptions": ["assumption 1", "assumption 2", ...],
    "roster_gaps_identified": ["gap 1", "gap 2"],
    "sub_agent_forecasts": [
        {"agent": "name", "p_yes": 0.XX, "confidence": "high|medium|low"},
        ...
    ]
}""",
        "p_yes must be a float between 0.0 and 1.0.",
        "",
    ]

    if vault_dir:
        lines.append(f"Vault directory: {vault_dir}")

    return "\n".join(lines)


def validate_orchestrated(raw: str) -> tuple[float, str, dict[str, Any], list[str]]:
    """Validate orchestrator output. Returns (p_yes, reasoning, metadata_dict, errors)."""
    payload = _extract_json(raw)
    if payload is None:
        return (0.5, "", {}, ["No valid JSON object found in output."])

    errors: list[str] = []
    py = payload.get("p_yes")
    if py is None:
        errors.append("Missing key 'p_yes'.")
        py = 0.5
    try:
        p = float(py)
    except (TypeError, ValueError):
        errors.append(f"'p_yes' is not a float: {py!r}.")
        p = 0.5
    if p < 0.0 or p > 1.0:
        errors.append(f"'p_yes'={p} outside [0, 1].")
        p = max(0.0, min(1.0, p))

    reason = str(payload.get("reasoning", "")).strip() or "orchestrated synthesis"

    metadata = {
        "agents_used": payload.get("agents_used", []),
        "vault_edits_summary": str(payload.get("vault_edits_summary", "")),
        "key_assumptions": payload.get("key_assumptions", []),
        "roster_gaps_identified": payload.get("roster_gaps_identified", []),
        "sub_agent_forecasts": payload.get("sub_agent_forecasts", []),
    }

    return (p, reason, metadata, errors)


# ---------------------------------------------------------------------------
# Orchestration log
# ---------------------------------------------------------------------------

_ORCHESTRA_LOG_DIR: Path | None = None


def _get_orch_log() -> Path:
    global _ORCHESTRA_LOG_DIR
    if _ORCHESTRA_LOG_DIR is None:
        # Default: data/orchestrator/ relative to project root
        _ORCHESTRA_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "orchestrator"
        _ORCHESTRA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _ORCHESTRA_LOG_DIR / "log.jsonl"


def _write_orch_log(entry: dict[str, Any]) -> None:
    """Append a structured orchestration run entry to the JSONL log."""
    log_path = _get_orch_log()
    entry["_logged_at"] = datetime.now().isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_orchestrated(
    question_text: str,
    cutoff: date | None = None,
    *,
    vault_dir: str | Path | None = None,
    policy_body: str = "",
    pit_context: str = "",
    question_id: str = "",
    source: str = "",
    category: str = "",
    resolution: bool | None = None,
    volume: float | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Run an orchestrated forecast using multi-agent sub-delegation.

    The orchestrator agent selects relevant sub-agents from graph-vault/agent-roles/,
    spawns them via delegate_task, collects their outputs, spawns the contrarian,
    synthesizes everything, and produces a final forecast.

    Returns (p_yes, reasoning, metadata_dict).
    metadata_dict contains: agents_used, vault_edits_summary, key_assumptions,
    roster_gaps_identified, sub_agent_forecasts.
    """
    prompt = _build_orchestrator_prompt(
        question_text, cutoff,
        vault_dir=vault_dir,
        volume=volume,
    )

    raw = _call_hermes(prompt, timeout=_ORCHESTRATOR_TIMEOUT)
    p_yes, reasoning, metadata, errors = validate_orchestrated(raw)

    if errors:
        raise RuntimeError(
            f"Orchestrated output validation failed for {question_text[:60]}...\n"
            + "\n".join(errors)
            + f"\nRaw (first 500): {raw[:500]}"
        )

    # Compute Brier if resolution known
    brier = None
    if resolution is not None:
        brier = (p_yes - (1.0 if resolution else 0.0)) ** 2

    # Write run note to vault
    if vault_dir is not None:
        note = RunNote(
            question_text=question_text,
            p_yes=p_yes,
            reasoning=f"[orchestrated] {reasoning}",
            cutoff=cutoff,
            source=source,
            category=category,
            brier=brier,
            resolution=resolution,
            question_id=question_id,
        )
        write_run(vault_dir, note)

    # Log to orchestration history
    try:
        log_entry = {
            "event": "orchestration_complete",
            "question": question_text[:120],
            "question_id": question_id,
            "cutoff": cutoff.isoformat() if cutoff else None,
            "p_yes": p_yes,
            "reasoning": reasoning,
            "brier": brier,
            "resolution": resolution,
            "source": source,
            "category": category,
            "agents_used": metadata.get("agents_used", []),
            "vault_edits_summary": metadata.get("vault_edits_summary", ""),
            "key_assumptions": metadata.get("key_assumptions", []),
            "roster_gaps_identified": metadata.get("roster_gaps_identified", []),
            "sub_agent_forecasts": metadata.get("sub_agent_forecasts", []),
        }
        _write_orch_log(log_entry)
    except Exception:
        pass  # logging failure shouldn't crash the forecast

    return p_yes, reasoning, metadata


def run_structured(
    question_text: str,
    cutoff: date | None = None,
    *,
    vault_dir: str | Path | None = None,
    policy_body: str = "",
    pit_context: str = "",
    question_id: str = "",
    source: str = "",
    category: str = "",
    resolution: bool | None = None,
    volume: float | None = None,
) -> tuple[float, str]:
    """Run a structured forecast (question + cutoff).

    The agent researches autonomously using its tools. No context is pre-baked
    except the question and cutoff. graph-vault is the agent's reference library.
    Writes run note to graph-vault/runs/ on completion.
    """
    prompt = _build_structured_prompt(
        question_text, cutoff,
        vault_dir=vault_dir,
        volume=volume,
    )

    raw = _call_hermes(prompt)
    p_yes, reasoning, errors = validate_structured(raw)

    if errors:
        raise RuntimeError(
            f"Structured output validation failed for {question_text[:60]}...\n"
            + "\n".join(errors)
            + f"\nRaw (first 500): {raw[:500]}"
        )

    # Compute Brier if resolution known
    brier = None
    if resolution is not None:
        brier = (p_yes - (1.0 if resolution else 0.0)) ** 2

    # Write run note to vault
    if vault_dir is not None:
        note = RunNote(
            question_text=question_text,
            p_yes=p_yes,
            reasoning=reasoning,
            cutoff=cutoff,
            source=source,
            category=category,
            brier=brier,
            resolution=resolution,
            question_id=question_id,
        )
        write_run(vault_dir, note)

    return p_yes, reasoning


def run_freeform(analysis_request: str) -> str:
    """Run a free-form analysis."""
    prompt = _build_freeform_prompt(analysis_request)
    raw = _call_hermes(prompt)
    text, errors = validate_freeform(raw)
    if errors:
        raise RuntimeError("Free-form output validation failed.\n" + "\n".join(errors))
    return text
