"""Minimal orchestration: load questions, prompt hermes, store runs in vault.

Two modes:
- **Structured** (Polymarket / Metaculus):  question + cutoff -> JSON p_yes
- **Free-form** (direct analysis):  any text -> natural-language response
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

# PIT-filtered search (tool-level date enforcement)
try:
    from harness.tools.pit_search import search as pit_search, results_to_prompt_block
    _PIT_AVAILABLE = True
except ImportError:
    _PIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 600
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
    policy_body: str = "",
    pit_context: str = "",
) -> str:
    lines = ["Forecast this question.", ""]
    lines.append(f"Question: {question_text}")
    if cutoff:
        lines.append(f"Cutoff date: {cutoff.isoformat()}")
    lines.append("")

    if pit_context:
        lines.append(f"## PIT-filtered Context\n\n{pit_context}\n")

    if policy_body.strip():
        lines.append(f"## Policy Notes\n\n{policy_body[:2500]}\n")

    lines.append(
        "Your response must be a single JSON object with no other text:\n"
        '{"p_yes": 0.XX, "reasoning": "one-sentence summary"}\n'
        "p_yes must be a float between 0.0 and 1.0."
    )
    return "\n".join(lines)


def _build_freeform_prompt(analysis_request: str) -> str:
    return f"""Analyse the following:

{analysis_request}

Provide a concise, evidence-grounded analysis using your own research.
Output in natural language -- no JSON required."""


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

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
) -> tuple[float, str]:
    """Run a structured forecast (question + cutoff).

    Auto-runs PIT-filtered search, calls hermes, validates JSON output,
    writes run note to vault/runs/.
    Returns (p_yes, reasoning).
    """
    # Auto PIT search
    if not pit_context and cutoff and _PIT_AVAILABLE:
        try:
            resp = pit_search(question_text, cutoff, max_results=5)
            if not resp.error:
                pit_context = results_to_prompt_block(resp.results, cutoff)
        except Exception:
            pit_context = "(PIT search failed)"

    prompt = _build_structured_prompt(
        question_text, cutoff,
        policy_body=policy_body,
        pit_context=pit_context,
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
            pit_context=pit_context or "",
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
