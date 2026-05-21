"""Cognitive forecasting pipeline — type-aware multi-path reasoning.

Handles binary, numeric, categorical, and discrete questions through the
same pipeline: outside-view anchor → 3-path reasoning → Delphi → premortem → aggregate.

Usage:
    from harness.orchestrator_v2 import run_cognitive_pipeline

    result = run_cognitive_pipeline(
        question_text="What percentage of seats will Prosperity win?",
        cutoff=date.today(),
        vault_dir="graph-vault",
    )
    # result.output_type → OutputType.NUMERIC
    # result.value → 45.2 (if numeric)
    # result.distribution → {"A": 0.3, "B": 0.7} (if categorical)
    # result.p_yes → 0.65 (if binary)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from harness.outside_view import (
    OutsideViewAnchor,
    OutputType,
    detect_output_type,
    format_anchor_for_prompt,
    get_outside_view_anchor,
)
from harness.runs import RunNote, write_run
from harness.vault_pit import materialize_pit_snapshot, list_admissible_paths

_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200
_JSON_FENCE = re.compile(r"\{[\s\S]*\}")


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class SubAgentOutput:
    """Output from a single reasoning-path sub-agent."""
    role: str
    p_yes: float | None = None          # binary only
    value: float | None = None           # numeric only
    ci_low: float | None = None
    ci_high: float | None = None
    distribution: dict[str, float] | None = None  # categorical only
    reasoning: str = ""
    confidence: str = "medium"
    raw_json: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineResult:
    """Unified result for any output type."""
    output_type: OutputType
    # Binary
    p_yes: float | None = None
    # Numeric
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # Categorical / Discrete
    distribution: dict[str, float] | None = None
    # Shared
    reasoning: str = ""
    confidence: str = "medium"
    # Full trace
    outside_view: dict[str, Any] = field(default_factory=dict)
    sub_agent_outputs: list[dict[str, Any]] = field(default_factory=list)
    delphi_adjustments: list[dict[str, Any]] = field(default_factory=list)
    disconfirmation: dict[str, Any] = field(default_factory=dict)
    aggregation: dict[str, Any] = field(default_factory=dict)


# ── Hermes helpers ───────────────────────────────────────────────────


def _call_hermes(prompt: str, *, timeout: int = _HERMES_TIMEOUT, yolo: bool = False) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _HERMES_PROFILE]
    if yolo:
        cmd.append("--yolo")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _JSON_FENCE.search(text)
    blob = m.group(0) if m else text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        py_m = re.search(r'"p_yes"\s*:\s*([\d.]+)', blob)
        val_m = re.search(r'"value"\s*:\s*([\d.]+)', blob)
        dist_m = re.search(r'"distribution"\s*:\s*(\{[\s\S]*?\})', blob)
        if py_m:
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)', blob)
            return {"p_yes": float(py_m.group(1)), "reasoning": reason_m.group(1) if reason_m else ""}
        if val_m:
            return {"value": float(val_m.group(1))}
        if dist_m:
            try:
                return {"distribution": json.loads(dist_m.group(1))}
            except json.JSONDecodeError:
                pass
        return None


# ── Prompt builders ──────────────────────────────────────────────────


ROLE_NAMES = {
    "causal": "causal-model-forecaster",
    "analogical": "analogical-forecaster",
    "narrative": "narrative-scenario-forecaster",
}


def _build_sub_agent_prompt(
    role: str,
    question_text: str,
    cutoff: date,
    anchor_block: str,
    output_type: OutputType,
    *,
    vault_dir: str | Path,
    pit_manifest: list[str] | None = None,
    pit_brief_block: str = "",
) -> str:
    role_file = ROLE_NAMES.get(role, role)
    cutoff_str = cutoff.isoformat()
    vault_label = Path(str(vault_dir)).name

    type_hint = {
        OutputType.BINARY: "Produce a binary probability (p_yes). See 'Output Format' in your role file.",
        OutputType.NUMERIC: "Produce a numeric estimate (value + ci_low/ci_high). See 'Non-Binary Output Formats > Numeric' in your role file.",
        OutputType.CATEGORICAL: "Produce a probability distribution over choices. See 'Non-Binary Output Formats > Categorical' in your role file.",
        OutputType.DISCRETE: "Produce a probability distribution over ordered outcomes. See 'Non-Binary Output Formats > Categorical / Discrete' in your role file.",
    }

    lines = [
        f"=== {role.upper()} FORECASTER ===",
        f"Output type: {output_type.value}",
        f"Instruction: {type_hint.get(output_type, type_hint[OutputType.BINARY])}",
        "",
        f"Read your role: graph-vault/agent-roles/{role_file}.md",
        "Follow its methodology and output format EXACTLY.",
        "",
        f"Question: {question_text}",
        f"Cutoff: {cutoff_str}",
        "",
        anchor_block,
        "",
    ]

    if pit_brief_block:
        lines += ["=== PIT RESEARCH BRIEF ===", pit_brief_block, ""]

    lines += [
        "=== RESEARCH ===",
        f"Use read_file and search_files to research the vault at {vault_dir}.",
        "Read relevant concepts, threads, entities, and timeline files.",
        "Read _forecast_instructions.md for behavioral rules.",
        "",
        "=== OUTPUT ===",
        "Respond with ONLY the JSON specified in your role's Output Format section.",
        "No other text before or after the JSON.",
    ]

    return "\n".join(lines)


def _build_orchestrator_prompt(
    question_text: str,
    cutoff: date,
    anchor: OutsideViewAnchor,
    output_type: OutputType,
    decomposition: dict[str, Any],
    sub_agent_outputs: list[dict[str, Any]],
    *,
    vault_dir: str | Path,
    question_id: str = "",
    source: str = "",
) -> str:
    vault_label = Path(str(vault_dir)).name
    anchor_block = format_anchor_for_prompt(anchor)

    # Format sub-agent outputs
    sa_lines = []
    for i, sa in enumerate(sub_agent_outputs):
        role = sa.get("role", f"agent-{i}")
        reasoning = sa.get("reasoning", "")[:300]
        confidence = sa.get("confidence", "medium")

        if output_type == OutputType.BINARY:
            py = sa.get("p_yes", "N/A")
            sa_lines.append(f"### {role} (p_yes={py}, confidence={confidence})")
        elif output_type == OutputType.NUMERIC:
            v = sa.get("value", "N/A")
            cl = sa.get("ci_low", "?")
            ch = sa.get("ci_high", "?")
            sa_lines.append(f"### {role} (value={v}, CI=[{cl}, {ch}], confidence={confidence})")
        else:
            dist = sa.get("distribution", {})
            dist_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(dist.items())[:5])
            sa_lines.append(f"### {role} (distribution: {dist_str}, confidence={confidence})")
        sa_lines.append(f"  Reasoning: {reasoning}")
        sa_lines.append("")

    decon_block = json.dumps(decomposition, indent=2) if decomposition else "(not available)"

    # Type-specific aggregation instructions
    if output_type == OutputType.BINARY:
        agg_instr = (
            "Produce a final weighted p_yes:\n"
            "- Outside-view base rate: minimum 35% weight (the anchor)\n"
            "- Causal model: weight by weakest-link identification quality\n"
            "- Analogical: weight by structural similarity of best analog\n"
            "- Narrative: weight by premortem quality\n"
            "If all 3 sub-agents agree within ±0.10, reduce weight on unanimity\n"
            "  (agreement may signal shared bias, not accuracy).\n"
            "If they diverge by >0.25, investigate why before averaging."
        )
        output_schema = (
            '"p_yes": 0.XX,\n'
            '  "reasoning": "synthesis of cognitive paths + outside view",\n'
        )
    elif output_type == OutputType.NUMERIC:
        agg_instr = (
            "Produce a final numeric estimate with confidence interval:\n"
            "- Trimmed mean of sub-agent values (discard outlier if one agent is far off)\n"
            "- CI should span from the most plausible low to high across agents\n"
            "- Wider CI if agents disagree significantly\n"
            "- If outside-view has similar cases with numeric data, use as additional anchor"
        )
        output_schema = (
            '"value": 2.1,\n'
            '  "ci_low": 1.8,\n'
            '  "ci_high": 2.4,\n'
            '  "reasoning": "synthesis of cognitive paths",\n'
        )
    else:
        agg_instr = (
            "Produce a final probability distribution over choices:\n"
            "- Blend sub-agent distributions with performance-based weighting\n"
            "- Normalize to sum to 1.0\n"
            "- If agents agree on the top choice but disagree on probability, average the probabilities\n"
            "- If agents disagree on which choice leads, investigate the structural disagreement\n"
            "- Outside-view base rates (if available) as anchor distribution"
        )
        output_schema = (
            '"distribution": {"choice_a": 0.35, "choice_b": 0.40},\n'
            '  "reasoning": "synthesis of cognitive paths",\n'
        )

    lines = [
        "=== COGNITIVE ORCHESTRATOR ===",
        "",
        f"Output type: {output_type.value}",
        f"Question: {question_text}",
        f"Cutoff: {cutoff.isoformat()}" if cutoff else "Cutoff: today",
        "",
        anchor_block,
        "",
        "=== DECOMPOSITION ===",
        decon_block,
        "",
        "=== SUB-AGENT FORECASTS (3 independent reasoning paths) ===",
        "\n".join(sa_lines),
        "",
        "=== STAGE 3: DELPHI ITERATION ===",
        "Each sub-agent produced its forecast independently. Now consider:",
        "1. Where do they agree? Is the agreement genuine or shared bias?",
        "2. Where do they diverge? Which divergence is most informative?",
        "3. If they were to see each other's reasoning, how would each revise?",
        "For each agent, produce a revised estimate based on cross-pollination of reasoning.",
        "",
        "=== STAGE 4: DISCONFIRMATION GATE ===",
        "1. PREMORTEM: 'Assume the consensus forecast is completely wrong.",
        "   What causal pathway did we miss? What evidence did we underweight?'",
        "2. DEVIL'S ADVOCATE: 'What is the strongest case against the consensus?'",
        "3. KEY ASSUMPTIONS: List the 3-5 assumptions that, if wrong, would",
        "   most change the forecast. Rate each assumption's fragility.",
        "",
        "=== STAGE 5: AGGREGATION ===",
        agg_instr,
        "",
        "=== OUTPUT FORMAT (MANDATORY) ===",
        "Respond with ONLY a single JSON object:",
        "{",
        f'  {output_schema}'
        '  "delphi_revisions": [',
        '    {"role": "...", "original": {...}, "revised": {...}, "rationale": "..."}',
        "  ],",
        '  "disconfirmation": {',
        '    "premortem": "what missed pathway was found",',
        '    "devils_advocate": "strongest case against consensus",',
        '    "key_assumptions": [{"assumption": "...", "fragility": "high|medium|low", "effect_if_wrong": "..."}]',
        "  },",
        '  "aggregation": {',
        '    "method": "weighted_blend|trimmed_mean|distribution_blend",',
        '    "weights": {"causal": 0.XX, "analogical": 0.XX, "narrative": 0.XX},',
        '    "consensus_level": "high|medium|low|divergent"',
        "  },",
        '  "confidence": "high|medium|low"',
        "}",
    ]

    return "\n".join(lines)


# ── PIT preparation ──────────────────────────────────────────────────


def _prepare_pit_vault(
    source_vault: Path,
    cutoff: date,
) -> tuple[Path, list[str], tempfile.TemporaryDirectory[str] | None]:
    manifest = list_admissible_paths(source_vault, cutoff)
    if not manifest:
        return source_vault, [], None
    tmp = tempfile.TemporaryDirectory(prefix="pit-cognitive-")
    dest = Path(tmp.name)
    copied = materialize_pit_snapshot(source_vault, dest, cutoff)
    return dest, copied, tmp


# ── Sub-agent execution ──────────────────────────────────────────────


def _decompose_question(question_text: str, cutoff: date) -> dict[str, Any]:
    try:
        prompt = (
            "=== DECOMPOSITION TASK ===\n\n"
            f"Question: {question_text}\n\n"
            "Decompose this question. Respond with ONLY JSON:\n"
            "{\n"
            '  "sub_questions": ["...", "..."],\n'
            '  "preconditions": ["condition needed", ...],\n'
            '  "blockers": ["what could prevent", ...],\n'
            '  "time_window_days": N,\n'
            '  "domains": ["geopolitics", ...],\n'
            '  "event_type": "ceasefire|election|...",\n'
            '  "output_type": "binary|numeric|categorical|discrete"\n'
            "}"
        )
        raw = _call_hermes(prompt, timeout=120, yolo=True)
        return _extract_json(raw) or _fallback_decompose(question_text)
    except Exception:
        return _fallback_decompose(question_text)


def _fallback_decompose(question_text: str) -> dict[str, Any]:
    from harness.outside_view import classify_question
    event_type, domain = classify_question(question_text)
    output_type = detect_output_type(question_text).value
    return {
        "sub_questions": [question_text],
        "preconditions": ["(unknown)"],
        "blockers": ["(unknown)"],
        "time_window_days": 90,
        "domains": [domain],
        "event_type": event_type,
        "output_type": output_type,
        "_fallback": True,
    }


def _run_sub_agents(
    question_text: str,
    cutoff: date,
    anchor_block: str,
    output_type: OutputType,
    *,
    vault_dir: Path,
    pit_manifest: list[str] | None = None,
    pit_brief_block: str = "",
) -> list[SubAgentOutput]:
    """Run 3 cognitive sub-agents sequentially with independent context.

    In a delegate_task-enabled environment these would run in true parallel.
    For now, sequential with independent context ensures no cross-contamination.
    """
    outputs: list[SubAgentOutput] = []

    for role in ("causal", "analogical", "narrative"):
        try:
            prompt = _build_sub_agent_prompt(
                role, question_text, cutoff, anchor_block, output_type,
                vault_dir=vault_dir, pit_manifest=pit_manifest,
                pit_brief_block=pit_brief_block,
            )
            raw = _call_hermes(prompt, timeout=600)
            sa_json = _extract_json(raw)

            if sa_json is None:
                outputs.append(SubAgentOutput(
                    role=role, reasoning="JSON parse failed",
                    confidence="low", error="no JSON in output",
                ))
                continue

            sa = SubAgentOutput(role=role, raw_json=sa_json)

            if output_type == OutputType.BINARY:
                py = sa_json.get("p_yes", 0.5)
                try:
                    sa.p_yes = max(0.0, min(1.0, float(py)))
                except (TypeError, ValueError):
                    sa.p_yes = 0.5
            elif output_type == OutputType.NUMERIC:
                v = sa_json.get("value")
                if v is not None:
                    try:
                        sa.value = float(v)
                    except (TypeError, ValueError):
                        sa.value = None
                cl = sa_json.get("ci_low")
                ch = sa_json.get("ci_high")
                if cl is not None:
                    try:
                        sa.ci_low = float(cl)
                    except (TypeError, ValueError):
                        pass
                if ch is not None:
                    try:
                        sa.ci_high = float(ch)
                    except (TypeError, ValueError):
                        pass
            else:
                dist = sa_json.get("distribution")
                if isinstance(dist, dict):
                    # Normalize
                    total = sum(dist.values())
                    if total > 0:
                        sa.distribution = {k: v / total for k, v in dist.items()}
                    else:
                        sa.distribution = dist

            sa.reasoning = str(sa_json.get("reasoning", ""))[:500]
            sa.confidence = str(sa_json.get("confidence", "medium"))
            outputs.append(sa)

        except Exception as e:
            outputs.append(SubAgentOutput(
                role=role, reasoning="", confidence="low", error=str(e)[:500],
            ))

    return outputs


def _sub_agent_to_dict(sa: SubAgentOutput) -> dict[str, Any]:
    d = {"role": sa.role, "reasoning": sa.reasoning, "confidence": sa.confidence}
    if sa.p_yes is not None:
        d["p_yes"] = sa.p_yes
    if sa.value is not None:
        d["value"] = sa.value
    if sa.ci_low is not None:
        d["ci_low"] = sa.ci_low
    if sa.ci_high is not None:
        d["ci_high"] = sa.ci_high
    if sa.distribution is not None:
        d["distribution"] = sa.distribution
    if sa.error:
        d["error"] = sa.error
    d["raw_json"] = sa.raw_json
    return d


# ── Main pipeline ────────────────────────────────────────────────────


def run_cognitive_pipeline(
    question_text: str,
    cutoff: date | None = None,
    *,
    vault_dir: str | Path | None = None,
    output_type: OutputType | None = None,
    question_id: str = "",
    source: str = "",
    category: str = "",
    resolution: bool | None = None,
    volume: float | None = None,
    enforce_pit: bool = True,
    query_polymarket: bool = True,
) -> PipelineResult:
    """Run the type-aware cognitive forecasting pipeline.

    Handles binary, numeric, categorical, and discrete questions.
    Detects output_type automatically if not provided.
    """
    cutoff = cutoff or date.today()
    source_vault = Path(vault_dir).resolve() if vault_dir else None
    if source_vault is None:
        raise ValueError("vault_dir is required")

    # ── Detect output type ──────────────────────────────────────
    if output_type is None:
        output_type = detect_output_type(question_text)

    # ── Stage 0: Decomposition ──────────────────────────────────
    decomposition = _decompose_question(question_text, cutoff)

    # ── Stage 1: Outside-view anchor ────────────────────────────
    anchor = get_outside_view_anchor(
        question_text, source_vault, query_polymarket=query_polymarket,
    )
    anchor_block = format_anchor_for_prompt(anchor)

    # ── PIT snapshot ────────────────────────────────────────────
    pit_tmp: tempfile.TemporaryDirectory[str] | None = None
    pit_research_tmp: tempfile.TemporaryDirectory[str] | None = None
    research_vault = source_vault
    pit_manifest: list[str] | None = None
    pit_brief_block = ""

    if enforce_pit and source_vault is not None:
        research_vault, pit_manifest, pit_tmp = _prepare_pit_vault(source_vault, cutoff)
        try:
            from harness.pit_research import run_pit_research
            brief, pit_research_tmp = run_pit_research(
                question_text, cutoff, vault_dir=source_vault, use_snapshot=True,
            )
            pit_brief_block = brief.to_prompt_block()
        except Exception:
            pit_brief_block = ""

    try:
        # ── Stage 2: Multi-path sub-agents ──────────────────────
        sub_outputs = _run_sub_agents(
            question_text, cutoff, anchor_block, output_type,
            vault_dir=research_vault, pit_manifest=pit_manifest,
            pit_brief_block=pit_brief_block,
        )
        sa_dicts = [_sub_agent_to_dict(sa) for sa in sub_outputs]

        # ── Stages 3-6: Orchestrator synthesis ──────────────────
        orch_prompt = _build_orchestrator_prompt(
            question_text, cutoff, anchor, output_type, decomposition, sa_dicts,
            vault_dir=research_vault, question_id=question_id, source=source,
        )
        raw = _call_hermes(orch_prompt, timeout=_HERMES_TIMEOUT)
        orch_json = _extract_json(raw)
    finally:
        if pit_tmp is not None:
            pit_tmp.cleanup()
        if pit_research_tmp is not None:
            pit_research_tmp.cleanup()

    # ── Parse orchestrator output ───────────────────────────────
    if orch_json is None:
        orch_json = _fallback_orchestrator(sub_outputs, output_type, anchor)

    reasoning = str(orch_json.get("reasoning", "cognitive pipeline synthesis"))
    confidence = str(orch_json.get("confidence", "medium"))

    result = PipelineResult(
        output_type=output_type,
        reasoning=reasoning,
        confidence=confidence,
        outside_view={
            "event_type": anchor.event_type,
            "domain": anchor.domain,
            "output_type": output_type.value,
            "strategy": anchor.anchoring_strategy,
            "polymarket_price": anchor.polymarket.price if anchor.polymarket else None,
        },
        sub_agent_outputs=sa_dicts,
        delphi_adjustments=orch_json.get("delphi_revisions", []),
        disconfirmation=orch_json.get("disconfirmation", {}),
        aggregation=orch_json.get("aggregation", {}),
    )

    if output_type == OutputType.BINARY:
        py = orch_json.get("p_yes", 0.5)
        try:
            result.p_yes = max(0.0, min(1.0, float(py)))
        except (TypeError, ValueError):
            result.p_yes = 0.5
    elif output_type == OutputType.NUMERIC:
        v = orch_json.get("value")
        if v is not None:
            try:
                result.value = float(v)
            except (TypeError, ValueError):
                pass
        cl = orch_json.get("ci_low")
        ch = orch_json.get("ci_high")
        if cl is not None:
            try:
                result.ci_low = float(cl)
            except (TypeError, ValueError):
                pass
        if ch is not None:
            try:
                result.ci_high = float(ch)
            except (TypeError, ValueError):
                pass
    else:
        dist = orch_json.get("distribution")
        if isinstance(dist, dict):
            total = sum(dist.values())
            result.distribution = {k: v / total for k, v in dist.items()} if total > 0 else dist

    # ── Write run note ──────────────────────────────────────────
    brier = None
    if resolution is not None and result.p_yes is not None:
        brier = (result.p_yes - (1.0 if resolution else 0.0)) ** 2

    if vault_dir is not None:
        summary = reasoning
        if result.p_yes is not None:
            summary = f"[cognitive] p_yes={result.p_yes:.3f} {reasoning}"
        elif result.value is not None:
            summary = f"[cognitive] value={result.value:.2f} CI=[{result.ci_low},{result.ci_high}] {reasoning}"
        elif result.distribution is not None:
            dist_str = ", ".join(f"{k}:{v:.1%}" for k, v in sorted(result.distribution.items())[:3])
            summary = f"[cognitive] dist={{{dist_str}}} {reasoning}"

        note = RunNote(
            question_text=question_text,
            p_yes=result.p_yes or 0.5,
            reasoning=summary,
            cutoff=cutoff,
            source=source,
            category=category,
            brier=brier,
            resolution=resolution,
            question_id=question_id,
            pit_context=pit_brief_block[:4000] if pit_brief_block else "",
        )
        write_run(vault_dir, note)

    return result


def _fallback_orchestrator(
    sub_outputs: list[SubAgentOutput],
    output_type: OutputType,
    anchor: OutsideViewAnchor,
) -> dict[str, Any]:
    """Fallback aggregation when orchestrator JSON parse fails."""
    valid = [sa for sa in sub_outputs if sa.error is None]
    result: dict[str, Any] = {
        "reasoning": "Fallback: sub-agent aggregation (orchestrator JSON parse failed)",
        "confidence": "low",
        "fallback": True,
    }

    if output_type == OutputType.BINARY:
        if valid:
            p_vals = [sa.p_yes for sa in valid if sa.p_yes is not None]
            result["p_yes"] = sum(p_vals) / len(p_vals) if p_vals else 0.5
        else:
            result["p_yes"] = anchor.binary.base_rate if anchor.binary else 0.5

    elif output_type == OutputType.NUMERIC:
        vals = [sa.value for sa in valid if sa.value is not None]
        if vals:
            result["value"] = sum(vals) / len(vals)
            lows = [sa.ci_low for sa in valid if sa.ci_low is not None]
            highs = [sa.ci_high for sa in valid if sa.ci_high is not None]
            result["ci_low"] = min(lows) if lows else None
            result["ci_high"] = max(highs) if highs else None
        else:
            result["value"] = None

    else:
        # Categorical: average distributions
        dists = [sa.distribution for sa in valid if sa.distribution is not None]
        if dists:
            keys = set()
            for d in dists:
                keys.update(d.keys())
            avg = {}
            for k in keys:
                vals_for_k = [d.get(k, 0.0) for d in dists]
                avg[k] = sum(vals_for_k) / len(vals_for_k)
            total = sum(avg.values())
            result["distribution"] = {k: v / total for k, v in avg.items()} if total > 0 else avg
        else:
            result["distribution"] = {}

    return result
