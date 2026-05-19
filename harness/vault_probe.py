"""Vault relevance probes: today-question per graph node, score retrieval, drive improvement."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from harness.config import VAULT_DIR

_HERMES_PROFILE = "forecasting"
_HERMES_TIMEOUT = 1200
_MIN_EXPLANATION_CHARS = 200
_DEFAULT_RELEVANCE_FLOOR = 0.4

_PROBE_TYPES = ("concepts", "threads", "entities", "timeline")
_PROBE_DIRS = {"concepts": "concept", "threads": "thread", "entities": "entity", "timeline": "timeline"}
_SKIP_PREFIXES = ("_",)
_INLINE_HERMES_TIMEOUT = 180
_MAX_CONTEXT_CHARS = 14_000

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*|\|[^\]]*)?\]\]")

_JSON_FENCE = re.compile(r"\{[\s\S]*\}")
_FRONTMATTER_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)

_QUESTIONS: dict[str, str] = {
    "concepts": (
        'As of {today}, why does the concept "{title}" matter for understanding current events? '
        "Explain its mechanisms using the vault and connect it to today."
    ),
    "threads": (
        'As of {today}, what is the current state of the thread "{title}" and which developments '
        "should a forecaster track? Use vault threads, quarters, and linked concepts."
    ),
    "entities": (
        'As of {today}, what role does "{title}" play in ongoing dynamics documented in the vault? '
        "Explain using threads and quarters, not a biography alone."
    ),
    "timeline": (
        "As of {today}, what from {title} remains live in the vault's threads and concepts? "
        "What would you carry forward for forecasting now?"
    ),
}

_FALLBACK_QUESTION = (
    'What is the relevance of "{title}" for understanding events as of {today}? '
    "Answer from the vault in detail."
)


@dataclass
class VaultNode:
    rel_path: str
    node_type: str
    slug: str
    title: str


@dataclass
class ProbeResult:
    node: VaultNode
    question: str
    today: str
    passed: bool
    verdict: str = "fail"  # keep | expand | reorganize | fail
    relevance_score: float | None = None
    relevance_floor: float = _DEFAULT_RELEVANCE_FLOOR
    disposition: str = ""
    merge_target: str = ""
    errors: list[str] = field(default_factory=list)
    vault_files_read: list[str] = field(default_factory=list)
    explanation: str = ""
    today_connection: str = ""
    gaps: str = ""
    raw_output: str = ""
    probe_mode: str = "tiered"


def _read_title(path: Path, slug: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return slug.replace("-", " ").title()
    m = _FRONTMATTER_TITLE.search(text)
    if m:
        return m.group(1).strip()
    hm = _HEADING.search(text)
    if hm:
        return hm.group(1).strip()
    return slug.replace("-", " ").title()


def enumerate_nodes(
    vault_dir: Path,
    *,
    types: tuple[str, ...] = _PROBE_TYPES,
) -> list[VaultNode]:
    vault = vault_dir.resolve()
    nodes: list[VaultNode] = []
    for folder in types:
        if folder not in _PROBE_DIRS:
            continue
        dir_path = vault / folder
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.glob("*.md")):
            if any(path.name.startswith(p) for p in _SKIP_PREFIXES):
                continue
            slug = path.stem
            nodes.append(
                VaultNode(
                    rel_path=f"{folder}/{path.name}",
                    node_type=_PROBE_DIRS[folder],
                    slug=slug,
                    title=_read_title(path, slug),
                )
            )
    return nodes


def question_for_node(node: VaultNode, today: date | None = None) -> str:
    today_s = (today or date.today()).isoformat()
    folder = node.rel_path.split("/")[0]
    template = _QUESTIONS.get(folder, _FALLBACK_QUESTION)
    return template.format(today=today_s, title=node.title)


def _wikilinks_in(text: str) -> list[str]:
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]


def _resolve_vault_path(vault: Path, target: str) -> Path | None:
    target = target.strip()
    for prefix in ("concepts", "threads", "entities", "timeline"):
        if target.startswith(f"{prefix}/"):
            p = vault / f"{target}.md" if not target.endswith(".md") else vault / target
            return p if p.is_file() else None
    for prefix in ("concepts", "threads", "entities", "timeline"):
        p = vault / prefix / f"{target}.md"
        if p.is_file():
            return p
    return None


def find_backlinks(vault_dir: Path, node: VaultNode, *, limit: int = 8) -> list[str]:
    """Paths of vault files that wikilink to this node (excluding target)."""
    vault = vault_dir.resolve()
    slug = node.slug
    rel = node.rel_path
    hits: list[str] = []
    for path in vault.rglob("*.md"):
        if path.name.startswith("_"):
            continue
        parts = path.relative_to(vault).parts
        if parts[0] in ("meta", "runs", "forecasts", "agent-roles"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f"[[{slug}]]" in text or f"[[{rel.replace('.md', '')}]]" in text or rel in text:
            r = str(path.relative_to(vault)).replace("\\", "/")
            if r != rel and r not in hits:
                hits.append(r)
            if len(hits) >= limit:
                break
    return hits


def gather_probe_context(vault_dir: Path, node: VaultNode) -> tuple[str, list[str]]:
    """Load target + 1-hop neighbors + backlinks locally (no Hermes tool loop)."""
    vault = vault_dir.resolve()
    paths_read: list[str] = []
    sections: list[str] = []
    budget = _MAX_CONTEXT_CHARS

    target_path = vault / node.rel_path
    if target_path.is_file():
        body = target_path.read_text(encoding="utf-8", errors="replace")[:5000]
        paths_read.append(node.rel_path)
        sections.append(f"### {node.rel_path}\n{body}")

    for link in _wikilinks_in(sections[0] if sections else ""):
        if budget <= 0:
            break
        resolved = _resolve_vault_path(vault, link)
        if not resolved:
            continue
        rel = str(resolved.relative_to(vault)).replace("\\", "/")
        if rel in paths_read or rel == node.rel_path:
            continue
        chunk = resolved.read_text(encoding="utf-8", errors="replace")[:2500]
        paths_read.append(rel)
        sections.append(f"### {rel}\n{chunk}")
        budget -= len(chunk)

    for rel in find_backlinks(vault, node, limit=5):
        if budget <= 0:
            break
        if rel in paths_read:
            continue
        p = vault / rel
        if not p.is_file():
            continue
        chunk = p.read_text(encoding="utf-8", errors="replace")[:2000]
        paths_read.append(rel)
        sections.append(f"### backlink: {rel}\n{chunk}")
        budget -= len(chunk)

    return "\n\n---\n\n".join(sections), paths_read


def heuristic_prefilter(
    node: VaultNode,
    vault_dir: Path,
    *,
    relevance_floor: float,
) -> ProbeResult | None:
    """Skip LLM for obvious orphan stubs."""
    path = vault_dir / node.rel_path
    if not path.is_file():
        return None
    body = path.read_text(encoding="utf-8", errors="replace")
    backlinks = find_backlinks(vault_dir, node)
    if node.node_type == "entity" and len(body) < 600 and len(backlinks) == 0:
        return ProbeResult(
            node=node,
            question=question_for_node(node),
            today=date.today().isoformat(),
            passed=False,
            verdict="reorganize",
            relevance_score=0.15,
            relevance_floor=relevance_floor,
            disposition="heuristic: orphan entity stub, no backlinks",
            merge_target="",
            vault_files_read=[node.rel_path],
            explanation="Auto: tiny entity file with no inbound links.",
            today_connection="Unlikely standalone forecasting value today.",
            probe_mode="heuristic",
        )
    return None


def build_probe_prompt_inline(
    node: VaultNode,
    question: str,
    context: str,
    context_paths: list[str],
    *,
    today: date | None = None,
) -> str:
    today_s = (today or date.today()).isoformat()
    paths_list = "\n".join(f"  - {p}" for p in context_paths)
    return f"""=== VAULT RELEVANCE PROBE (INLINE — NO TOOL SEARCH) ===

Today: {today_s}
Target node: {node.rel_path} ({node.node_type}: {node.title})

Question: {question}

Context below was pre-loaded from the vault (Python). Do NOT use read_file or search_files.
Answer ONLY from this context. You may use web_search briefly for today's news hook.

Files in context:
{paths_list}

=== VAULT CONTEXT ===
{context[:_MAX_CONTEXT_CHARS]}

Rate standalone worthiness (relevance_score 0.0–1.0):
  0.0–0.3  anecdote → merge into parent
  0.4–0.6  marginal standalone
  0.7–1.0  clear standalone forecasting value today

=== OUTPUT (JSON ONLY) ===
{{
  "vault_files_read": {json.dumps(context_paths[:12])},
  "explanation": "...",
  "today_connection": "...",
  "relevance_score": 0.75,
  "disposition": "keep|merge|demote",
  "disposition_reason": "...",
  "merge_target": "",
  "gaps": ""
}}
"""


def build_probe_prompt(
    node: VaultNode,
    question: str,
    *,
    vault_dir: Path,
    today: date | None = None,
) -> str:
    today_s = (today or date.today()).isoformat()
    vault = str(vault_dir.resolve())
    return f"""=== VAULT RELEVANCE PROBE (FULL AGENT SEARCH) ===

Today: {today_s}
Target node: {node.rel_path} ({node.node_type}: {node.title})

Question: {question}

You are testing whether the knowledge graph can answer a today-relevant question.
Research using graph-vault at {vault}.

Required steps:
1. Read the target node ({node.rel_path}) with read_file.
2. Follow wikilinks to threads/, concepts/, timeline/ as needed — prefer interactions over entity lists.
3. You MAY use web_search for current events to connect vault content to today ({today_s}).

Do NOT answer from parametric memory alone. Cite vault files you actually read.

Also rate whether this node **deserves to exist as its own file** vs belonging inline
in a parent thread, quarter section, or entity page (anecdote / subset content).

Relevance rubric (relevance_score 0.0–1.0):
  0.0–0.3  anecdote or detail — merge into parent; no standalone node warranted
  0.4–0.6  marginal — only keep if tightly linked to active forecasting threads
  0.7–1.0  clearly warrants standalone node for today's forecasting

=== OUTPUT (MANDATORY JSON ONLY) ===
{{
  "vault_files_read": ["concepts/example.md", "threads/example.md"],
  "explanation": "Multi-paragraph answer grounded in vault content",
  "today_connection": "How this node helps understand events as of {today_s}",
  "relevance_score": 0.75,
  "disposition": "keep|merge|demote",
  "disposition_reason": "Why this score; if merge/demote, why it is not standalone-worthy",
  "merge_target": "threads/parent.md or timeline/2024-Q3.md or empty if keep",
  "gaps": "Vault gaps that blocked a fuller answer, or empty string if adequate"
}}
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _JSON_FENCE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_relevance(payload: dict[str, Any]) -> float | None:
    raw = payload.get("relevance_score")
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def score_probe_response(
    node: VaultNode,
    payload: dict[str, Any] | None,
    *,
    min_chars: int = _MIN_EXPLANATION_CHARS,
    relevance_floor: float = _DEFAULT_RELEVANCE_FLOOR,
    context_paths: list[str] | None = None,
) -> tuple[bool, list[str], float | None]:
    errors: list[str] = []
    if payload is None:
        return False, ["No valid JSON in agent output."], None

    files = payload.get("vault_files_read") or []
    if not isinstance(files, list):
        files = []
    files_norm = [str(f).replace("\\", "/").lstrip("/") for f in files]

    explanation = str(payload.get("explanation", "")).strip()
    today_connection = str(payload.get("today_connection", "")).strip()
    combined = f"{explanation}\n{today_connection}"
    relevance = _parse_relevance(payload)

    if relevance is None:
        errors.append("Missing or invalid relevance_score (0.0–1.0 required).")

    if len(explanation) < min_chars:
        errors.append(f"Explanation too short ({len(explanation)} chars, need {min_chars}).")

    if not files_norm and not context_paths:
        errors.append("vault_files_read is empty — agent did not cite files read.")

    target = node.rel_path
    target_slug = node.slug
    if context_paths and target in context_paths:
        read_target = True
    else:
        read_target = any(
            target in f or f.endswith(f"{target_slug}.md") or f == target
            for f in files_norm
        )
    mentions_target = target_slug.replace("-", " ") in combined.lower() or node.title.lower() in combined.lower()
    if not read_target and not mentions_target:
        errors.append(f"Target node {target} not in vault_files_read and not clearly used in explanation.")

    if not today_connection:
        errors.append("Missing today_connection — must link vault to current relevance.")

    return len(errors) == 0, errors, relevance


def classify_verdict(
    retrieval_ok: bool,
    relevance: float | None,
    *,
    relevance_floor: float,
    disposition: str = "",
) -> str:
    """keep = good standalone; expand = fix content; reorganize = merge/demote; fail = broken probe."""
    if relevance is None:
        return "fail"

    if relevance < relevance_floor:
        return "reorganize"

    disp = disposition.strip().lower()
    if disp in ("merge", "demote", "inline", "subsection") and relevance < relevance_floor + 0.2:
        return "reorganize"

    if retrieval_ok:
        return "keep"

    return "expand"


def call_hermes(
    prompt: str,
    *,
    vault_dir: Path,
    timeout: int = _HERMES_TIMEOUT,
) -> str:
    if not shutil.which("hermes"):
        raise RuntimeError("hermes CLI not found on PATH")
    cmd = ["hermes", "-z", prompt, "--profile", _HERMES_PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(vault_dir))
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:500]
        raise RuntimeError(f"hermes failed (exit {result.returncode}): {err}")
    out = (result.stdout or "").strip()
    if not out:
        raise RuntimeError("hermes returned empty stdout")
    return out


def _finalize_probe(
    node: VaultNode,
    question: str,
    today_d: date,
    raw: str,
    *,
    relevance_floor: float,
    min_chars: int,
    context_paths: list[str] | None,
    probe_mode: str,
) -> ProbeResult:
    payload = _extract_json(raw)
    retrieval_ok, errors, relevance = score_probe_response(
        node, payload, min_chars=min_chars, relevance_floor=relevance_floor, context_paths=context_paths,
    )
    disposition = str(payload.get("disposition", "")) if payload else ""
    merge_target = str(payload.get("merge_target", "")) if payload else ""
    disposition_reason = str(payload.get("disposition_reason", "")) if payload else ""
    verdict = classify_verdict(
        retrieval_ok, relevance, relevance_floor=relevance_floor, disposition=disposition,
    )
    return ProbeResult(
        node=node,
        question=question,
        today=today_d.isoformat(),
        passed=verdict == "keep",
        verdict=verdict,
        relevance_score=relevance,
        relevance_floor=relevance_floor,
        disposition=disposition_reason or disposition,
        merge_target=merge_target,
        errors=errors if verdict != "keep" else [],
        vault_files_read=list(payload.get("vault_files_read") or context_paths or []) if payload else (context_paths or []),
        explanation=str(payload.get("explanation", "")) if payload else "",
        today_connection=str(payload.get("today_connection", "")) if payload else "",
        gaps=str(payload.get("gaps", "")) if payload else "",
        raw_output=raw[:4000],
        probe_mode=probe_mode,
    )


def run_probe(
    node: VaultNode,
    *,
    vault_dir: Path | None = None,
    today: date | None = None,
    min_chars: int = _MIN_EXPLANATION_CHARS,
    relevance_floor: float = _DEFAULT_RELEVANCE_FLOOR,
    mode: str = "tiered",
) -> ProbeResult:
    """Probe a node. Modes: tiered (default), inline, agent.

    tiered — heuristic → inline one-shot (context pre-loaded) → full search only if expand/fail
    inline — heuristic → inline only
    agent  — full Hermes vault search every time (slow)
    """
    vault = (vault_dir or VAULT_DIR).resolve()
    today_d = today or date.today()
    question = question_for_node(node, today_d)

    early = heuristic_prefilter(node, vault, relevance_floor=relevance_floor)
    if early is not None and mode != "agent":
        return early

    if mode in ("tiered", "inline"):
        context, context_paths = gather_probe_context(vault, node)
        prompt = build_probe_prompt_inline(
            node, question, context, context_paths, today=today_d,
        )
        try:
            raw = call_hermes(prompt, vault_dir=vault, timeout=_INLINE_HERMES_TIMEOUT)
            result = _finalize_probe(
                node, question, today_d, raw,
                relevance_floor=relevance_floor, min_chars=min_chars,
                context_paths=context_paths, probe_mode="inline",
            )
        except Exception as e:
            result = ProbeResult(
                node=node, question=question, today=today_d.isoformat(),
                passed=False, verdict="fail", relevance_floor=relevance_floor,
                errors=[str(e)], probe_mode="inline",
            )

        if mode == "inline" or result.verdict in ("keep", "reorganize"):
            return result

        # tiered: escalate expand/fail to full agent search
        prompt = build_probe_prompt(node, question, vault_dir=vault, today=today_d)
        try:
            raw = call_hermes(prompt, vault_dir=vault)
            return _finalize_probe(
                node, question, today_d, raw,
                relevance_floor=relevance_floor, min_chars=min_chars,
                context_paths=None, probe_mode="tiered+agent",
            )
        except Exception as e:
            result.errors.append(f"agent escalate: {e}")
            result.probe_mode = "tiered+agent-failed"
            return result

    prompt = build_probe_prompt(node, question, vault_dir=vault, today=today_d)
    try:
        raw = call_hermes(prompt, vault_dir=vault)
    except Exception as e:
        return ProbeResult(
            node=node, question=question, today=today_d.isoformat(),
            passed=False, verdict="fail", relevance_floor=relevance_floor,
            errors=[str(e)], probe_mode="agent",
        )
    return _finalize_probe(
        node, question, today_d, raw,
        relevance_floor=relevance_floor, min_chars=min_chars,
        context_paths=None, probe_mode="agent",
    )


def build_improvement_prompt(failures: list[ProbeResult], vault_dir: Path) -> str:
    vault = str(vault_dir.resolve())
    expand = [f for f in failures if f.verdict == "expand"]
    lines = [
        "=== VAULT RELEVANCE IMPROVEMENT (EXPAND) ===",
        "",
        "These nodes are **standalone-worthy** (relevance at or above floor) but the agent",
        "could not fetch/explain them well enough. Extend conjunctures — not entity bloat.",
        "",
        f"Vault: {vault}",
        "",
    ]
    for i, f in enumerate(expand[:20], 1):
        lines.append(f"## Expand {i}: {f.node.rel_path} (relevance={f.relevance_score})")
        lines.append(f"Question: {f.question}")
        lines.append(f"Errors: {'; '.join(f.errors)}")
        if f.gaps:
            lines.append(f"Agent-reported gaps: {f.gaps}")
        if f.explanation:
            lines.append(f"Partial explanation: {f.explanation[:600]}")
        lines.append("")
    if not expand:
        lines.append("(No expand candidates in this batch.)")
    lines += [
        "Fix timeline/, threads/, concepts/ so the next probe passes with verdict=keep.",
        "Do not create new standalone files for anecdotal content.",
    ]
    return "\n".join(lines)


def build_reorganize_prompt(candidates: list[ProbeResult], vault_dir: Path) -> str:
    vault = str(vault_dir.resolve())
    floor = candidates[0].relevance_floor if candidates else _DEFAULT_RELEVANCE_FLOOR
    lines = [
        "=== VAULT REORGANIZATION (BELOW RELEVANCE FLOOR) ===",
        "",
        f"These nodes scored below relevance floor {floor} or were flagged merge/demote.",
        "They are anecdotes or subsets — **do not expand them as standalone nodes**.",
        "",
        "Actions (pick per node):",
        "  - Merge content into merge_target (thread, quarter Conjuncture/Chronicle, parent entity)",
        "  - Demote to a subsection inside the parent file",
        "  - Delete the standalone file if fully redundant; fix wikilinks pointing to it",
        "  - Prefer one rich conjuncture over many isolated micro-nodes",
        "",
        f"Vault: {vault}",
        "",
    ]
    for i, f in enumerate(candidates[:25], 1):
        lines.append(f"## Reorganize {i}: {f.node.rel_path}")
        lines.append(f"  relevance_score: {f.relevance_score} (floor: {f.relevance_floor})")
        lines.append(f"  disposition: {f.disposition or '(none)'}")
        if f.merge_target:
            lines.append(f"  suggested merge_target: {f.merge_target}")
        lines.append(f"  reason: {f.disposition or '; '.join(f.errors) or 'below floor'}")
        if f.explanation:
            lines.append(f"  salvageable detail: {f.explanation[:400]}")
        lines.append("")
    lines += [
        "Report each change: merged into X, deleted Y, updated links in Z.",
        "Do not inflate low-relevance nodes — reorganize the graph.",
    ]
    return "\n".join(lines)


def probe_result_to_dict(result: ProbeResult) -> dict[str, Any]:
    d = asdict(result)
    d["node"] = asdict(result.node)
    return d


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def passed_node_keys(path: Path) -> set[str]:
    return {r["node"]["rel_path"] for r in load_results(path) if r.get("verdict") == "keep"}


def probed_node_keys(path: Path) -> set[str]:
    """All nodes that already have a probe row (any verdict)."""
    return {r["node"]["rel_path"] for r in load_results(path) if r.get("node")}


def reorganize_candidates(path: Path) -> list[dict[str, Any]]:
    return [r for r in load_results(path) if r.get("verdict") == "reorganize"]


def expand_candidates(path: Path) -> list[dict[str, Any]]:
    return [r for r in load_results(path) if r.get("verdict") == "expand"]
