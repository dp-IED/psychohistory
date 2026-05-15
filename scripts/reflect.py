"""Post-backtest reflection: analyze performance with an LLM, update vault + policy."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from harness.calibration import compute_calibration
from harness.memory_schema import EpisodicRecord
from harness.memory_store import JsonlMemoryStore
from harness.policy_loader import PolicyConfig, load_policy, save_policy
from harness.tools.strategy_runtime import STRATEGY_FILENAME, load_strategy_markdown

WORST_EPISODE_COUNT = 3
BATCH_WORST_WINDOW = 20  # how many recent episodes to consider the "current batch"
RECENT_EPISODE_COUNT = 30  # how many recent scored episodes to include in reflection context


def _call_llm_cursor(prompt: str, model: str = "composer-2-fast") -> str:
    """Call Cursor Agent in headless mode — low startup overhead, built for editing."""
    agent_bin = shutil.which("agent")
    if not agent_bin:
        raise RuntimeError("cursor agent CLI not found on PATH")

    cmd = [agent_bin, "--model", model, "--print", "--trust", "--output-format", "text", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr[:500] if result.stderr else ""
        raise RuntimeError(f"Cursor call failed (exit {result.returncode}): {err}")

    out = result.stdout.strip() if result.stdout else ""
    if not out:
        raise RuntimeError("Cursor returned empty stdout")
    return out


def _call_llm_hermes(prompt: str, model: str, provider: str = "opencode-go") -> str:
    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        raise RuntimeError("hermes CLI not found on PATH")

    cmd = [hermes_bin, "-z", prompt, "-m", model, "--provider", provider]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr[:500] if result.stderr else ""
        raise RuntimeError(f"Hermes call failed (exit {result.returncode}): {err}")

    out = result.stdout.strip() if result.stdout else ""
    if not out:
        raise RuntimeError("Hermes returned empty stdout")
    return out


def _call_llm_openrouter(prompt: str, model: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You reorganize a forecasting agent's Obsidian vault and machine policy. "
                    "Respond with a single JSON object only (no markdown fences). "
                    "Keys: policy_markdown (string, full policy file: YAML frontmatter between --- "
                    "with only machine fields blind_spot_checks, max_steps, convergence_epsilon, shrinkage; "
                    "no markdown body or use empty body after closing ---), "
                    "strategy_markdown (string, full _strategy.md content for the vault), "
                    "vault_files (array of objects: each has path relative to vault root, "
                    "either content string for upsert or delete=true)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.35,
    }
    body = json.dumps(payload).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        err_blob = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {err_blob}") from exc

    data = json.loads(raw)
    choice = data["choices"][0]["message"]["content"]
    if choice is None or (isinstance(choice, str) and not choice.strip()):
        raise RuntimeError("OpenRouter returned empty content")
    return choice if isinstance(choice, str) else str(choice)


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    lines = raw.splitlines()
    if not lines:
        return raw
    first = lines[0].strip()
    if first.startswith("```"):
        lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("LLM output did not contain JSON object")
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start : i + 1]
                return json.loads(blob)
    raise ValueError("Unbalanced JSON object in LLM output")


def _safe_vault_path(vault_root: Path, relative: str) -> Path:
    root = vault_root.expanduser().resolve()
    rel = relative.strip().lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise ValueError(f"unsafe vault path: {relative!r}")
    out = (root / rel).resolve()
    out.relative_to(root)
    return out


def _gather_markdown_dir(vault_root: Path, subdir: str, cap_per_file: int = 14_000) -> str:
    d = vault_root / subdir
    if not d.is_dir():
        return f"({subdir}/ missing or empty)\n"

    chunks: list[str] = []
    for path in sorted(d.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if len(text) > cap_per_file:
            text = text[:cap_per_file] + "\n…\n"
        chunks.append(f"#### File `{path.name}`\n{text}\n")
    return "\n".join(chunks) if chunks else f"(no .md files in {subdir}/)\n"


def build_open_reflection_prompt(
    *,
    runs_digest: str,
    approaches_digest: str,
    strategy_content: str,
    policy_file_raw: str,
    overall_brier: float,
    by_category: dict[str, float],
    shrinkage: str,
    recent_episodes_block: str = "",
) -> str:
    cat_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in sorted(by_category.items()))
    recent_fallback = f"(fewer than {RECENT_EPISODE_COUNT} episodes available)"
    return f"""You are a forecasting agent reviewing your recent performance. Your goal: make better forecasts next time.

## Performance summary

- Overall Brier: {overall_brier:.4f}
- By category: {cat_lines or '  (none)'}
- Shrinkage: {shrinkage}

## Recent episodes (last {RECENT_EPISODE_COUNT} scored)

{recent_episodes_block or recent_fallback}

## Vault contents

### Runs (in runs/)
{runs_digest}

### Approaches (in approaches/)
{approaches_digest}

### Strategy (in {STRATEGY_FILENAME})
{strategy_content}

## Policy (YAML frontmatter + markdown body)

{policy_file_raw}

## What to do

Identify 1-2 concrete failure patterns. Propose specific changes to policy or vault that would fix them.
Suggest one new information-gathering approach the agent hasn't tried yet.

Return JSON only with keys: policy_markdown, strategy_markdown, vault_files (see system schema)."""


def apply_reflection_payload(
    payload: dict[str, Any],
    *,
    policy_path: Path,
    vault_dir: Path | None,
    dry_run: bool,
) -> list[str]:
    """Apply vault upserts/deletes and policy save. Returns log lines."""

    log: list[str] = []
    policy_md = payload.get("policy_markdown")
    if not isinstance(policy_md, str) or not policy_md.strip():
        raise ValueError("reflection JSON missing policy_markdown")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(policy_md.strip() + "\n")
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        parsed_policy = load_policy(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not dry_run:
        history_dir = policy_path.parent / "policy_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if policy_path.exists():
            archive = history_dir / f"policy-{stamp}.md"
            archive.write_text(policy_path.read_text(encoding="utf-8"), encoding="utf-8")
        save_policy(parsed_policy, policy_path)
        log.append(f"policy updated -> {policy_path}")

    strategy_md = payload.get("strategy_markdown")
    if vault_dir is not None and isinstance(strategy_md, str) and strategy_md.strip():
        strat_path = vault_dir.expanduser().resolve() / STRATEGY_FILENAME
        if not dry_run:
            strat_path.parent.mkdir(parents=True, exist_ok=True)
            strat_path.write_text(strategy_md.strip() + "\n", encoding="utf-8")
            log.append(f"strategy updated -> {strat_path}")

    files = payload.get("vault_files") or []
    if vault_dir is None or not isinstance(files, list):
        return log

    root = vault_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for item in files:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        target = _safe_vault_path(root, rel)
        if item.get("delete") is True:
            if dry_run:
                log.append(f"would delete {target}")
            elif target.is_file():
                target.unlink()
                log.append(f"deleted {target}")
            continue

        content = item.get("content")
        if not isinstance(content, str):
            continue
        if dry_run:
            log.append(f"would write {target} ({len(content)} chars)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log.append(f"wrote {target}")

    return log


def _format_episode_excerpt(ep: EpisodicRecord) -> str:
    b = ep.brier_score if ep.brier_score is not None else -1.0
    return (
        f"- job_id={ep.job_id}\n"
        f"  market_family={ep.market_family}\n"
        f"  question={ep.question!r}\n"
        f"  final_p_yes={ep.final_p_yes:.4f}\n"
        f"  brier_score={b:.4f}\n"
        f"  fired={ep.blind_spot_checks_fired}\n"
        f"  skipped={ep.blind_spot_checks_skipped}\n"
        f"  notes={ep.notes[:120]!r}{'…' if len(ep.notes) > 120 else ''}\n"
    )


def _checks_rollup(episodes: list[EpisodicRecord]) -> str:
    fired_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()
    for ep in episodes:
        for c in ep.blind_spot_checks_fired:
            fired_counter[c] += 1
        for c in ep.blind_spot_checks_skipped:
            skipped_counter[c] += 1
    if not fired_counter and not skipped_counter:
        return "No blind_spot check labels across episodes."
    lines = ["Per-check counts (fired / skipped):"]
    all_checks = sorted(set(fired_counter) | set(skipped_counter))
    for name in all_checks:
        lines.append(f"  {name}: fired={fired_counter[name]} skipped={skipped_counter[name]}")
    return "\n".join(lines)


def _select_worst_episodes(resolved: list[EpisodicRecord], n: int) -> list[EpisodicRecord]:
    with_brier = [e for e in resolved if e.brier_score is not None]
    sorted_ep = sorted(with_brier, key=lambda e: float(e.brier_score or 0.0), reverse=True)
    if not sorted_ep:
        return []
    take = min(n, len(sorted_ep))
    return sorted_ep[:take]


def run_reflection(
    *,
    memory_dir: Path,
    policy_path: Path,
    vault_dir: Path | None,
    runs_subdir: str,
    approaches_subdir: str,
    backend: str,
    model: str,
    provider: str,
    dry_run: bool,
) -> tuple[PolicyConfig, str, dict[str, Any]]:
    memory_store = JsonlMemoryStore(memory_dir.expanduser().resolve())
    policy_path = policy_path.expanduser().resolve()
    old_policy_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""

    calibration = compute_calibration(memory_store)
    episodes = memory_store.read_all_episodes()
    resolved = [e for e in episodes if e.brier_score is not None]

    by_family: dict[str, list[float]] = defaultdict(list)
    for e in resolved:
        if e.brier_score is not None:
            by_family[e.market_family].append(float(e.brier_score))
    by_category = {fam: mean(scores) for fam, scores in by_family.items()}

    if not calibration.insufficient_data:
        overall_brier = float(calibration.overall_brier)
    elif resolved:
        overall_brier = mean(float(e.brier_score) for e in resolved if e.brier_score is not None)
    else:
        overall_brier = 0.0

    shrinkage_line = "None (insufficient calibration data)"
    if not calibration.insufficient_data:
        shrinkage_line = f"suggested_shrinkage={calibration.suggested_shrinkage:.4f}"
    pc = load_policy(policy_path)
    if pc.shrinkage is not None:
        shrinkage_line += f"; policy.shrinkage={pc.shrinkage:.4f}"

    runs_digest = "(vault not configured)\n"
    approaches_digest = "(vault not configured)\n"
    strategy_content = "(vault not configured)\n"
    policy_file_raw = old_policy_text or "(missing file)\n"

    if vault_dir is not None:
        vr = vault_dir.expanduser().resolve()
        vr.mkdir(parents=True, exist_ok=True)
        (vr / runs_subdir).mkdir(parents=True, exist_ok=True)
        (vr / approaches_subdir).mkdir(parents=True, exist_ok=True)
        runs_digest = _gather_markdown_dir(vr, runs_subdir)
        approaches_digest = _gather_markdown_dir(vr, approaches_subdir)
        strategy_content = load_strategy_markdown(vr) or "(file missing — create it)\n"

    # Build recent episodes block (last RECENT_EPISODE_COUNT scored episodes in detail)
    recent_window = resolved[-RECENT_EPISODE_COUNT:] if len(resolved) > RECENT_EPISODE_COUNT else resolved
    recent_episodes_block = "\n".join(_format_episode_excerpt(ep) for ep in recent_window) if recent_window else "(none)"

    prompt_core = build_open_reflection_prompt(
        runs_digest=runs_digest,
        approaches_digest=approaches_digest,
        strategy_content=strategy_content,
        policy_file_raw=policy_file_raw,
        overall_brier=overall_brier,
        by_category=by_category,
        shrinkage=shrinkage_line,
        recent_episodes_block=recent_episodes_block,
    )
    worst = _select_worst_episodes(resolved, WORST_EPISODE_COUNT)
    worst_block = "\n".join(_format_episode_excerpt(ep) for ep in worst)

    checks_summary = _checks_rollup(episodes)
    tail = (
        f"\n## Check usage\n{checks_summary}\n"
        f"\n## All-time worst episodes\n{worst_block}\n"
    )
    prompt = prompt_core + tail + "\nOutput JSON only.\n"

    if backend == "cursor":
        raw_response = _call_llm_cursor(prompt, model)
    elif backend == "hermes":
        raw_response = _call_llm_hermes(prompt, model, provider)
    elif backend == "openrouter":
        raw_response = _call_llm_openrouter(prompt, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    payload = _extract_json_object(raw_response)
    log = apply_reflection_payload(payload, policy_path=policy_path, vault_dir=vault_dir, dry_run=dry_run)
    parsed = load_policy(policy_path) if not dry_run else load_policy_from_payload(payload)

    new_policy_text = payload.get("policy_markdown", "")
    if not isinstance(new_policy_text, str):
        new_policy_text = ""
    for line in log:
        print(line)
    return parsed, new_policy_text.strip() + "\n", payload


def load_policy_from_payload(payload: dict[str, Any]) -> PolicyConfig:
    policy_md = payload.get("policy_markdown")
    if not isinstance(policy_md, str):
        return PolicyConfig(blind_spot_checks=[])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(policy_md.strip() + "\n")
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        return load_policy(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _summarize_diff(old_text: str, new_text: str) -> None:
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="policy.md (before)",
        tofile="policy.md (after)",
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        print("No textual changes.")
        return
    print("".join(lines[:200]))
    if len(lines) > 200:
        print(f"... ({len(lines) - 200} more diff lines truncated)")


def main(argv: list[str] | None = None) -> int:
    default_vault = Path.home() / "vaults" / "harness-journal"
    parser = argparse.ArgumentParser(description="Reflect on backtest episodes and update forecasting vault + policy.")
    parser.add_argument("--memory-dir", type=Path, default=Path(".harness_memory"))
    parser.add_argument("--policy-path", type=Path, default=Path(".harness/policy.md"))
    parser.add_argument("--vault-dir", type=Path, default=default_vault, help="Obsidian vault root")
    parser.add_argument(
        "--no-vault",
        action="store_true",
        help="Skip vault digest/output; only read/write machine policy",
    )
    parser.add_argument("--runs-subdir", type=str, default="runs")
    parser.add_argument("--approaches-subdir", type=str, default="approaches")
    parser.add_argument(
        "--backend",
        choices=("cursor", "hermes", "openrouter"),
        default=None,
        help="LLM backend (default: openrouter if OPENROUTER_API_KEY is set, else hermes)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Chat model (default: HERMES_REFLECT_MODEL env, else backend default)",
    )
    parser.add_argument("--provider", default="opencode-go", help="Provider for hermes backend (default: opencode-go)")
    parser.add_argument("--dry-run", action="store_true", help="Parse + print plan; do not write files")
    args = parser.parse_args(argv)

    if args.backend is None:
        if os.environ.get("OPENROUTER_API_KEY", "").strip():
            args.backend = "openrouter"
        elif shutil.which("agent"):
            args.backend = "cursor"
        else:
            args.backend = "hermes"

    if args.model is None:
        env_model = os.environ.get("HERMES_REFLECT_MODEL", "").strip()
        if env_model:
            args.model = env_model
        elif args.backend == "cursor":
            args.model = "composer-2-fast"
        elif args.backend == "hermes":
            args.model = "deepseek-v4-pro"
        else:
            args.model = "deepseek/deepseek-chat"

    policy_path = args.policy_path.expanduser().resolve()
    old_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    vault_dir: Path | None = None if args.no_vault else args.vault_dir.expanduser().resolve()

    try:
        _policy, new_raw, payload = run_reflection(
            memory_dir=args.memory_dir,
            policy_path=policy_path,
            vault_dir=vault_dir,
            runs_subdir=args.runs_subdir,
            approaches_subdir=args.approaches_subdir,
            backend=args.backend,
            model=args.model,
            provider=args.provider,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(exc)
        return 2
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"reflection failed: {exc}")
        return 2

    if args.dry_run:
        print(new_raw)
        print(json.dumps(payload, indent=2))
    _summarize_diff(old_text, new_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
