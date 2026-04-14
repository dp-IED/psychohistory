"""Karpathy-style iteration driver: propose → sandbox → eval → log → keep best."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import typer
from pydantic import BaseModel, Field, ValidationError

from autoresearch.agent_interface import load_proposal, noop_proposal
from autoresearch.candidate_manager import (
    apply_proposal,
    load_schema_from_candidate,
    new_candidate_id,
    prepare_candidate_tree,
    schema_spec_for_candidate,
)
from autoresearch.config import HarnessPaths
from autoresearch.logging import ExperimentRecord, append_experiment, read_prior_scores
from autoresearch.scoring import compose_from_eval_report
from evals.reporting import report_to_jsonable
from evals.run_eval import evaluate_schema
from schemas.schema_registry import load_graph_schema


class AdversarialResult(BaseModel):
    adversarial_score: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    regression_flags: list[str] = Field(default_factory=list)


@dataclass
class CandidateChangeInfo:
    changed: bool
    before_hash: str
    after_hash: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _append_agent_subprocess_log(
    log_path: Path,
    *,
    phase: str,
    iteration: int,
    candidate_id: str,
    command: str,
    proc: subprocess.CompletedProcess[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    out = proc.stdout or ""
    err = proc.stderr or ""
    block = (
        f"\n{'=' * 72}\n"
        f"ts={ts} phase={phase} iter={iteration} candidate={candidate_id} "
        f"exit={proc.returncode}\n"
        f"command: {command}\n"
        f"--- stdout ---\n{out if out.strip() else '(empty)'}\n"
        f"--- stderr ---\n{err if err.strip() else '(empty)'}\n"
    )
    log_path.open("a", encoding="utf-8").write(block)


def _echo_agent_streams(proc: subprocess.CompletedProcess[str]) -> None:
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        typer.echo(out)
    if err:
        typer.secho(err, fg=typer.colors.YELLOW)


def _best_score(log_path: Path) -> float:
    s = read_prior_scores(log_path)
    return max(s) if s else float("-inf")


def _resolve_schema_source_root(paths: HarnessPaths, seed_from_best: bool) -> Path:
    if seed_from_best:
        best_latest = paths.best_dir / "latest"
        if (best_latest / "schemas").is_dir():
            return best_latest
    return paths.root


def _run_proposal_command(
    command_template: str,
    proposal_file: Path,
    iteration: int,
    prior_best: float,
    candidate_id: str,
    root: Path,
    show_output: bool,
    agent_log_path: Path | None = None,
) -> None:
    values = {
        "proposal_file": str(proposal_file),
        "iteration": str(iteration),
        "prior_best": str(prior_best),
        "candidate_id": candidate_id,
        "root": str(root),
    }
    command = command_template
    for key, value in values.items():
        command = command.replace(f"{{{key}}}", value)
    command = _ensure_agent_command_permissions(command)

    proc = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if agent_log_path is not None:
        _append_agent_subprocess_log(
            agent_log_path,
            phase="proposal",
            iteration=iteration,
            candidate_id=candidate_id,
            command=command,
            proc=proc,
        )
    if show_output:
        _echo_agent_streams(proc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"proposal command failed (exit {proc.returncode}): {msg}")
    if not proposal_file.exists():
        raise FileNotFoundError(f"proposal command did not create {proposal_file}")


def _ensure_agent_command_permissions(command: str) -> str:
    stripped = command.lstrip()
    if not stripped.startswith("agent "):
        return command
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "agent":
        return command

    existing = set(tokens)
    extras: list[str] = []
    if "--trust" not in existing:
        extras.append("--trust")
    if "--force" not in existing and "-f" not in existing and "--yolo" not in existing:
        extras.append("--force")
    if "-p" not in existing and "--print" not in existing:
        extras.append("--print")

    if not extras:
        return command
    normalized = [tokens[0], *extras, *tokens[1:]]
    return shlex.join(normalized)


def _score_delta(prior_best: float, composite: float) -> float:
    if prior_best == float("-inf"):
        return composite
    return composite - prior_best


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_json_from_stdout(stdout: str) -> dict:
    """Parse a JSON object from CLI stdout.

    Accepts: whole-buffer JSON, one JSON line, ```json ... ``` fences, or the first
    top-level `{...}` object embedded in prose (typical for agent CLIs).
    """
    text = stdout.strip()
    if not text:
        raise ValueError("adversarial command returned empty stdout")
    decoder = json.JSONDecoder()

    def dict_from_string(s: str) -> dict | None:
        s = s.strip()
        if not s:
            return None
        try:
            payload = json.loads(s)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        for i, ch in enumerate(s):
            if ch != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(s, i)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    chunks: list[str] = [text]
    for line in reversed(text.splitlines()):
        t = line.strip()
        if t:
            chunks.append(t)
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        b = m.group(1).strip()
        if b:
            chunks.append(b)

    seen: set[str] = set()
    for chunk in chunks:
        if chunk in seen:
            continue
        seen.add(chunk)
        got = dict_from_string(chunk)
        if got is not None:
            return got
    raise ValueError("adversarial command stdout did not contain JSON object")


def _run_adversarial_command(
    command_template: str,
    iteration: int,
    prior_best: float,
    candidate_id: str,
    root: Path,
    candidate_root: Path,
    schema_spec: str,
    probe_dir: Path,
    show_output: bool,
    agent_log_path: Path | None = None,
) -> dict:
    values = {
        "iteration": str(iteration),
        "prior_best": str(prior_best),
        "candidate_id": candidate_id,
        "root": str(root),
        "candidate_root": str(candidate_root),
        "schema_spec": schema_spec,
        "probe_dir": str(probe_dir),
    }
    command = command_template
    for key, value in values.items():
        command = command.replace(f"{{{key}}}", value)
    command = _ensure_agent_command_permissions(command)

    proc = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if agent_log_path is not None:
        _append_agent_subprocess_log(
            agent_log_path,
            phase="adversarial",
            iteration=iteration,
            candidate_id=candidate_id,
            command=command,
            proc=proc,
        )
    if show_output:
        _echo_agent_streams(proc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"adversarial command failed (exit {proc.returncode}): {msg}")
    raw = _parse_json_from_stdout(proc.stdout)
    try:
        parsed = AdversarialResult.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid adversarial output JSON: {exc}") from exc
    return parsed.model_dump()


def _apply_adversarial_result(report_obj: dict, adversarial_result: dict) -> tuple[dict, float]:
    stub_scores = report_obj.setdefault("stub_scores", {})
    if not isinstance(stub_scores, dict):
        stub_scores = {}
        report_obj["stub_scores"] = stub_scores
    raw = adversarial_result.get("adversarial_score", adversarial_result.get("score", 0.0))
    adv_score = _clip01(float(raw))
    stub_scores["adversarial_agent"] = adv_score

    meta = report_obj.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        report_obj["meta"] = meta
    details = {k: v for k, v in adversarial_result.items() if k not in {"adversarial_score", "score"}}
    meta["adversarial_eval"] = {"score": adv_score, **details}

    composite, parts = compose_from_eval_report(report_obj)
    report_obj["composite_score"] = composite
    meta["score_parts"] = parts
    return report_obj, composite


def _hash_dir(path: Path) -> str:
    h = sha256()
    if not path.exists():
        return ""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        h.update(str(file.relative_to(path)).encode("utf-8"))
        h.update(file.read_bytes())
    return h.hexdigest()


def _candidate_change_info(candidate_root: Path) -> CandidateChangeInfo:
    before = _hash_dir(candidate_root / "schemas_before")
    after = _hash_dir(candidate_root / "schemas")
    return CandidateChangeInfo(changed=before != after, before_hash=before, after_hash=after)


def _apply_noop_penalty(report_obj: dict, composite: float, changed: bool, penalty: float) -> tuple[dict, float]:
    if changed:
        return report_obj, composite
    adjusted = _clip01(composite - penalty)
    report_obj["composite_score"] = adjusted
    meta = report_obj.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        report_obj["meta"] = meta
    meta["noop_penalty"] = {"applied": True, "penalty": penalty}
    return report_obj, adjusted


def _format_commit_message(
    candidate_id: str, prior_best: float, composite: float
) -> str:
    delta = _score_delta(prior_best, composite)
    prev = "none" if prior_best == float("-inf") else f"{prior_best:.4f}"
    return (
        f"autoresearch: candidate {candidate_id} benchmark {composite:.4f} "
        f"(delta {delta:+.4f} vs {prev})"
    )


def _git_commit_schema_update(repo_root: Path, message: str) -> tuple[str | None, str]:
    check = subprocess.run(
        "git status --short -- schemas",
        cwd=repo_root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        stderr = (check.stderr or check.stdout or "").strip()
        return None, f"git status failed: {stderr}"
    if not check.stdout.strip():
        return None, "no schema diff to commit"

    add = subprocess.run(
        "git add -- schemas",
        cwd=repo_root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if add.returncode != 0:
        stderr = (add.stderr or add.stdout or "").strip()
        return None, f"git add failed: {stderr}"

    commit = subprocess.run(
        f"git commit -m {shlex.quote(message)}",
        cwd=repo_root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        stderr = (commit.stderr or commit.stdout or "").strip()
        return None, f"git commit failed: {stderr}"

    rev = subprocess.run(
        "git rev-parse --short HEAD",
        cwd=repo_root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if rev.returncode != 0:
        stderr = (rev.stderr or rev.stdout or "").strip()
        return None, f"commit succeeded but rev-parse failed: {stderr}"
    return rev.stdout.strip(), "ok"


def main(
    iterations: int = typer.Option(1, "--iterations", "-n", min=1),
    probe_dir: Path = typer.Option(
        Path("probes"), "--probe-dir", exists=True, file_okay=False
    ),
    seed_schema: str = typer.Option("schemas.base_schema", "--seed-schema"),
    proposal_file: Path | None = typer.Option(None, "--proposal-file", dir_okay=False),
    proposal_cmd: str | None = typer.Option(
        None,
        "--proposal-cmd",
        help=(
            "Shell command run before each iteration to generate proposal JSON. "
            "Supports {proposal_file}, {iteration}, {prior_best}, {candidate_id}, {root}."
        ),
    ),
    adversarial_cmd: str | None = typer.Option(
        None,
        "--adversarial-cmd",
        help=(
            "Optional command to adversarially score each candidate. "
            "Must emit JSON with adversarial_score in [0,1]. "
            "Supports {iteration}, {prior_best}, {candidate_id}, {root}, {candidate_root}, {schema_spec}, {probe_dir}."
        ),
    ),
    root: Path | None = typer.Option(
        None, "--root", help="Repository root (default: cwd)"
    ),
    print_schema: bool = typer.Option(
        False, "--print-schema", help="Print seed schema stats and exit"
    ),
    seed_from_best: bool = typer.Option(
        True,
        "--seed-from-best/--seed-from-repo",
        help="Start each candidate from experiments/best/latest when available.",
    ),
    allow_noop: bool = typer.Option(
        False,
        "--allow-noop",
        help="Allow no-op baseline run when --proposal-file is omitted.",
    ),
    auto_commit: bool = typer.Option(
        False,
        "--auto-commit",
        help="Commit schema updates when a candidate improves the benchmark.",
    ),
    concise_output: bool = typer.Option(
        True,
        "--concise-output/--verbose-output",
        help="Show concise per-iteration summary (default) or verbose logs.",
    ),
    show_agent_output: bool = typer.Option(
        False,
        "--show-agent-output",
        help="Print raw stdout/stderr from proposal and adversarial agent commands.",
    ),
    agent_log_file: Path | None = typer.Option(
        None,
        "--agent-log-file",
        help=(
            "Append full stdout/stderr from each agent subprocess to this file. "
            "Default: autoresearch/experiments/agent_subprocess.log when "
            "--proposal-cmd or --adversarial-cmd is set (use --no-agent-log to disable)."
        ),
    ),
    no_agent_log: bool = typer.Option(
        False,
        "--no-agent-log",
        help="Do not write agent subprocess transcripts to disk.",
    ),
    noop_penalty: float = typer.Option(
        0.01,
        "--noop-penalty",
        min=0.0,
        max=0.25,
        help="Penalty subtracted from score when candidate schema is unchanged.",
    ),
    graph_artifact_dir: Path = typer.Option(
        Path("autoresearch/experiments/graph_cache"),
        "--graph-artifact-dir",
        file_okay=False,
        help="Directory containing precomputed per-probe graph artifacts.",
    ),
    objective_strict_mode: bool = typer.Option(
        True,
        "--objective-strict-mode/--objective-lenient-mode",
        help="Fail candidate eval when objective fields/artifacts are missing or stale.",
    ),
    graph_precompute_cmd: str | None = typer.Option(
        "__builtin__",
        "--graph-precompute-cmd",
        help=(
            "Command to refresh stale graph artifacts before eval. "
            "Use '__builtin__' for deterministic synthesis, or a shell command with "
            "{probe_file}, {probe_id}, {artifact_file}, {schema_json}."
        ),
    ),
    auto_refresh_graph_artifacts: bool = typer.Option(
        True,
        "--auto-refresh-graph-artifacts/--no-auto-refresh-graph-artifacts",
        help="Rebuild graph artifacts when fingerprints disagree with the current schema/probe YAML.",
    ),
    graph_precompute_show_output: bool = typer.Option(
        False,
        "--graph-precompute-show-output",
        help="Print stdout from graph precompute commands.",
    ),
    graph_builder_version: str = typer.Option(
        "v1",
        "--graph-builder-version",
        help="Expected graph builder version in artifact manifests.",
    ),
    traversal_promotion_threshold: float | None = typer.Option(
        None,
        "--traversal-promotion-threshold",
        help="If set, block external stub contribution unless traversal score meets threshold.",
    ),
) -> None:
    """
    Run autoresearch iterations.

    Cursor Agent CLI workflow:
      1. Read autoresearch/prompts/*.md and prior experiments/log.jsonl
      2. Emit a JSON proposal (see autoresearch/agent_interface.py)
      3. Pass it via --proposal-file on the next runner invocation
    """

    if print_schema:
        s = load_graph_schema(seed_schema)
        typer.echo(
            f"{s.name} v{s.version} nodes={len(s.node_types)} edges={len(s.edge_types)}"
        )
        raise typer.Exit(code=0)

    paths = HarnessPaths.default(root)
    paths.experiments_dir.mkdir(parents=True, exist_ok=True)
    paths.best_dir.mkdir(parents=True, exist_ok=True)
    paths.candidates_dir.mkdir(parents=True, exist_ok=True)

    agent_log_path: Path | None = None
    if not no_agent_log:
        if agent_log_file is not None:
            agent_log_path = agent_log_file
        elif proposal_cmd or adversarial_cmd:
            agent_log_path = paths.experiments_dir / "agent_subprocess.log"

    opt_prompt = _read_text(paths.optimization_prompt)
    agent_prompt = _read_text(paths.agent_system_prompt)
    _ = (opt_prompt, agent_prompt)  # reserved for future agent subprocess

    if proposal_cmd and proposal_file is None:
        proposal_file = paths.root / "proposal.json"
    if proposal_file is not None and not proposal_file.suffix:
        raise typer.BadParameter(
            "--proposal-file should include a filename (e.g. proposal.json)"
        )

    if proposal_file is None and not allow_noop:
        typer.secho(
            "Missing --proposal-file. Provide a proposal JSON, use --proposal-cmd, or pass --allow-noop.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    if proposal_file is None:
        if not concise_output:
            typer.secho(
                "No --proposal-file: using a no-op proposal (schema copy only).",
                fg=typer.colors.YELLOW,
            )
    else:
        if not concise_output:
            typer.echo(f"Applying proposal: {proposal_file.resolve()}")

    for i in range(iterations):
        prior_best = _best_score(paths.log_jsonl)
        cid = new_candidate_id()
        if proposal_cmd and proposal_file is not None:
            _run_proposal_command(
                command_template=proposal_cmd,
                proposal_file=proposal_file,
                iteration=i,
                prior_best=prior_best,
                candidate_id=cid,
                root=paths.root,
                show_output=show_agent_output,
                agent_log_path=agent_log_path,
            )

        schema_source_root = _resolve_schema_source_root(
            paths, seed_from_best=seed_from_best
        )
        cand_root = prepare_candidate_tree(
            paths.root, cid, seed_schema, schema_source_root=schema_source_root
        )
        shutil.copytree(cand_root / "schemas", cand_root / "schemas_before", dirs_exist_ok=True)

        if proposal_file:
            if not proposal_file.exists():
                raise FileNotFoundError(f"Proposal file not found: {proposal_file}")
            proposal = load_proposal(proposal_file)
        else:
            proposal = noop_proposal()

        failure_class: str | None = None
        eval_report_path: Path | None = None
        composite = 0.0
        report_obj: dict | None = None

        try:
            apply_proposal(cand_root, proposal, paths.root)
            schema, _used = load_schema_from_candidate(cand_root, seed_schema)
            report = evaluate_schema(
                schema,
                probe_dir,
                weights=None,
                graph_artifact_dir=graph_artifact_dir,
                objective_strict_mode=objective_strict_mode,
                graph_builder_version=graph_builder_version,
                traversal_promotion_threshold=traversal_promotion_threshold,
                graph_precompute_cmd=graph_precompute_cmd,
                auto_refresh_graph_artifacts=auto_refresh_graph_artifacts,
                graph_precompute_show_output=graph_precompute_show_output,
                graph_precompute_repo_root=paths.root,
            )
            report_obj = report_to_jsonable(report)
            composite = float(report_obj.get("composite_score", 0.0))
            if adversarial_cmd is not None:
                adv_result = _run_adversarial_command(
                    command_template=adversarial_cmd,
                    iteration=i,
                    prior_best=prior_best,
                    candidate_id=cid,
                    root=paths.root,
                    candidate_root=cand_root,
                    schema_spec=_used,
                    probe_dir=probe_dir,
                    show_output=show_agent_output,
                    agent_log_path=agent_log_path,
                )
                report_obj, composite = _apply_adversarial_result(report_obj, adv_result)
            change_info = _candidate_change_info(cand_root)
            report_obj, composite = _apply_noop_penalty(
                report_obj,
                composite,
                changed=change_info.changed,
                penalty=noop_penalty,
            )
            report_obj.setdefault("meta", {})["candidate_changed"] = change_info.changed
            eval_report_path = cand_root / "eval_report.json"
            eval_report_path.write_text(
                json.dumps(report_obj, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            failure_class = report_obj.get("failure_class")
        except SyntaxError:
            failure_class = "syntax_failure"
            traceback.print_exc()
        except Exception:
            failure_class = "eval_crash"
            traceback.print_exc()

        if failure_class is None and report_obj is not None:
            if composite + 1e-12 < prior_best:
                failure_class = "score_regression"

        record = ExperimentRecord(
            iteration=i,
            candidate_id=cid,
            timestamp_utc="",
            composite_score=composite,
            score_parts=(report_obj or {}).get("meta", {}).get("score_parts", {}),
            proposal=proposal.to_json_obj(),
            eval_report_path=str(eval_report_path) if eval_report_path else None,
            failure_class=failure_class,
            artifacts={
                "candidate_root": str(cand_root),
                "schema_spec_used": schema_spec_for_candidate(cand_root, seed_schema),
            },
        )
        append_experiment(paths.log_jsonl, record)

        improved = failure_class is None and composite > prior_best + 1e-12
        accepted = failure_class is None and composite + 1e-12 >= prior_best
        commit_sha: str | None = None
        commit_message: str | None = None
        commit_status: str | None = None

        if accepted:
            shutil.copytree(cand_root, paths.best_dir / "latest", dirs_exist_ok=True)
            (paths.best_dir / "README_HINT.txt").write_text(
                "latest/ mirrors the best-scoring candidate tree so far.\n",
                encoding="utf-8",
            )
            if auto_commit and improved:
                shutil.copytree(
                    cand_root / "schemas", paths.root / "schemas", dirs_exist_ok=True
                )
                commit_message = _format_commit_message(cid, prior_best, composite)
                commit_sha, commit_status = _git_commit_schema_update(
                    paths.root, commit_message
                )

        if concise_output:
            delta = _score_delta(prior_best, composite)
            if failure_class is None:
                base = (
                    f"iter={i} candidate={cid} score={composite:.4f} delta={delta:+.4f} "
                    f"status={'improved' if improved else 'accepted_tie'}"
                )
                if auto_commit and improved:
                    if commit_sha and commit_message:
                        typer.echo(f"{base} commit={commit_sha} msg={commit_message}")
                    else:
                        typer.echo(
                            f"{base} commit=skipped reason={commit_status or 'unknown'}"
                        )
                else:
                    typer.echo(base)
            else:
                typer.echo(
                    f"iter={i} candidate={cid} score={composite:.4f} delta={delta:+.4f} status=failed:{failure_class}"
                )

    if concise_output:
        typer.echo(f"done log={paths.log_jsonl}")
    else:
        typer.echo(f"Done. Log: {paths.log_jsonl}")


if __name__ == "__main__":
    typer.run(main)
