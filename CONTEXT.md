# Psychohistory

A harness-agnostic plugin for live forecasting: agents write justified claims in a ledger, a host loop wakes a Parent, the Parent wakes today's agents, reflection updates the plugin and the vault graph. This repo is a training bed, not an in-process orchestrator.

## Language

**Plugin**:
The unit this repo ships: a Claude-style package with `skills/`, `agents/`, `references/`, and `scripts/`. Consuming harnesses (Cursor, Claude Code, Codex, and others) load it; they own orchestration.
_Avoid_: In-repo runner, hermes replacement, harness.orchestrator as the product

**Skill**:
A markdown instruction unit under `skills/` that an agent reads and follows. Not the Python helpers currently at `harness/skills/`.
_Avoid_: Deterministic compression module, Cursor-only skill

**Agent**:
A markdown agent definition under `agents/`: a named role with its own prompt and tool surface. Fan-out and rostering happen in the consuming harness, not in this plugin.
_Avoid_: Subagent-as-Python-type, delegate_task roster inside orchestrator.py

**Reference**:
Long-form material under `references/` that skills and agents consult (procedures, vault conventions). Not runtime forecast output.
_Avoid_: Spec dump, CONTEXT.md as a spec

**Script**:
Deterministic code under `scripts/`. No LLM.
_Avoid_: Hermes CLI wrapper, LLM synthesis

**Harness**:
The host that runs agents: Cursor, Claude Code, Codex, or any other. It schedules skills and agents and supplies the LLM. This plugin does not inject a model backend.
_Avoid_: hermes, orchestrator.py

**PIT**:
Legacy vault filter in `harness/vault_pit.py`. It is not the training loop. The loop does not replay historical cutoffs.
_Avoid_: Using PIT as the name for live wakeups or host cron

**Training epoch**:
One scored pass whose intended outcome is plugin improvement: new or tighter skills, new tools, clearer references — not a weight update. Guard against skill bloat (prefer writing-great-skills / `/writing-for-agents` over adding files).
_Avoid_: Fine-tuning run, gradient step

**Plugin overlay**:
`skills/`, `agents/`, and `references/` sit on top of the existing vault and `harness/` Python. Deterministic helpers stay in `harness/` until a later move into plugin `scripts/`.
_Avoid_: Relocating the vault into the plugin this session

**Polymarket testbed**:
Historical gold cases, probes, and `schemas/polymarket_agentic.py`. Parked. Live Polymarket is a discovery and scoring surface, not this object.
_Avoid_: Branch portfolio as v1 plugin output; treating live markets as the parked testbed

**Ledger**:
A single markdown file at repo root (`ledger.md`). Two sections: **problems** (with **motivation**) and **dated claims** (due, owner, claim, **justification**). `K` (max new problems per discovery tick) is a knob on that file; start at 1.
_Avoid_: hypotheses.json as source of truth; a wakeups/ directory of many files; burying the schedule inside graph-vault or the plugin

**Wakeup**:
One dated claim in the ledger. Not its own file.
_Avoid_: One markdown file per hypothesis; backup; child market

**Justification**:
Explanation and reasoning trace for a **claim**. Graded after `Y`.
_Avoid_: Backup, chain, second clock, problem motivation

**Problem**:
A prediction setup that may emit many dated claims. Opened without a quality gate; reflection culls losers later.
_Avoid_: Requiring a transfer speech before opening; treating a problem as one due date

**Motivation**:
Why this problem was opened. Stored on the problem in the ledger so Parent does not forget.
_Avoid_: Justification (that belongs on a dated claim)

**Parent**:
Due-today tick: read the ledger, wake agents scheduled for today. Discovery is a different, rarer tick. At most **K** new problems per discovery tick; open inventory unbounded.
_Avoid_: Repo crontab; discovery on the same tick as due-today; a cap on total open claims

**Fan-out**:
Breadth of live claims plus ungated proactive **problem** finding. Reflection culls. Calibration is scoring at `Y` plus whether methods transferred.
_Avoid_: Multi-agent roster on one question; replaying old gold

**Reflection**:
After `Y`, grade the claim and the justification, then write **plugin** (skills/agents/references, with bloat guardrails) **and vault** (entities, threads, concepts — the graph). GNN scoring of that graph is later, not this loop.
_Avoid_: Plugin-only patches; silent weight updates

**Training loop**:
Forward only: discover problems and claims → append ledger → host due-today tick → Parent wakes due agents → at `Y` reflect → plugin + vault. Discovery tick is separate.
_Avoid_: PIT backtest as the epoch; gold Brier as quality
