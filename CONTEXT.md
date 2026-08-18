# Psychohistory

A harness-agnostic plugin for live forecasting: agents write justified claims in a ledger, a host loop wakes a Parent, the Parent wakes today's agents, reflection grows the plugin. This repo is a training bed, not an in-process orchestrator.

## Language

**Plugin**:
The unit this repo ships: a Claude-style package with `skills/`, `agents/`, `references/`, and `scripts/`. Consuming harnesses (Cursor, Claude Code, Codex, and others) load it; they own orchestration.
_Avoid_: In-repo runner, hermes replacement, harness.orchestrator as the product

**Skill**:
A markdown instruction unit under `skills/` that an agent reads and follows. Not the retired Python helpers that lived at `harness/skills/`.
_Avoid_: Deterministic compression module, Cursor-only skill

**Agent**:
A markdown agent definition under `agents/`: a named role with its own prompt and tool surface. Fan-out and rostering happen in the consuming harness, not in this plugin.
_Avoid_: Subagent-as-Python-type, delegate_task roster inside orchestrator.py

**Reference**:
Long-form material under `references/` that skills and agents consult (procedures, strategies, durable facts). Not a sidecar vault and not runtime forecast output.
_Avoid_: Spec dump; restoring `graph-vault/` layout; CONTEXT.md as a spec

**Script**:
Deterministic code. The live schedule reader is `harness.ledger`. No LLM.
_Avoid_: Hermes CLI wrapper, LLM synthesis

**Harness**:
The host that runs agents: Cursor, Claude Code, Codex, or any other. It schedules skills and agents and supplies the LLM. This plugin does not inject a model backend.
_Avoid_: hermes, orchestrator.py

**PIT**:
Retired vault filter (git history). Not the training loop.
_Avoid_: Using PIT as the name for live wakeups or host cron

**Training epoch**:
One scored pass whose intended outcome is plugin growth: new or rewritten skills, agents, tools/scripts, strategies, and instructions — not a weight update. Later reflection culls overlay that failed to transfer.
_Avoid_: Fine-tuning run, gradient step

**Plugin overlay**:
`skills/`, `agents/`, and `references/` at the plugin root (this repository). Load in place: `claude --plugin-dir .`
_Avoid_: Marketplace/cache install as the live copy

**Polymarket testbed**:
Historical gold cases, probes, and schemas. In git history, not this branch. Live markets are a discovery and scoring surface, not that object.
_Avoid_: Treating live markets as the parked testbed

**Ledger**:
A single markdown file at repo root (`ledger.md`). Two sections: **problems** (with **motivation**) and **dated claims** (due, optional **Y**, owner, claim, **justification**). `K` (max new problems per discovery tick) is a knob on that file; start at 1.
_Avoid_: hypotheses.json as source of truth; a wakeups/ directory of many files

**Wakeup**:
One dated claim in the ledger. Not its own file.
_Avoid_: One markdown file per hypothesis

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
Due-today tick (`/due-today`): read the ledger, wake agents scheduled for today. Discovery is a different, rarer tick (`/discover`). At most **K** new problems per discovery tick; open inventory unbounded. Reflection tick (`/reflect`): after `Y`, grade and grow or cull the plugin.
_Avoid_: Repo crontab; combining due-today, discovery, and reflect in one host job

**Fan-out**:
Breadth of live claims plus ungated proactive **problem** finding. Reflection culls. Calibration is scoring at `Y` plus whether methods transferred.
_Avoid_: Multi-agent roster on one question; replaying old gold

**Reflection**:
After `Y`, grade the claim and the justification, then change the **plugin** (new or rewritten skills, agents, references, scripts/tools, strategies). Add files when the grade earned a new capability. Cull overlay that failed. Durable facts live in the overlay.
_Avoid_: Sidecar `graph-vault/`; silent weight updates

**Training loop**:
Forward only: discover problems and claims → append ledger → host due-today tick → Parent wakes due agents → at `Y` reflect → plugin. Discovery tick is separate.
_Avoid_: PIT backtest as the epoch; gold Brier as quality
