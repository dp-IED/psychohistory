# Psychohistory

A harness-agnostic plugin training bed. Live dated claims train the plugin; the overlay is where forecasting skill is supposed to accumulate so horizons can lengthen; later epochs may add the plugin’s own simulation tools. This repo is not an in-process orchestrator.

## Language

**Psychohistory**:
The target capability named by this repo: longer-horizon forecasting from accumulated overlay patterns, eventually with plugin-grown simulation tools. Not Seldon mathematics, and not a restored GNN corpus as the product on this branch.
_Avoid_: Treating one dated claim as psychohistory; France/warehouse GNN as this branch’s loop

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
`skills/`, `agents/`, `references/`, and `scripts/` at the plugin root (this repository). Load in place: `claude --plugin-dir .` This is the plugin’s memory: transferable methods and durable facts. **Reflection** writes it.
_Avoid_: Marketplace/cache install as the live copy; ledger as memory; graph-vault; harness session notes as the plugin’s memory

**Polymarket testbed**:
Historical gold cases, probes, and schemas. In git history, not this branch. Live markets are a discovery and scoring surface, not that object.
_Avoid_: Treating live markets as the parked testbed

**Ledger**:
A single markdown file at repo root (`ledger.md`). **Problems** (id, title, **motivation**, **resolution day**) and **dated claims** (**forecast day**, owner, claim, **justification**). `K` is max new problems per **discover** tick.
_Avoid_: hypotheses.json; a wakeups/ directory; two clocks named Due and Y

**Wakeup**:
One dated claim in the ledger. Not its own file.
_Avoid_: One markdown file per hypothesis

**Justification**:
Explanation and reasoning trace for a **claim**. Graded after **resolution day**, as part of the whole series.
_Avoid_: Backup, chain, second clock, problem motivation

**Problem**:
A prediction setup that may emit many dated claims. It owns the **resolution day**. Opened without a quality gate; reflection culls losers later.
_Avoid_: Requiring a transfer speech before opening; a second Due clock on the claim

**Resolution day**:
The date the world answers the **problem**. **Discover** sets it. **Predict** may run while today is on or before it. **Reflection** runs after it. Old names **Due** and **Y** meant this one date, not two.
_Avoid_: Due as wakeup day and Y as a second score day

**Forecast day**:
The calendar date **predict** wrote this **dated claim**. How early the outcome showed up. Not **resolution day**.
_Avoid_: Using Due for this; relying on git history as the domain clock

**Live problem**:
A **problem** whose **resolution day** is today or later. **Predict** may wake it. After **resolution day** has passed, it is not live.
_Avoid_: Waking every problem forever; treating the parse seed as live

**Revision**:
A new dated claim on a live problem, appended when the predicted outcome changes. The new row has **forecast day**, **claim**, and **justification**. Earlier rows stay.
_Avoid_: Overwriting an old row; a justification-only update with the same outcome

**Horizon**:
How far **resolution day** sits from today when the **problem** is opened. Lengthens when **discover** prefers farther **resolution day**. More **revisions** of a near event are not a longer horizon. **Resolution day** moves later only if the public date moved.
_Avoid_: Counting daily revisions as horizon growth; using Y as a second field

**Discover**:
Host tick that opens at most **K** new **problems** (each with **Motivation** and **resolution day**). When opening more than one problem, spread the batch across domains (politics, economics/markets, courts/legal, science/tech, conflict/security, sports, culture) rather than clustering one vein, weighed against what is already open (ADR 0011). It does not write **claim** or **justification**.
_Avoid_: Full forecast on discover; leaving a problem with no resolution day; letting a large `K` cluster in one domain

**Predict**:
Host tick that writes **claim** and **justification** on **live problems** (first row or **revision** when the outcome changes). Formerly called due-today. It does not invent **resolution day**.
_Avoid_: Due-today as the current name; waking only rows whose Due equals today

**Memory**:
The plugin overlay. Not the ledger and not a sidecar store.
_Avoid_: Using memory for justification traces, Cursor automation memories, or graph-vault

**Motivation**:
Why this problem was opened. Stored on the problem in the ledger so Parent does not forget.
_Avoid_: Justification (that belongs on a dated claim)

**Parent**:
**Predict** tick: write today’s forecasts and revisions on **live problems**. **Discover** tick: open **problems** and set **resolution day**. **Reflection** tick: after **resolution day**, grade the whole series and grow or cull the plugin. At most **K** new problems per discover tick; open inventory unbounded.
_Avoid_: Repo crontab; combining predict, discover, and reflect in one host job

**Fan-out**:
Breadth of live claims plus ungated proactive **problem** finding. Reflection culls. Calibration is the whole series after **resolution day**, plus whether methods transferred.
_Avoid_: Multi-agent roster on one question; replaying old gold

**Reflection**:
After **resolution day**, grade every **dated claim** in the series (claim and justification). The earliest matching **claim** is the prize. Then change the **plugin**. Add files when the grade earned a new capability. Cull overlay that failed. Durable facts live in the overlay.
_Avoid_: Sidecar `graph-vault/`; silent weight updates; grading only the last row

**Training loop**:
Forward only: **discover** (problem + **resolution day**) → **predict** (claim + justification, **revisions** while live) → after **resolution day**, **reflection** → **plugin**. Three separate host jobs.
_Avoid_: PIT backtest as the epoch; gold Brier as quality; due-today as the current tick name; Due and Y as two clocks
