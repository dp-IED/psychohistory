# Psychohistory

A harness-agnostic plugin training bed. Live dated claims train the plugin; the overlay is where forecasting skill is supposed to accumulate so horizons can lengthen; later epochs may add the plugin’s own simulation tools. This repo is not an in-process orchestrator.

## Language

**Psychohistory**:
The target capability named by this repo: longer-horizon forecasting from accumulated overlay patterns, eventually with plugin-grown simulation tools. Intended later use: **tenants** (LFI, DSA, or another asking party) see **openings**, not run their campaigns. Not Seldon mathematics, and not a restored GNN corpus as the product on this branch.
_Avoid_: Treating one dated claim as psychohistory; France/warehouse GNN as this branch’s loop; treating a playbook as psychohistory

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
The host that runs agents: Cursor, Claude Code, Codex, or any other. It schedules skills and agents, supplies the LLM, and holds **conversation history**, artifacts, and local files. **Openings** live there, not in this plugin repo. This plugin does not inject a model backend.
_Avoid_: hermes, orchestrator.py; committing tenant openings into `ledger.md` or overlay

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
Explanation and reasoning trace for a **claim**, including a **Structure** block (class, mechanism, base rate, disanalogy, falsifiers). Graded after **resolution day**, as part of the whole series.
_Avoid_: Backup, chain, second clock, problem motivation; reciting parametric history as if it were a card; treating an **opening** as the claim

**Opening**:
A note about where **the asking tenant** can push, where they will be blocked, or where the other side is weak. Default: **one-off in conversation history**. User may ask to save it on the harness filesystem after. Not in this plugin repo. Not a **dated claim**. Not graded on **resolution day**. Not a **playbook**.
_Avoid_: Mixing openings into the claim sentence; scoring “you should organise X”; copying a tenant opening into `ledger.md` or overlay

**Chat forecast**:
A forecast written in conversation. The wording the user saw stays in chat (and on harness FS only if they ask). An **anonymized** copy must be written to the **ledger** as a dated **claim** so the training loop can score it.
_Avoid_: Leaving a chat-made forecast only in conversation; copying tenant openings or party names into the ledger copy

**Opening type**:
The **typical openings** bit on an **analog card** (same pile, not a second filing system). Anonymized: no tenant name. Instantiations still name countries and cases. Written when **reflection** earns it.
_Avoid_: “DSA should…”; a separate opening-types library; treating typical openings as a dated claim

**Tenant**:
Who is asking this run: a party or similar organisation (LFI, DSA, or another). The system is **multitenant**. Shared: forecast methods, **analog cards**, **opening types**. Private to the run: **openings**.
_Avoid_: Hardcoding one party as the only user; treating “the left” as the tenant; sharing tenant openings as if they were opening types

**Playbook**:
A list of actions a party should take next. Not a scored plugin output. Humans decide that.
_Avoid_: Grading strategy advice as if it were a forecast

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
How far **resolution day** sits from today when the **problem** is opened. Lengthens when **discover** prefers farther **resolution day**. A multi-problem discover tick still keeps some **near** horizons so reflection has a fast series (ADR 0012). More **revisions** of a near event are not a longer horizon. **Resolution day** moves later only if the public date moved.
_Avoid_: Counting daily revisions as horizon growth; using Y as a second field; filling `K` with only this week’s news

**Evidence regime**:
How a live problem is supposed to be reasoned: **news-now** (deadline, live talks) vs **analog/base-rate** (a past class of cases, a structural rate, or a method this plugin already claimed). Discover mixes both (ADR 0012). Analog-regime Motivation names a **structural class**. Scoring stays in the future.
_Avoid_: Historical cutoffs as the epoch; opening already-resolved questions; PIT/gold replay as discover

**Analog card**:
A loadable **case card** in `references/` (usually `references/cases/`). One pile: class, instantiations (country and case names plus sources), mechanism, base rate, optional **typical openings**, disanalogy, falsifiers. Not a second library of opening-type files. Predict consults it; reflection writes or culls it (ADR 0013, 0015, `references/structure.md`).
_Avoid_: graph-vault; a global ontology of types; scoring historical episodes as forecasts; “I remember 1992”; stripping country from instantiations; a separate opening-types folder

**Discover**:
Host tick that opens at most **K** new **problems** (each with **Motivation** and **resolution day**). When opening more than one problem, spread the batch across domains (ADR 0011) and across **horizon** / **evidence regime** (ADR 0012), weighed against what is already open. Analog-regime Motivation names the class (ADR 0013). It does not write **claim** or **justification**.
_Avoid_: Full forecast on discover; leaving a problem with no resolution day; letting a large `K` cluster in one domain or in the current week; scoring the past

**Predict**:
Host tick that writes **claim** and **justification** on **live problems** (first row or **revision** when the outcome changes). Formerly called due-today. It does not invent **resolution day**. Workers fill a Structure block from analog cards before news (`references/structure.md`).
_Avoid_: Due-today as the current name; waking only rows whose Due equals today; forecasting from uncited parametric history

**Memory**:
The plugin overlay. Not the ledger, not a sidecar store, and not harness conversation/artifacts (**openings** live there).
_Avoid_: Using memory for justification traces, Cursor automation memories, graph-vault, or tenant openings

**Motivation**:
Why this problem was opened. Stored on the problem in the ledger so Parent does not forget. Analog-regime rows name the structural class.
_Avoid_: Justification (that belongs on a dated claim)

**Parent**:
**Predict** tick: write today’s forecasts and revisions on **live problems**. **Discover** tick: open **problems** and set **resolution day**. **Reflection** tick: after **resolution day**, grade the whole series and grow or cull the plugin. Tiny edits may land on the live branch. Risky experiments go on an **experiment branch**; the reflect **host job** may start a Cursor automation on that branch. At most **K** new problems per discover tick; open inventory unbounded.
_Avoid_: Repo crontab; combining predict, discover, and reflect in one host job; making the reflector the daily traffic cop; experiment runs spawning more experiments

**Experiment branch**:
A git branch that holds a plugin experiment (new agent, skill, tool, or organisation). Not the live daily branch. Created from **reflection**. A host run on that branch tries the experiment. Merge or cull from a later live reflect tick (ADR 0016).
_Avoid_: Treating every overlay typo as a branch; in-repo daemon that launches agents

**Fan-out**:
Breadth of live claims plus ungated proactive **problem** finding. Reflection culls. Calibration is the whole series after **resolution day**, plus whether methods transferred.
_Avoid_: Multi-agent roster on one question; replaying old gold

**Reflection**:
After **resolution day**, grade every **dated claim** in the series (claim, justification, Structure block). The earliest matching **claim** is the prize. Then change the **plugin**: tiny edits on live; new capabilities may go to an **experiment branch** whose host run the reflect job may start (ADR 0016). Add or rewrite analog cards when a class transferred; cull false mechanisms.
_Avoid_: Sidecar `graph-vault/`; silent weight updates; grading only the last row; minting forecasts for historical episodes used to deepen a card; a plugin-owned automation daemon

**Training loop**:
Forward only: **discover** (problem + **resolution day**) → **predict** (claim + justification, **revisions** while live) → after **resolution day**, **reflection** → **plugin**. Three separate host jobs.
_Avoid_: PIT backtest as the epoch; gold Brier as quality; due-today as the current tick name; Due and Y as two clocks
