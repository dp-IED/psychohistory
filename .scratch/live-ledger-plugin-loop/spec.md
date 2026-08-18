---
title: "Live ledger plugin loop — overlay, ledger, Parent due-today"
status: ready-for-agent
labels: [ready-for-agent]
feature: live-ledger-plugin-loop
created: 2026-08-18
---

# Live ledger plugin loop

Harness-invoked, in-repo plugin: overlay, `ledger.md`, Parent due-today. Cursor Agent CLI is the first host. Claude Code and Codex later use the same in-project invocation.

## Problem Statement

This repo is meant to be a training bed for a live forecasting plugin. After the grill, the durable shape is a Claude-style overlay (`skills/`, `agents/`, `references/`) plus a root ledger, with a Parent that runs due-today and reflects at `Y` into the plugin and the vault graph. None of that overlay or ledger exists yet.

The host has to be able to **edit** the plugin. Training epochs are file writes, not weight updates. Claude Code and Codex plugin-install caches are unstable, so installing a copy and training against the cache would lose or desync epochs. The first candidate host is Cursor Agent CLI. The Parent must run as a project-root agent on this repo, spawn workers that consume the same live overlay, and keep every write in the working tree.

## Solution

Ship the overlay and ledger **in this repository**. Invoke the harness in the project: the Parent’s project is the repo root; spawned subagents get that same root as their plugin dir and are **consumers only**. Do not use Claude Code or Codex plugin install caches.

On a due-today tick, Parent reads the ledger, wakes workers whose claims are due today, and those workers write claims and justifications back to the ledger. After `Y`, Parent reflects: grade claim and justification, then edit the overlay (with skill-bloat discipline) and the vault graph. Discovery is a different, rarer tick (ungated problems, `K` new per tick, motivation on each problem). This spec specifies that Parent behavior; it does not wire the host ticker.

The test seam is the ledger as a deterministic schedule book. Pytest, no LLM.

## User Stories

1. As an operator, I want the plugin overlay to live in this repo, so training epochs are ordinary working-tree edits.
2. As an operator, I want Cursor Agent CLI to be the first host, so Parent is invoked from the project root rather than from an installed plugin copy.
3. As an operator, I want Claude Code and Codex to use the same in-project invocation later, so we never depend on their plugin install caches.
4. As Parent, I want the repo root to be my project, so I can read the ledger, spawn workers, and edit overlay and vault with the host’s normal file tools.
5. As Parent, I want spawned subagents to receive the repo root as their plugin dir, so they load the same live overlay I am training.
6. As a worker subagent, I want to consume skills, agent defs, and references without editing them, so only Parent mutates the plugin.
7. As a worker, I want to append or update my dated claims and justifications on the ledger, so my work product is the schedule book, not the overlay.
8. As an operator, I want a single `ledger.md` at the repo root, so Parent has one place to read problems and claims.
9. As Parent on due-today, I want to read the ledger and wake only agents whose claims are due today, so the daily tick is a wakeup, not a discovery sweep.
10. As a worker woken for a claim, I want the claim’s due date, owner, text, and prior justification, so I resume rather than invent a new problem.
11. As a problem on the ledger, I want a stored motivation, so Parent does not forget why it was opened.
12. As a dated claim, I want a justification (reasoning trace), so reflection can grade the method after `Y`.
13. As an operator, I want `K` on the ledger starting at 1, so each discovery tick opens at most that many new problems.
14. As Parent on a discovery tick, I want to open ungated problems up to `K`, so intake is capped and inventory is not.
15. As an operator, I want due-today and discovery to remain separate ticks, so a daily wakeup cannot silently become a discovery run.
16. As Parent after `Y`, I want to grade the claim and the justification, so the epoch has a scored pass.
17. As Parent after that grade, I want to edit skills, agents, and references, so the epoch improves the plugin.
18. As Parent after that grade, I want to write entities, threads, and concepts into the vault graph, so reflection is not plugin-only.
19. As Parent editing the overlay, I want skill-bloat discipline (tighten or disclose before adding files), so epochs do not accumulate sediment.
20. As an operator, I want foundation weights frozen, so this loop never becomes a fine-tune.
21. As a portable skill or agent, I want no import of hermes or the leftover in-repo orchestrator, so any host can load the overlay.
22. As an operator, I want deterministic helpers to stay where they already live until a later move into plugin scripts, so this spec does not relocate the Python stack.
23. As pytest, I want a ledger fixture plus an as-of date to determine due-today claims, problems with motivation, claims with justification, and `K`, so the schedule book is locked without an LLM.
24. As an operator, I want default pytest to keep passing without hermes on PATH, so the overlay work does not revive the temporary runner.
25. As Parent, I want fan-out to mean many live claims and ungated problems, so workers are not a multi-agent debate on one question.
26. As an operator, I want no cap on total open claims, so reflection culls losers instead of an intake quality gate.
27. As an operator, I want live markets available as a discovery and scoring surface, without this spec building the parked historical Polymarket testbed.
28. As a later host job, I want `/loop` or `/automate` to be able to invoke Parent due-today versus discovery, without this spec wiring those jobs.
29. As an operator, I want every write (overlay, ledger, vault) in the repo, so git is the training history.
30. As a worker, I want the overlay I consume to be the files on disk at spawn time, so a Parent edit in this session is visible to the next worker without a reinstall.
31. As Parent, I want host-specific discovery (slash skill or project agent) to be a thin pointer into the overlay, so we do not keep two copies of Parent or worker bodies.
32. As an operator, I want GNN analysis of the vault graph to stay parked, so reflection may write the graph without scoring it.
33. As an operator, I want PIT vault filters left as legacy, so due-today is not implemented as a historical cutoff replay.
34. As a skill author, I want `/writing-for-agents` discipline when adding or tightening skills, so pointers, steps, and completion criteria stay load-bearing.
35. As Parent, I want to refuse to open a problem on a due-today tick, so discovery cannot leak into the daily wakeup.
36. As a worker, I want an owner field on the claim that names me, so Parent’s due-today fan-out is rostering from the ledger rather than a hardcoded list in Python.
37. As an operator, I want the leftover orchestrator to remain unextended, so this product is plugin plus host, not a new in-repo runner.
38. As Parent, I want references available for vault conventions and procedures, so workers load long-form material on demand instead of stuffing it into every skill.
39. As reflection, I want to cull or keep problems using what was learned at `Y`, so ungated intake has a later filter.
40. As an operator, I want this spec’s first demo to be: empty-enough ledger + Parent due-today in Cursor Agent CLI at repo root, spawning a consumer worker that reads overlay and writes a claim, so the loop is visible without a ticker.

## Implementation Decisions

- The repository **is** the plugin and the Parent’s project. Invoke the first host (Cursor Agent CLI) from the repo root. Do not install the overlay through Claude Code or Codex plugin marketplaces or caches.
- Overlay layout stays Claude-style at the plugin root: `skills/`, `agents/`, `references/` (and `scripts/` later for deterministic plugin code). That tree is the single source of truth for skill and agent bodies.
- Parent is a project-root agent session. It reads the ledger, wakes due workers, and is the only writer of overlay files. After `Y` it also writes the vault graph.
- Worker subagents are **consumers only** of the overlay. They are spawned with the repo root as plugin dir / working tree so they load the live overlay. They may update ledger claims they own; they do not edit skills, agents, or references.
- If the first host needs a project-level skill or agent file to *invoke* Parent, that file is a thin pointer into the overlay. Do not duplicate Parent or worker prompts into a second tree.
- Claude Code and Codex, when added, use the same pattern: session opened on this repo (plugin dir = repo root), not a cached install. Cache instability is why.
- `ledger.md` at repo root is the schedule book. Two sections: problems (each with motivation) and dated claims (due, owner, claim, justification). `K` lives on that file; start at 1. Exact markdown syntax is an implementation choice locked by the ledger-reader tests, not a second schema language (no parallel JSON book).
- Due-today: Parent selects claims whose due date is today (as-of date supplied by the tick), wakes the named owners, does not open new problems.
- Discovery: specified on Parent as a separate tick — ungated problems, at most `K` new, motivation required. Not scheduled by this spec.
- Reflection: Parent-only; grades claim and justification; overlay edits with bloat guard (prefer tighten / disclose / leading-word refactor over new files); vault graph writes in scope; GNN analysis not in scope.
- Deterministic Python stays in the existing harness package until a later move. New ledger reading for tests is a small deterministic surface; it is not an LLM port and is not a rewrite of the leftover orchestrator.
- Portable overlay markdown must not instruct agents to import hermes or the leftover orchestrator.
- Host `/loop` and `/automate` are not wired here. The first demo is a manual Parent invocation at repo root.
- Polymarket historical testbed, PIT backtest-as-objective, France/GNN/builder, and in-repo orchestrator-as-product remain refused.

## Testing Decisions

- Good tests assert external behavior of the schedule book: given ledger text and an as-of date, which claims are due, which problems exist with motivation, what `K` is, and that justification is present on claims. They do not parse Parent prose, do not call an LLM, and do not inspect overlay wording.
- The module under test is the deterministic ledger reader (the agreed seam). Existing pytest in this repo is prior art: fixture in, pure function out, default suite must not require hermes on PATH.
- Do not add PIT-admissibility tests for this work. Re-run those only if vault layout or the PIT helper changes.
- Do not test host ticker behavior. Do not test Claude Code or Codex caches. An optional manual check for the first host: Parent invoked at repo root can spawn a worker that reads overlay and writes a ledger claim; that check is not a pytest seam.
- Overlay and vault writes from reflection are agent behavior; they are specified, not unit-tested in this spec.

## Out of Scope

- Wiring `/loop` or `/automate` (due-today vs discovery jobs)
- Claude Code / Codex as first-run hosts (same in-repo pattern later; not this implementation pass)
- Plugin marketplace or cache installs
- Rewriting or replacing the leftover orchestrator as the product
- Historical PIT Brier / gold backtest as the epoch objective
- GNN analysis of the vault graph
- Polymarket gold testbed, probes, and schemas work
- Fine-tuning / weight updates
- A `wakeups/` directory or one file per hypothesis
- Quality gate before opening problems; cap on total open claims
- Discovery on the same tick as due-today
- Relocating `graph-vault/` into the plugin
- Moving existing harness Python into plugin `scripts/`
- Conductor worktrees; merging this branch to `origin/main`

## Further Notes

- Domain language: `CONTEXT.md`. Binding decisions: ADRs 0001–0006. Work order: `next_steps.md`.
- Skill authorship should follow `/writing-for-agents` (pointers, steps with completion criteria, disclose reference, prune duplication).
- This spec is a snapshot for `/to-tickets` / `/implement`. Durable changes belong in `CONTEXT.md` and ADRs, not in an edited spec after ship.
- Tracker was not configured for this repo (`/setup-matt-pocock-skills` never run). This file is the local publish under `.scratch/live-ledger-plugin-loop/`. Re-home to GitHub issues if that setup is run later.
