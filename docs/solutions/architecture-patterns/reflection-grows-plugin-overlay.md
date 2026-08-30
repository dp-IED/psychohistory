---
title: "Reflection grows the plugin overlay, not a sidecar vault"
date: "2026-08-19"
category: architecture-patterns
module: "reflection/plugin overlay"
problem_type: architecture_pattern
component: assistant
severity: high
applies_when:
  - "After Y, Parent is about to grade a claim and change the plugin"
  - "Choosing where durable world facts live after reflection"
  - "Tempted to prefer patch-only overlay edits or restore graph-vault/"
related_components:
  - tooling
  - documentation
  - development_workflow
tags:
  - reflection
  - plugin-overlay
  - adr-0007
  - graph-vault
  - self-improving-plugin
---

# Reflection grows the plugin overlay, not a sidecar vault

## Context

Psychohistory is a harness-agnostic plugin whose training loop is meant to change the **plugin overlay**, not model weights. A training epoch is a scored pass whose output is plugin change: new or rewritten skills, agents, references, scripts/tools, and strategies, with later reflection culling overlay that failed (`docs/adr/0002-training-epochs-improve-plugin.md:3`; `CONTEXT.md:35-37`).

After a claim’s `Y`, Parent grades both the claim and the justification, then edits this repository so the next due-today tick is a stronger system (`docs/adr/0007-reflection-grows-the-plugin.md:3`). Adding overlay files is allowed when the grade earned a new capability (`docs/adr/0007-reflection-grows-the-plugin.md:3`; `skills/reflect/SKILL.md:15`). Durable world knowledge belongs in the overlay, usually `references/`, not in a sidecar `graph-vault/` (removed from this branch; do not recreate) (`docs/adr/0007-reflection-grows-the-plugin.md:5`; `CONTEXT.md:19-21`, `CONTEXT.md:75-77`).

That reading **supersedes** two easy misreads of older ADRs. ADR 0004 still describes scheduling correctly (the consuming harness owns `/loop` / `/automate`; no repo daemon; claims live in one ledger), but its vault-write half is superseded: reflection writes only the plugin and must not create `graph-vault/` (`docs/adr/0004-reflection-writes-graph-host-schedules.md:3-7`; `docs/adr/0007-reflection-grows-the-plugin.md:7`). ADR 0002 is the epoch-as-plugin-growth rule; it must not be read as “prefer never add a file” (`docs/adr/0007-reflection-grows-the-plugin.md:7`).

A competing product reading — **tighten existing files first, and do not add a skill per claim** — was rejected. Cull is later reflection deleting or merging overlay that failed to transfer, not a standing ban on adding files (`docs/adr/0007-reflection-grows-the-plugin.md:3`; `skills/reflect/SKILL.md:15`).

(session history) Earlier sessions on this branch locked dual-write reflection (plugin **and** `graph-vault/`), then deleted the inherited vault corpus while still planning for Parent to recreate the folder on first `/reflect`. Ticket-era `/reflect` also said tighten existing overlay files first. Those paths left a sidecar that would return on the first live reflect. ADR 0007 closes that: growth is overlay files; the vault folder is not a write target.

Role split: Parent is the overlay writer (`skills/reflect/SKILL.md:17`). Claim workers consume `skills/`, `agents/`, and `references/` and update only **Claim** and **Justification** on their ledger row; they leave overlay trees as they found them (`agents/claim-worker.md:6`, `agents/claim-worker.md:14`). Parent runs exactly one tick per host job (`agents/parent.md:8-12`).

Tests: `references/vault.md` was removed by this decision and must stay gone; reflect’s body must contain the substring `graph-vault` (the skill uses it as “do not write”) and “new or rewritten”; the worker must not mention `references/vault.md` (`tests/test_plugin_overlay.py:45`, `tests/test_plugin_overlay.py:66-67`, `tests/test_plugin_overlay.py:77`). As of `origin/harness-only` this is on the branch, not only a local working tree.

The 2026-08-18 cleanup plan still lists `graph-vault/.gitkeep` as a write target (`docs/plans/2026-08-18-plugin-first-branch-cleanup.md`). That is a pre-fix plan, historical relative to ADR 0007. The live reflect skill forbids writing `graph-vault/` (`skills/reflect/SKILL.md:15`).

## Guidance

Treat reflection as **plugin surgery with an add/cull pair**, not as a vault append and not as an “edit in place only” tax.

1. **After resolution day, grade then grow.** On a reflection tick, parse the ledger, select `ledger.after_resolution(as_of)`, grade the Claim/Justification series **and** the overlay system that produced it (cards, skills, scripts, models, `exp/` tools), then change the plugin so the next tick is stronger (`skills/reflect/SKILL.md`). Completion is either working-tree plugin diffs that match both verdicts, or an explicit note that nothing in the plugin needed to change.

2. **Add files when the grade earned a new capability.** New or rewritten skills, agents, references, **scripts/models**, strategies, and instructions are in-scope (`skills/reflect/SKILL.md`; ADR 0019). Do not interpret “cull later” as “never create a new skill” or as “markdown only.” One new skill (or agent, reference, script, or strategy) per transferred method is fine when that is the point of the grade. Avoid a mechanical new skill per dated claim only when the grade did not earn a distinct capability — that is a quality judgment, not a product ban.

3. **Cull is deletion/merge of failed overlay, later.** Delete or merge overlay that failed to transfer (`skills/reflect/SKILL.md:15`; `docs/adr/0002-training-epochs-improve-plugin.md:3`). Keep successful overlay. Do not pre-cull by refusing to add files.

4. **Put durable facts in the overlay, not a sidecar vault.** Prefer `references/` for long-form procedures and facts; put **code** in `scripts/` when the grade needs a system (`docs/adr/0007-reflection-grows-the-plugin.md`; ADR 0019). Do not write `graph-vault/`. Do not restore `references/vault.md`. Do not restore the France GNN tree. A **new** GNN or simulator as overlay code is in scope.

5. **Workers do not grow the plugin.** Claim workers update the assigned ledger row only (`agents/claim-worker.md:12-14`). Parent (or whoever the host wakes for `/reflect`) owns overlay diffs (`skills/reflect/SKILL.md:17`).

6. **Epochs are not weight updates.** Do not treat forecast scores as fine-tuning labels (`docs/adr/0002-training-epochs-improve-plugin.md:3-5`; `CONTEXT.md:35-37`).

7. **Keep the ticker on the host.** Scheduling remains `/loop`, `/automate`, or equivalent; do not add a repo-owned daemon (`docs/adr/0004-reflection-writes-graph-host-schedules.md:5`).

## Why This Matters

If reflection only tightens existing files, the training loop cannot acquire new methods, owners, tools, or strategies. ADR 0002 and ADR 0007 define the epoch as plugin growth including new files (`docs/adr/0002-training-epochs-improve-plugin.md:3`; `docs/adr/0007-reflection-grows-the-plugin.md:3`). Blocking adds would freeze capability at the seed overlay and turn “cull” into a ban instead of a later prune.

If reflection writes a sidecar `graph-vault/` (or a `references/vault.md` stand-in), durable knowledge splits from the loadable plugin. Overlay tests fail if `vault.md` exists or if workers point at it (`tests/test_plugin_overlay.py:45`, `tests/test_plugin_overlay.py:77`). The reflect skill must contain the substring `graph-vault` (`tests/test_plugin_overlay.py:66`; `skills/reflect/SKILL.md:15`).

If workers edit overlay, predict ticks mix forecasting with training, and Parent no longer has a single writer for growth vs cull (`agents/claim-worker.md`; `skills/reflect/SKILL.md`).

Misreading ADR 0004 as still authorizing vault writes would resurrect a layout the current ADRs forbid (`docs/adr/0004-reflection-writes-graph-host-schedules.md:3`; `docs/adr/0007-reflection-grows-the-plugin.md:7`).

## When to Apply

- **Reflection tick (`/reflect`) after `Y`:** grade, then add or rewrite overlay, or cull failed overlay (`skills/reflect/SKILL.md:9-15`; `agents/parent.md:9`).
- **Designing or reviewing overlay diffs:** add a skill/agent/reference when the grade shows a new transferable capability; rewrite/merge when the capability already has an owner file; delete when it failed to transfer (`docs/adr/0007-reflection-grows-the-plugin.md:3`).
- **Due-today and discovery ticks:** do not grow the overlay from claim-worker runs (`agents/claim-worker.md:14`; `agents/parent.md:8-12`).
- **Any change that would create `graph-vault/` or `references/vault.md`:** do not (`skills/reflect/SKILL.md:15`; `tests/test_plugin_overlay.py:45`).
- **Interpreting ADR 0002 / 0004:** use ADR 0007 as the overlay-growth and no-vault reading (`docs/adr/0007-reflection-grows-the-plugin.md:7`).
- **Uncommitted overlay work:** treat as pending until it is in the tree you ship; do not describe it as merged.

## Examples

**After a justified miss, add a capability (allowed).** A claim resolves after `Y`. Grading shows the worker lacked a repeatable source-check procedure. Parent adds a new skill (and optionally a `references/` note) and points the next owner at it. That matches “Write whatever the grade earned … Add files when a new capability is the point” (`skills/reflect/SKILL.md:15`) and “new or rewritten skills, agents, references, scripts/tools, and strategies” (`CONTEXT.md:75-76`). Tests require reflect’s body to contain “new or rewritten” (`tests/test_plugin_overlay.py:67`).

**Rejected anti-pattern: tighten-only / no new skill per claim as product.** Collapsing every grade into edits of the same two markdown files, or refusing a new skill because “files are bloat,” is the reading ADR 0007 supersedes (`docs/adr/0007-reflection-grows-the-plugin.md:7`). Cull is not that refusal; cull is deleting or merging overlay that failed (`skills/reflect/SKILL.md:15`).

**Later cull (required when transfer failed).** A skill added last epoch did not change justifications or scores. On a later reflect tick, Parent deletes or merges it (`docs/adr/0002-training-epochs-improve-plugin.md:3`; `skills/reflect/SKILL.md:15`). The inventory can grow first and shrink later; discovery remains ungated at the problem layer (`CONTEXT.md:59-61`, `CONTEXT.md:71-73`).

**No sidecar vault.** Durable facts go in `references/` (or another overlay path Parent owns). Reflect step 3 ends with “Do not write `graph-vault/`” (`skills/reflect/SKILL.md:15`). Overlay tests assert `references/vault.md` is absent (`tests/test_plugin_overlay.py:45`) and that reflect mentions `graph-vault` (`tests/test_plugin_overlay.py:66`). Claim-worker text must not mention `references/vault.md` (`tests/test_plugin_overlay.py:77`).

**Worker vs Parent.** Due-today: worker updates `ledger.md` Claim/Justification only (`agents/claim-worker.md:12-14`). Reflection: Parent follows `skills/reflect/SKILL.md` (`agents/parent.md:9`). Combining those in one host job is forbidden (`agents/parent.md:12`).

**Epoch is plugin change, not weights.** Scoring at `Y` informs what to add or cull in the overlay (`docs/adr/0002-training-epochs-improve-plugin.md:3-5`). Do not treat the ledger as a fine-tuning dataset in this product.

## Related

- [ADR 0007 — Reflection grows the plugin](../../adr/0007-reflection-grows-the-plugin.md)
- [ADR 0002 — Training epochs improve the plugin](../../adr/0002-training-epochs-improve-plugin.md)
- [ADR 0004 — Host ticker (vault-write half superseded)](../../adr/0004-reflection-writes-graph-host-schedules.md)
- `skills/reflect/SKILL.md`
- `CONTEXT.md` (Reflection, Training epoch, Plugin overlay)
