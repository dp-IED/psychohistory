---
name: reflect
description: Parent reflection after resolution day — grade claims and the overlay system, then grow or cull the plugin.
disable-model-invocation: true
---

# Reflect

Run from the repository root. Cursor: `.cursor/agents/parent.md`. Claude Code: `claude --plugin-dir .`. Live-tick contract: `references/host-jobs.md`. Today is the session calendar date. This tick is reflection, not predict. Parent starts `agents/reflector.md`; the reflector follows this skill.

## Steps

1. Pull the current `harness-only` branch with fast-forward only. If git is dirty from another tick or pull fails, stop. Completion: you are on a clean `harness-only` matching origin.
2. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Select `ledger.after_resolution(as_of)` where `as_of` is today. Completion: every selected claim belongs to a problem whose **resolution day** is strictly before today. Live problems stay on the predict tick.
3. Group selected claims by problem. Grade two things, same series:

   **Claims.** Every **Claim** and **Justification**, including the **Structure** block (`references/structure.md`). The earliest matching **claim** is the prize; a last-day-only hit is not the same as early skill. A last-day scrape that ignored a good analog is a method miss.

   **The system.** The overlay that was supposed to produce those claims: analog cards actually named, Structure procedure, skills/agents in force, and any deterministic tool predict was to use (`scripts/`, tests, a model, a GNN, an `exp/` branch). Ask: did this machinery transfer, was it load-bearing and wrong, or was it missing? A script that fired the wrong clock phase is a system miss, not only a worker miss. An `exp/` tool that did not beat the live overlay on this series does not merge. Be proactive: if the series shows a missing method **or** a broken tool, change overlay rather than waiting for a later tick.

   If a graded series (or its cited public instrument) named a **later** answering date than the problem’s **Resolution**, slide that **Resolution** later (same heading, later only). Select using the frozen date; then repair. If the new date is still on or after today, later predict ticks can score the next phase. If it is already before today, still slide so the ledger clock matches the public clock, and write overlay so the next freeze is repaired while that new date is still live. This tick does not write claims. Completion: you can state a verdict for the series **and** for the system (what transferred, what failed, which forecast day first matched, which card/script/exp to keep, rewrite, or cull).
4. Change the plugin so the next tick is stronger. Apply the **system** verdict, not only the claim verdict. The overlay is **not markdown-only**. Keep or deepen tools that transferred. Rewrite or delete scripts, models, and cards that were load-bearing and wrong. Write whatever the grade earned: new or rewritten skills, agents, references, strategies, instructions, **and deterministic systems** (`scripts/`, tests, small models, a GNN, a class-local index) when prose cannot keep the mechanism honest. You choose the tool from the miss; do not wait for a pre-designed cliodynamics stack (ADR 0019). Add or rewrite **analog case cards** when the Structure block earned a transferable class (deepen with several past instantiations from public sources; do not score those as forecasts). Cull or rewrite cards whose mechanism was load-bearing and wrong. Add files when a new capability is the point. Delete or merge overlay that failed to transfer. Durable facts belong in the overlay, usually `references/`. Do not write `graph-vault/`. Do not restore the France/warehouse GNN tree; a **new** simulator as overlay code is in scope. Completion: the working tree shows plugin diffs that match the grades, or an explicit note that nothing in the plugin needed to change.
5. If the overlay changed, commit and push to `harness-only`. Do not open a pull request.

The reflector is the overlay writer on this tick. Workers stay consumers until the reflector adds or retargets an owner.
