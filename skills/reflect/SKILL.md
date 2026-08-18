---
name: reflect
description: Parent reflection after Y — grade claim and justification, then edit overlay and vault.
disable-model-invocation: true
---

# Reflect

Run from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date. This tick is reflection, not due-today.

## Steps

1. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Select `ledger.after_y(as_of)` where `as_of` is today. Completion: every selected claim has `Y` strictly before today; due-today claims stay on the due-today tick.
2. For each selected claim, grade the **Claim** and the **Justification** against what is now known. Completion: you can state a verdict for both (what transferred, what failed).
3. Edit the overlay from those verdicts. Tighten or disclose an existing skill, agent, or reference before adding a file. Completion: the working tree shows overlay edits that match the grades, or an explicit note on the claim that nothing in the overlay needed to change.
4. Write vault graph material that the grades earned: entities, threads, or concepts under `graph-vault/` following `references/vault.md`. Create `graph-vault/` if it is missing. Completion: each selected claim has at least one vault write, or an explicit note that the vault already held the fact.

Parent is the overlay writer. Workers stay consumers.
