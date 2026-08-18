---
name: reflect
description: Parent reflection after Y — grade claim and justification, then grow or cull the plugin.
disable-model-invocation: true
---

# Reflect

Run from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date. This tick is reflection, not due-today.

## Steps

1. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Select `ledger.after_y(as_of)` where `as_of` is today. Completion: every selected claim has `Y` strictly before today; due-today claims stay on the due-today tick.
2. For each selected claim, grade the **Claim** and the **Justification** against what is now known. Completion: you can state a verdict for both (what transferred, what failed).
3. Change the plugin so the next tick is stronger. Write whatever the grade earned: new or rewritten skills, agents, references, scripts/tools, strategies, and instructions. Add files when a new capability is the point. Delete or merge overlay that failed to transfer. Durable facts belong in the overlay, usually `references/`. Do not write `graph-vault/`. Completion: the working tree shows plugin diffs that match the grades, or an explicit note on the claim that nothing in the plugin needed to change.

Parent is the overlay writer. Workers stay consumers until Parent adds or retargets an owner.
