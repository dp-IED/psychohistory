---
name: reflect
description: Parent reflection after resolution day — grade the whole claim series, then grow or cull the plugin.
disable-model-invocation: true
---

# Reflect

Run from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date. This tick is reflection, not predict.

## Steps

1. Pull the current `harness-only` branch with fast-forward only. If git is dirty from another tick or pull fails, stop. Completion: you are on a clean `harness-only` matching origin.
2. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Select `ledger.after_resolution(as_of)` where `as_of` is today. Completion: every selected claim belongs to a problem whose **resolution day** is strictly before today. Live problems stay on the predict tick.
3. Group selected claims by problem. Grade the **whole series**: every **Claim** and **Justification**. The earliest matching **claim** is the prize; a last-day-only hit is not the same as early skill. Be proactive: if the series shows a missing method, add overlay rather than waiting for a later tick. Completion: you can state a verdict for the series (what transferred, what failed, which forecast day first matched).
4. Change the plugin so the next tick is stronger. Write whatever the grade earned: new or rewritten skills, agents, references, scripts/tools, strategies, and instructions. Add files when a new capability is the point. Delete or merge overlay that failed to transfer. Durable facts belong in the overlay, usually `references/`. Do not write `graph-vault/`. Completion: the working tree shows plugin diffs that match the grades, or an explicit note that nothing in the plugin needed to change.
5. If the overlay changed, commit and push to `harness-only`. Do not open a pull request.

Parent is the overlay writer. Workers stay consumers until Parent adds or retargets an owner.
