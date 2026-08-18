---
name: due-today
description: Parent due-today tick — wake ledger owners whose claims are due today.
disable-model-invocation: true
---

# Due-today

Run this from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date.

## Steps

1. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger` (the schedule-book reader). Completion: you have `K`, every problem motivation, and every dated claim.
2. Select due claims: `ledger.due_today(as_of)` where `as_of` is today. Completion: the due set is known; claims due any other day stay asleep.
3. For each due claim, spawn the subagent whose name equals `Owner`, working in this repo root (plugin dir = this project). Pass claim id, due date, claim text, justification, and the parent problem's motivation. Completion: every due owner has been spawned.
4. After workers return, re-read `ledger.md`. Completion: each due claim's **Claim** and **Justification** reflect that worker's write; no new problem headings were added.

This tick updates due claims. Open problems on a discovery tick; that procedure is `references/discovery.md`.
