---
name: discover
description: Parent discover tick — open at most K new problems with motivation and resolution day.
disable-model-invocation: true
---

# Discover

Run from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date. This tick is discovery, not predict and not reflect.

## Steps

1. Pull the current `harness-only` branch with fast-forward only. If git is dirty from another tick or pull fails, stop. Completion: you are on a clean `harness-only` matching origin.
2. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Completion: you have `K` and every existing problem motivation and **resolution day**.
3. Open at most `K` new problems under `## Problems`. Each needs **Motivation** and **Resolution** (`YYYY-MM-DD`, after today). Prefer farther **resolution day** as the overlay strengthens. No quality gate. Do not add dated claims. Do not write **claim** or **justification**. If opening more than one problem this tick, mix domains per `references/discovery.md` (at least three domains represented, no domain over half the batch, weighed against what is already open). Completion: new problem count is ≤ `K`; existing claims are untouched.
4. Re-read `ledger.md`. Completion: only problem headings, motivations, and resolution days changed; claim rows are identical.
5. If the ledger changed, commit and push to `harness-only`. Do not open a pull request.

Procedure detail: `references/discovery.md`.
