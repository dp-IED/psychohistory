---
name: discover
description: Parent discovery tick — open at most K new problems with motivation.
disable-model-invocation: true
---

# Discover

Run from the repository root with the plugin loaded in place (`claude --plugin-dir .`). Today is the session calendar date. This tick is discovery, not due-today and not reflect.

## Steps

1. Read `ledger.md`. Parse it with `harness.ledger.parse_ledger`. Completion: you have `K` and every existing problem motivation.
2. Open at most `K` new problems under `## Problems`. Each needs a **Motivation**. No quality gate. Do not add dated claims on this tick. Completion: new problem count is ≤ `K`; existing claims are untouched.
3. Re-read `ledger.md`. Completion: only problem headings and their motivations changed; claim rows are identical.

Procedure detail: `references/discovery.md`.
