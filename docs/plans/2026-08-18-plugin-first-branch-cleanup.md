# Plugin-first branch cleanup

**Branch:** `harness-only`  
**Date:** 2026-08-18  
**Goal:** The merged tree is only the live plugin loop and the docs that define it. No PIT, hermes, Polymarket eval, and **no inherited `graph-vault/` corpus**. Those trees stay in git history.

## Problem

`harness-only` still carries ~1300 files from the old training bed. Recent work is three commits (`c289ed1b`, `f3d7a939`, `04c5e847`) plus the uncommitted grill docs. Everything else on the branch is leftover and should not merge.

## Method

Ordinary **delete commits** on `harness-only`. No history rewrite, no force-push. Retrieve later from `origin/main`, `b8ac2d66`, or this branch’s parent `e867bec3`.

## Keep (this is the whole product tree)

```text
.claude-plugin/plugin.json
skills/due-today/SKILL.md
skills/reflect/SKILL.md
agents/parent.md
agents/claim-worker.md
references/discovery.md
references/vault.md
.cursor/agents/parent.md
.cursor/rules/memory.mdc          # rewrite to plugin-first
ledger.md
harness/ledger.py
harness/__init__.py               # ledger exports only
tests/test_ledger.py
tests/test_plugin_overlay.py
README.md                         # rewrite
project.md                        # rewrite
next_steps.md                     # rewrite
CONTEXT.md                        # restore (grill; not currently in the tree)
docs/adr/0001 … 0006              # restore
docs/plans/2026-08-18-plugin-first-branch-cleanup.md  # this plan; optional after ship
pyproject.toml                    # slim
requirements.txt                  # slim
.gitignore                        # add .scratch/
graph-vault/.gitkeep              # empty write target for reflection; no corpus
```

That is the keep list. If it is not in this list, it leaves the branch.

## Delete (history retains)

Everything else currently tracked, including:

- All of `harness/` except `ledger.py` + slim `__init__.py`
- `scripts/` `ingest/` `schemas/` `data/` `forecasts/` `pit_blind_test/` `seed30_blind_test/` `_specs/` `artifacts/`
- All other `tests/`
- `docs/polymarket_agentic_harness.md` `docs/history/`
- **Entire current `graph-vault/` contents** (domains, cases, runs, forecasts, agent-roles, timeline, gold, …) — replace with `graph-vault/.gitkeep` so Parent still has a folder to write into
- Any other root files not in the keep list

## Restore / rewrite

Grill artifacts (`CONTEXT.md`, ADRs 0001–0006) were never committed and are missing from the working tree. Restore them in the same cleanup so the branch has strategy docs.

Rewrite `README.md`, `project.md`, `next_steps.md`, `.cursor/rules/memory.mdc` to plugin-first (in-place `--plugin-dir .`, ledger, due-today, reflect). Drop hermes/PIT runbooks.

`pyproject.toml`: plugin-first description; package `harness*` only; drop `openai` / `pydantic` unless something remaining imports them (`ledger.py` does not).

## Tests after cleanup

`pytest` (no hermes): `tests/test_ledger.py`, `tests/test_plugin_overlay.py`.  
`claude plugin validate .`

## Sequencing

1. Restore `CONTEXT.md` + `docs/adr/0001`–`0006`.
2. Slim `harness/__init__.py`.
3. Delete everything not on the keep list; leave `graph-vault/.gitkeep`.
4. Rewrite README / project / next_steps / memory / pyproject.
5. Pytest + plugin validate.
6. Merge when you choose.

## Out of scope

Host ticker, history rewrite, un-parking GNN/France/gold testbed (check out history when needed).
