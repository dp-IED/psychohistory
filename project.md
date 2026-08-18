# Psychohistory

A **training bed** for a forecasting **plugin** that any harness can load (Cursor, Claude Code, Codex, …). This repository is the plugin root.

## What “done” means here

1. **Plugin overlay.** `skills/`, `agents/`, `references/` at the repo root. Host owns orchestration (`--plugin-dir .`, then separate `/due-today`, `/reflect`, `/discover` jobs via `/loop` or `/automate`).
2. **Live training loop.** Discover problems and claims → `ledger.md` → Parent wakes due agents → at `Y` reflect into the plugin (new skills, agents, tools, strategies; cull what failed).
3. **Epochs grow the plugin.** Frozen model weights. Later reflection culls overlay that did not transfer.

## Temporary vs durable

- **Durable:** plugin overlay, `ledger.md`, `harness.ledger`, `CONTEXT.md`, `docs/adr/`.
- **Not on this branch:** hermes orchestrator, PIT filters, gold probes, inherited vault corpus (git history).

## Not this repo on this branch

France GDELT GNN, warehouse, learned graph builder, and the old eval trees were removed from this branch. Retrieve from git history if needed.
