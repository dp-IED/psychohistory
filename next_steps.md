# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next

1. **Arm the host ticker** — three separate jobs (due-today daily, reflect daily, discover weekly) via `/loop` or `/automate`. No repo daemon. Cloud jobs must check out `harness-only` after this branch is pushed.
2. **Live claims** — keep scoring `C-usca-338-deal` at `Y`; replace `C-seed` when convenient. Reflection edits the plugin only (ADR 0007).
3. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Merging until you choose to
- Conductor worktrees
