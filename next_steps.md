# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next

1. **Host ticker** — wire due-today vs discovery vs reflect with `/loop` or `/automate` (not a repo daemon).
2. **Live claims** — replace the seed ledger row with real wakeups; Parent creates a vault on first reflection write.
3. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Merging until you choose to
- Conductor worktrees
