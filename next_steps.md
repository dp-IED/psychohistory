# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next

1. **Arm the host ticker** — three separate jobs (predict daily, reflect daily several hours later, discover weekly) via `/loop` or `/automate`. No repo daemon. Each job pull-ff, one tick, commit+push `harness-only`, no PR.
2. **Live claims** — keep scoring `P-usca-338` after 19 Aug; `P-il-knesset-26` is the hard months-out starter. Reflection edits the plugin only (ADR 0007).
3. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.
4. **Use of run results** — grill in [`docs/plans/2026-08-19-scale-forecast-roadmap.md`](docs/plans/2026-08-19-scale-forecast-roadmap.md). ADR 0014–0015: one analog-card pile; typical openings optional on that card; tenant openings private.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Merging until you choose to
- Conductor worktrees
