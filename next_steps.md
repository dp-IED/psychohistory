# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next

1. **Arm the host ticker** — three separate jobs (predict daily, reflect daily several hours later, discover weekly) via `/loop` or `/automate`. No repo daemon. Each job pull-ff, one tick, commit+push `harness-only`, no PR.
2. **Live claims** — keep scoring `P-usca-338` after 19 Aug; `P-il-knesset-26` is the hard months-out starter. Reflector edits the plugin (ADR 0007, 0017).
3. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.
4. **Use of run results / evolution** — ADR 0014–0018. Party-sourced discover; reflector + `exp/` experiments.
5. **Party-source scrape (next session)** — public LFI Amfis / DSA channel transcripts → discover index + live future problems. Handoff: [`docs/plans/ce-handoff-lfi-dsa-transcripts.md`](docs/plans/ce-handoff-lfi-dsa-transcripts.md).

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Merging until you choose to
- Conductor worktrees
