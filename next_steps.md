# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/).

## Next

1. **Host ticker** — already three Cursor dashboard automations (predict daily, reflect daily later, discover weekly). No repo daemon. Live-tick contract: [`references/host-jobs.md`](references/host-jobs.md). Each job pull-ff, one tick, commit+push `harness-only`, no PR. Party-source ledger rows land on `harness-only` when this branch merges.
2. **Live claims** — keep scoring `P-usca-338` after 19 Aug; `P-il-knesset-26` is the hard months-out starter. Reflector edits the plugin (ADR 0007, 0017).
3. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.
4. **Use of run results / evolution** — ADR 0014–0018. Party-sourced discover; reflector + `exp/` experiments.
5. **Party-source scrape** — first corpus is in [`references/party-sources.md`](references/party-sources.md) (welcome + DSA keynotes/summit). **Rest of Amfis 2026** waits for the official LFI playlist and a home-network caption pull; parked on [`docs/plans/2026-08-19-scale-forecast-roadmap.md`](docs/plans/2026-08-19-scale-forecast-roadmap.md). Do not score past recordings. On-site notes stay in chat unless already public.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Merging until you choose to
- Conductor worktrees
