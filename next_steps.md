# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/). Continuity: [`docs/plans/ce-handoff-lfi-dsa-transcripts.md`](docs/plans/ce-handoff-lfi-dsa-transcripts.md).

## Next

1. **Home caption scrape** — done 2026-08-25 from this residential IP. No `AMFIS 2026` year playlist; multi-year *Nos AMFIS d'été* + channel RSS. French captions in gitignored `.scratch/`. Index refreshed. No new ledger problems (anti-cluster: France already has 15 Sep street; Senate PJL and Canal+ have no public vote/hearing day). Do not score past recordings.
2. **Host ticker** — already three Cursor dashboard automations (predict daily, reflect daily later, discover weekly). They **are firing** (22–24 Aug commits on `harness-only`). Contract: [`references/host-jobs.md`](references/host-jobs.md). Do not re-arm. Do not open PRs on ticks.
3. **Live claims / reflect** — `P-usca-338` is past (Resolution slid to 2026-08-22; no dated take-effect row). `P-il-knesset-26` remains the hard months-out starter. 25 Aug runoffs (`P-sc-sen-r-gop`, `P-ok-sen-d-run`, `P-ok-gov-r-gop`) grade on the **26 Aug** reflect (polls still open 25 Aug; no revision). First predict claims written 25 Aug on the five 24 Aug discover rows: `P-ca-retal-08`, `P-232-poly-04`, `P-ndaa27-iltech`, `P-nobel-lit-26`, `P-genalo-bond`.
4. **Use of run results / evolution** — ADR 0014–0018. Transfer reopen in a graded class (do not reopen `P-usca-338`). Reflector + `exp/` (max 3).
5. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Treating “arm the ticker” as unfinished
- Asking the user for Google cookies / YouTube login
- Conductor worktrees
