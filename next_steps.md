# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/). Continuity: [`docs/plans/ce-handoff-lfi-dsa-transcripts.md`](docs/plans/ce-handoff-lfi-dsa-transcripts.md).

## Next

1. **Home caption scrape (now)** — user is home (2026-08-25). Residential `yt-dlp` of remaining public Amfis 2026 (and new DSA) into gitignored `.scratch/`. No year playlist id as of 25 Aug; scrape known ids + channel uploads. Refresh [`references/party-sources.md`](references/party-sources.md). Open further live problems only if anti-cluster still holds. Do not score past recordings. On-site notes stay in chat unless already public. Procedure: the handoff file above.
2. **Host ticker** — already three Cursor dashboard automations (predict daily, reflect daily later, discover weekly). They **are firing** (22–24 Aug commits on `harness-only`). Contract: [`references/host-jobs.md`](references/host-jobs.md). Do not re-arm. Do not open PRs on ticks.
3. **Live claims / reflect** — `P-usca-338` is past (Resolution slid to 2026-08-22; no dated take-effect row). `P-il-knesset-26` remains the hard months-out starter. 25 Aug runoffs (`P-sc-sen-r-gop`, `P-ok-sen-d-run`, `P-ok-gov-r-gop`) grade on the **26 Aug** reflect. Five 24 Aug discover rows still need first predict claims: `P-ca-retal-08`, `P-232-poly-04`, `P-ndaa27-iltech`, `P-nobel-lit-26`, `P-genalo-bond`.
4. **Use of run results / evolution** — ADR 0014–0018. Transfer reopen in a graded class (do not reopen `P-usca-338`). Reflector + `exp/` (max 3).
5. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Treating “arm the ticker” as unfinished
- Asking the user for Google cookies / YouTube login
- Conductor worktrees
