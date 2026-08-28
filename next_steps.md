# Next steps

This branch is a harness-agnostic **plugin**. Do not restore the hermes runner or PIT eval trees onto it.

Domain language: [`CONTEXT.md`](CONTEXT.md). Decisions: [`docs/adr/`](docs/adr/). Continuity: [`docs/plans/ce-handoff-lfi-dsa-transcripts.md`](docs/plans/ce-handoff-lfi-dsa-transcripts.md).

## Next

1. **Home caption scrape** — done 2026-08-25 from this residential IP. No `AMFIS 2026` year playlist; multi-year *Nos AMFIS d'été* + channel RSS. French captions in gitignored `.scratch/`. Index refreshed. No new ledger problems (anti-cluster: France already has 15 Sep street; Senate PJL and Canal+ have no public vote/hearing day). Do not score past recordings.
2. **Host ticker** — already three Cursor dashboard automations (predict daily, reflect daily later, discover weekly). They **are firing** (22–24 Aug commits on `harness-only`). Contract: [`references/host-jobs.md`](references/host-jobs.md). Do not re-arm. Do not open PRs on ticks. Do not hand-run predict / reflect / discover while the jobs are armed.
3. **Live claims / reflect** — `P-usca-338` is past (Resolution 2026-08-22; tariff card already transferred). 25 Aug runoffs re-graded 28 Aug: Graham, Thomas, and Mazzei still match the 19 Aug rows (prize = early hit). No Resolution slide. Overlay deepen: MAGA consolidator outranks a non-consolidator first-round plurality (`references/cases/majority-primary-runoff.md`). `P-il-knesset-26` remains the hard months-out starter. Near clocks: Iceland 29 Aug, GW 30 Aug, MA 1 Sep.
4. **Owner between ticks** — overlay review after a graded series, not another tick. First pass: after the 26 Aug runoff reflect. See **While the ticker runs** below.
5. **Use of run results / evolution** — ADR 0014–0018. Transfer reopen in a graded class (do not reopen `P-usca-338`). Reflector + `exp/` (max 3).
6. **Polymarket testbed** — historical gold/probes; retrieve from git history after the live loop is ticking. Live markets are a discovery surface, not that testbed.

## While the ticker runs

The ticker owns the loop. The human job is **review**, not a fourth automation.

- **After a graded series** (first: 26 Aug runoff reflect): read the overlay diff. Cards that survived, cards that should have been written and were not, Motivation treated as a card, sports refill, 338-style reopen. Chat back if a change should stick; otherwise leave the next tick alone.
- **Chat, not the ledger:** openings you want on a later discover pass; hallway / on-site notes that are not already public. If you make a forecast in chat, copy it into the ledger anonymized (ADR 0014).
- **Keep `harness-only` clean** so automations can ff-pull. Do not leave uncommitted work on the live branch.
- **Near clocks after 26 Aug:** Iceland 29 Aug, GW 30 Aug, MA 1 Sep, Canada retal 8 Sep. `P-il-knesset-26` is the long series. France September street is taken; wait for a named Senate vote day or Canal+ hearing day. Do not refill sports as a quota.
- **Do not** restore parked trees to fill quiet (PIT, hermes, gold vault, France GNN, Polymarket historical testbed, tournament watcher).

## Do not pick up

- Restoring `harness/orchestrator.py`, PIT, or the old `graph-vault/` corpus onto this branch
- Historical PIT backtest as the epoch objective
- Graph builder / WM / France GNN
- Treating “arm the ticker” as unfinished
- Asking the user for Google cookies / YouTube login
- Conductor worktrees
