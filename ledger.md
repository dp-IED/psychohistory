# Ledger

Parent's schedule book. `K` is max new problems per discovery tick.

K: 1

## Problems

### P-seed — Open the live loop

Motivation: The training bed needs one problem row before ungated discovery runs, so Parent has a book to read.

### P-usca-338 — US Section 338 tariffs on Canada

Motivation: A public 12:01 a.m. ET 19 Aug 2026 deadline on 50% Section 338 duties is in the news today, so the live loop can score a claim within a day instead of waiting on the 2099 seed.

## Claims

### C-seed

- Problem: P-seed
- Due: 2099-01-01
- Y: 2099-01-01
- Owner: claim-worker
- Claim: This seed claim exists so the ledger is parseable before live wakeups.
- Justification: Placeholder reasoning trace; replace on the first real due-today write.

### C-usca-338-deal

- Problem: P-usca-338
- Due: 2026-08-18
- Y: 2026-08-19
- Owner: claim-worker
- Claim: The US and Canada will not, before 12:01 a.m. ET on 19 Aug 2026, announce a deal that delays or cancels the scheduled 50% Section 338 tariffs on the listed Canadian goods.
- Justification: As of 2026-08-18 evening (about six hours before the deadline in ET), no bilateral announcement exists. July 20 White House Section 338 proclamations still set 50% additional duties on listed Canadian goods at 12:01 a.m. ET 19 Aug 2026, with statutory power to suspend later but no public suspend/revoke found today. CBC (Tasker): Carney–Trump Tuesday call described only as “ongoing negotiations,” with unnamed sources still at loggerheads on autos (US 15% headline / 7.5% effective vs Canada wanting lower) plus dairy TRQ and alcohol-ban politics. Globe and Mail: LeBlanc/Charette still in Washington; Wiseman recalled; auto impasse lasting days; Canada seeking 338 deferral plus 232 cuts; Canadian sources split, some expecting a 338 delay, US-briefed sources skeptical of more than a short extension because 338 is leverage. Financial Post (anon US-side): Tuesday deal a “coin flip or worse,” US “best offer” already tabled, leader call a long shot, delay “appears unlikely.” AP: talks continue; LeBlanc “the work is continuing”; Greer (Iowa) will not tolerate retaliation. Claim needs a public US–Canada *deal* that delays or cancels 338, not a unilateral pause or a 232-only bargain that leaves 338 in force. Gaps that remain (autos, lumber, dairy, alcohol) plus US signaling make a pre-deadline deal less likely than tariffs taking effect or silence. Vault: `graph-vault/` is not present on this branch, so no vault facts. No CBP live entry feed, White House press pool, or official prediction-market API in this session.
