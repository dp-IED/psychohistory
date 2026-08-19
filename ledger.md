# Ledger

Parent's schedule book. `K` is max new problems per discover tick.

K: 10

## Problems

### P-usca-338 — US Section 338 tariffs on Canada

Resolution: 2026-08-19

Motivation: A public 12:01 a.m. ET 19 Aug 2026 deadline on 50% Section 338 duties. Near-term score so reflection has a first series quickly.

### P-il-knesset-26 — Largest party in Israel's 26th Knesset

Resolution: 2026-10-27

Motivation: First Israeli national election since the Gaza war. Seat lead is not a one-source scrape: wartime incumbency, Gaza and Iran, polling error, and coalition math all move the outcome. A months-out resolution day gives predict a long revision trail so reflection has methods to add, not a one-day news hit.

## Claims

### C-usca-338-deal

- Problem: P-usca-338
- Forecast: 2026-08-18
- Owner: claim-worker
- Claim: The US and Canada will not, before 12:01 a.m. ET on 19 Aug 2026, announce a deal that delays or cancels the scheduled 50% Section 338 tariffs on the listed Canadian goods.
- Justification: As of 2026-08-18 evening (about six hours before the deadline in ET), no bilateral announcement exists. July 20 White House Section 338 proclamations still set 50% additional duties on listed Canadian goods at 12:01 a.m. ET 19 Aug 2026, with statutory power to suspend later but no public suspend/revoke found today. CBC (Tasker): Carney–Trump Tuesday call described only as “ongoing negotiations,” with unnamed sources still at loggerheads on autos (US 15% headline / 7.5% effective vs Canada wanting lower) plus dairy TRQ and alcohol-ban politics. Globe and Mail: LeBlanc/Charette still in Washington; Wiseman recalled; auto impasse lasting days; Canada seeking 338 deferral plus 232 cuts; Canadian sources split, some expecting a 338 delay, US-briefed sources skeptical of more than a short extension because 338 is leverage. Financial Post (anon US-side): Tuesday deal a “coin flip or worse,” US “best offer” already tabled, leader call a long shot, delay “appears unlikely.” AP: talks continue; LeBlanc “the work is continuing”; Greer (Iowa) will not tolerate retaliation. Claim needs a public US–Canada *deal* that delays or cancels 338, not a unilateral pause or a 232-only bargain that leaves 338 in force. Gaps that remain (autos, lumber, dairy, alcohol) plus US signaling make a pre-deadline deal less likely than tariffs taking effect or silence. Vault: `graph-vault/` is not present on this branch, so no vault facts. No CBP live entry feed, White House press pool, or official prediction-market API in this session.

### C-usca-338-deal-announced

- Problem: P-usca-338
- Forecast: 2026-08-19
- Owner: claim-worker
- Claim: Before 12:01 a.m. ET on 19 Aug 2026, the United States and Canada publicly announced an arrangement that delayed (did not cancel) the scheduled 50% Section 338 tariffs on listed Canadian goods; the duties did not take effect at that deadline and were postponed to 12:01 a.m. ET on 22 Aug 2026.
- Justification: Forecast day 2026-08-19 (resolution day). Overlay: `ledger.md` (C-usca-338-deal), `agents/claim-worker.md`, `skills/predict/SKILL.md`, `CONTEXT.md`. No `graph-vault/`. Outcome moved vs the 18 Aug row (no pre-deadline public US–Canada deal delaying or cancelling 338). AP (Wiseman/Gillies): Trump said Tuesday the countries reached a last-minute deal less than two hours before 12:01 a.m. Wednesday ET, posting on Truth Social that he paused the 50% tariffs for three days “based on the fact that Canada and the U.S.A., subject to the finalization of documents, have a DEAL!” (https://apnews.com/article/tariffs-trump-canada-usmca-trade-aae597c22617bec7a99f670a2c787ef9). Same timing: CBC (https://www.cbc.ca/news/politics/new-trump-tariffs-deadline-9.7311125), NBC (18 Aug 2026 10:22–11:29 p.m. EDT; https://www.nbcnews.com/business/economy/trump-pauses-canada-tariffs-rcna593176). White House presidential action “Temporary Suspension of Additional Duties…” amends Proclamations 11046/11047/11048 by replacing Annex II effective date “August 19, 2026” with “August 22, 2026,” so additional duties take effect 12:01 a.m. eastern 22 Aug 2026 (https://www.whitehouse.gov/presidential-actions/2026/08/temporary-suspension-of-additional-duties-to-offset-canadian-discrimination-against-the-commerce-of-the-united-states-with-respect-to-alcoholic-beverages-dairy-and-motor-vehicles/). The instrument is a US Section 338 suspend/amend proclamation; the announcement is not a silent unilateral pause. Carney (PMO/CBS): substantial progress, work remaining; US “has agreed to postpone” Section 338 “until end of day, August 21”; Canada confirmed the three-day delay but did not immediately confirm White House claims that Canada committed to remove alcohol/dairy/motor-vehicle measures. Delay of 338, not cancellation, and not a 232-only bargain that left the 19 Aug clock in force. Unsigned papers and a live 22 Aug clock do not restore the old outcome line. No CBP live entry feed in this session.

### C-il-knesset-26-likud

- Problem: P-il-knesset-26
- Forecast: 2026-08-18
- Owner: claim-worker
- Claim: Likud will win the most seats of any single list in Israel's 26th Knesset election (largest party), even if it does not lead the largest bloc.
- Justification: First forecast; no prior claim. Overlay consulted: `agents/claim-worker.md`, `skills/predict/SKILL.md`, `CONTEXT.md` (largest-party claim ≠ government; forecast day is today, resolution day is 2026-10-27). `references/discovery.md` not applied. No `graph-vault/`. No overlay method yet for Israeli house effects; live polls used. Midgam/HaHadashot 12 (17 Aug, n=501): Yashar 24, Likud 22, Together 15. Kantar/Kan 11 (16 Aug, n=1,103, MoE 2.9%): Likud 23, Yashar 23, Together 14; anti-Netanyahu Jewish lists 57, coalition 52, Arab lists 11 — short of 61 without Arab parties (Haaretz). Maariv/Lazar (12–13 Aug, n=506, MoE 4.4%): Yashar 23, Likud 21; Eisenkot 49% vs Netanyahu 38% as preferred PM (Jerusalem Post). Direct Polls/i24NEWS (18 Aug, n=522, ±4.2pp): Likud 31, Yashar 24; coalition 58 / opposition 52 / Arab 10. Channel 14 (18 Aug, JFeed): Likud 32, Yashar 24. The Channel 12/13/Kan cluster has a 0–2 seat Yashar edge or a tie, inside sampling and 3.25% threshold allocation error; the Direct Polls/Channel 14 house still has a Likud plurality. Base rate: Likud has been the largest list in every completed election since 2009, including cycles it lost the bloc. Yashar is a new generals’ list absorbing a Bennett fade; 70 days of campaign usually re-concentrates the right on Likud more than it crowns a first-time largest party. Wartime incumbency, Gaza, and Iran are already in the numbers and two-sided (hostage/ceasefire boosts vs later Iran-war drawdown); a security spike through 27 Oct is more likely to protect an incumbent plurality than to mint a new one. 2022 Likud miss vs averages was small, so Midgam’s 22 is not a floor; a true 24–26 Likud vs 22–24 Yashar still makes Likud largest while losing the bloc. Falsifiers for later ticks: consecutive Midgam+Lazar+Kantar waves with Yashar ≥+4 and Likud stuck ≤22; a center merger that reallocates without shrinking a Yashar plurality. Tools: WebSearch/WebFetch of Haaretz, i24NEWS, JPost, JFeed, Wikipedia poll table.
