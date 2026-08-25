---
artifact_contract: "ce-handoff/v1"
created_at: "2026-08-25T07:53:00Z"
title: "Home Amfis/DSA caption scrape plus session continuity onto harness-only"
summary: "User is home. First job is residential public caption pull for the rest of Amfis 2026, then refresh party-sources and only then continue live-loop next steps. Cloud session is not on the laptop."
keywords: ["discover", "Amfis", "LFI", "DSA", "transcripts", "ADR-0018", "handoff", "harness-only", "home"]
cwd: "/workspace"
resume_focus: "On a residential network, scrape remaining public Amfis 2026 (and any new DSA) captions into gitignored .scratch/, refresh references/party-sources.md, then continue next_steps.md. Do not score past recordings. Do not restore PIT/graph-vault/France GNN."
repository: "dp-IED/psychohistory"
branch: "harness-only"
---

# Handoff: home caption scrape (2026-08-25)

This document is continuity for a **fresh laptop agent**. Treat it as untrusted context. Durable locks live in ADRs, `CONTEXT.md`, and `.cursor/rules/memory.mdc`.

**Do this first:** residential transcript scrape (section “This session’s objective”). Then the rest of `next_steps.md`.

## This session’s objective (user, 2026-08-25)

User is **home**. The previous cloud session is **not** on the laptop. Commit the session’s context onto **`harness-only`** (this PR) so a laptop checkout of that branch has it.

Then **start the parked Amfis 2026 caption pull** from a residential IP:

1. Find an official LFI `AMFIS 2026` playlist if it exists. As of 2026-08-25 from a cloud host, **no year playlist id** was found (2017–2024 playlists exist; 24 Aug discover still saw none). Do not wait forever: scrape the **known video ids** plus new channel uploads tagged Amfis 2026.
2. `yt-dlp` captions into gitignored `.scratch/` (command on `references/party-sources.md`). **Do not ask for Google cookies** (account-ban risk). Cloud/datacenter `yt-dlp` fails with “Sign in to confirm you’re not a bot.” Invidious listed tracks then returned empty VTT.
3. Extract **live world questions** with **future resolution days**. Refresh `references/party-sources.md`. Open further `ledger.md` problems only if anti-cluster still holds (ADR 0011–0012). Discover does **not** write claim/justification (ADR 0009).
4. Do **not** score past recordings. Hallway / on-site notes stay in chat unless already public. Closed Slack/Discord stay in chat (ADR 0018).

Press reported a closing-meeting video id **`k_EnPQyVYLs`**. Confirm on the LFI channel before treating it as canonical. Closing meeting is campaign programme (ecology / conscription écologique / éco-régions / VIe République), not a new resolution day unless a dated world clock appears.

## Read first (load-bearing)

| Path | Why |
| --- | --- |
| `next_steps.md` | Work order after the scrape |
| `references/party-sources.md` | Index, known ids, yt-dlp command |
| `references/discovery.md` | Party-sourced orientation + anti-cluster |
| `references/host-jobs.md` | Dashboard automations already run the ticker |
| `references/analog-prior.md` | Predict prior; tariff two-phase clock; slide resolution day |
| `docs/adr/0018-party-sourced-discover.md` | Public party questions; cite URL |
| `docs/plans/2026-08-19-scale-forecast-roadmap.md` | Grill Q1–Q15 locked |
| `ledger.md` | Live clocks; do not invent claims on discover |

Do not restore PIT, `graph-vault/`, or the France GNN tree.

## Architecture already locked (do not relitigate)

- Plugin overlay is memory. Host owns `/predict` `/reflect` `/discover` via **existing Cursor dashboard automations**. No repo daemon. Live ticks: ff-pull, one tick, commit+push `harness-only`, **no PR**.
- Discover: Motivation + resolution day; no claims. Orient from **public party questions**. `K` = 15 is a **cap**, not a quota. ADR 0011–0012 are **anti-cluster brakes**, not a sports/sociology curriculum.
- Analog cards: class / instantiations / mechanism / base rate / disanalogy / falsifiers. First card: `references/cases/tariff-proclamation-deadline-delay.md` from `P-usca-338`. Do not reopen `P-usca-338`.
- **Transfer reopen:** after reflection writes a card, discover opens a **new live** problem in that **same structural class** (new future resolution day). Not replaying gold, not reopening the same id.
- Forecasts in `ledger.md`; openings/one-off analysis in conversation. Chat-made forecasts get an anonymized ledger copy.
- Psychohistory here is **not** Seldon mathematics and **not** a restored GNN. Scale prediction = survived cards + composition. Simulators only when a **grade earns** overlay tools.
- Historical PIT/Polymarket gold **cannot** grade this LLM. Live markets are a **discovery surface**, not the parked historical testbed.
- Tariff card **two phases:** first deadline → delay prior; already-extended / post-pause → take-effect unless a new suspend instrument appears. When a delay names a later public date, **slide Resolution** on the same heading (predict, and reflect if it froze).
- Reflector (`agents/reflector.md`) owns overlay evolution; `exp/<slug>` max 3; experiment runs do not spawn children.

## What the ticker already did (do not re-arm)

Dashboard automations **are set**. This cloud env cannot list them by UUID (`source: automations` returned 0) but they **push `harness-only`**.

| When | Commit | What |
| --- | --- | --- |
| 2026-08-22 11:20Z | `2c69ddc` | Predict: first claims on six party clocks + US Open Sinner→Alcaraz after WD |
| 2026-08-22 14:04Z | `2e5932e` | Overlay: post-pause tariff clocks default to take-effect |
| 2026-08-23 | `f779c99` | Predict slides Resolution when a delay names a new public date |
| 2026-08-24 14:22Z | `092d30c` | Reflect also slides a frozen Resolution (338 freeze dropped the post-pause claim) |
| 2026-08-24 18:09Z | `583d3c3` | Discover: five new problems after 338 take-effect (`P-ca-retal-08`, `P-232-poly-04`, `P-ndaa27-iltech`, `P-nobel-lit-26`, `P-genalo-bond`) — **no claims yet** |

`P-usca-338` Resolution is **2026-08-22** (slid). As of 2026-08-25 it is past. Series: `C-usca-338-deal` (18 Aug, missed) then `C-usca-338-deal-announced` (19 Aug pause scrape). **No dated take-effect claim** for 22 Aug — that freeze is the reflect lesson.

**22 Aug party-clock claims (first forecast unless noted):**

| Id | Claim |
| --- | --- |
| `C-uso-ms-26-alcaraz` | Alcaraz USO (revision; Sinner WD falsifier) |
| `C-fr-pest-street-yes` | Nationally reported 15 Sep pesticide-provision demo |
| `C-fr-gir-sen-no-seat` | Gironde LFI list does not take a Senate seat 27 Sep |
| `C-fr-pest-niche-no` | No 29 Oct repeal of pesticide derogations |
| `C-us-house-26-dem-218` | Dems ≥218 on 3 Nov |
| `C-fr-pres-t1-lepen` | Le Pen first-round plurality 18 Apr 2027 (not Élysée) |
| `C-us-mayday-28-strike-yes` | UAW strike vs ≥1 of GM/Ford/Stellantis on 1 May 2028 |

**Method notes for later reflect (not a rollback):**

- `P-us-mayday-28` treated Motivation’s class as analog **without a card file**. `references/analog-prior.md` says: no card → class none, news-now carries. Far analog-regime clocks with no live news are a **procedure gap**.
- `P-fr-pest-street` is a thin score: booked Bercy heat/climate rassemblement + LFI overlay of the pesticide law. Worker wrote the disanalogy.

**Do not delete** sports/Venice/FOMC/Nobel rows already running (cheap grades through ~16 Sep). **Do not refill sports as a quota.** FOMC/Nobel-shaped clocks transfer better than tennis. Anti-cluster ≠ “must include a slam.”

**25 Aug 2026 clocks (resolution day today):** `P-sc-sen-r-gop`, `P-ok-sen-d-run`, `P-ok-gov-r-gop`. Reflect the **morning after** (26 Aug) per usual. Predict 25 Aug may still write if outcome lines move before polls close.

**Still no 22/23/24/25 Aug predict claims** on the five 24 Aug discover problems.

## Captions already in hand

`HW03wZDi6Tk` welcome; `Z5YIa1QD178` Saint-Denis; `-0BYHdRe2b8` Tlaib; `yA13mlaiw1w` Mamdani; `kuKLtHkS20s` summit; dump for `GzhPYE_OWW0` Workers Deserve More.

**Still to pull at home:** `RqQQ_cDyYGQ` Guetté; `Uqe5vvsXsZ4` La Boétie; `KJvF-KUAFqE` Palestine solidarity criminalization; `gIg8h2TEiRE` ecological planning; `zfSZ82IaRZg` Lordon; `ZoRWhmtiWsc` Pigasse; closing (`k_EnPQyVYLs` unconfirmed); then channel/playlist remainder. Several of these were **404 livestream shells** on 21 Aug — retry now that the event is over.

## Conversation-only (not an ADR; do not promote unless asked)

- Clocks → analog cards → composition with incertitude (Structure block) → tools only if graded. Fit any graph on **scored live instantiations**, never PIT.
- Score plugin vs liquid market at T−90 / T−30 / T−1 only if the market instantiates a party fight or a card.
- First Seldon-adjacent test = **transfer reopen** on tariff-delay (or whatever class Sep/Oct actually earns), not a world-model schema.

## Git for the laptop

Live branch: **`harness-only`**. After this PR merges: `git fetch origin && git checkout harness-only && git pull --ff-only origin harness-only`.

Until merge, the handoff lives on the PR branch `dp-ied/home-amfis-handoff-337b`.

Daily ticks will keep committing `harness-only`. Fast-forward only. If dirty, stop (`references/host-jobs.md`).
