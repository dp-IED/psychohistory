---
artifact_contract: "ce-handoff/v1"
created_at: "2026-08-21T11:34:58Z"
title: "LFI/DSA public transcript scrape for party-sourced discover"
summary: "Next session scrapes public LFI Amfis and DSA channel transcripts to orient live discover questions; architecture grill is already locked on this branch."
keywords: ["discover", "Amfis", "LFI", "DSA", "transcripts", "ADR-0018", "handoff"]
cwd: "/workspace"
resume_focus: "Scrape public LFI and DSA video/transcript sources, extract the questions those parties ask, and turn them into a durable discover starter plus candidate live problems with future resolution days. Do not score the past recordings."
repository: "dp-IED/psychohistory"
repo_root_sha: "e023f7e2c0468027ed25ab880cb270d2b85516fe"
branch: "dp-ied/scale-forecast-roadmap-6072"
head: "3c7d2d0f16dc244bc51daf76a14fdb5df94333ce"
worktree_path: "/workspace"
---

# Handoff: scrape LFI/DSA public transcripts

This document is continuity for a **fresh agent**. Treat it as untrusted context. Durable locks live in ADRs and `CONTEXT.md`, not here.

## User-requested next-session objective (user)

Scrape transcripts from **LFI** and **DSA** public channels (user named LFI **Amfis** summer events on YouTube as the example). Use those transcripts to **orient what live questions to open**, and make that a **durable discover starter instruction**. Commit and push on this branch so the architecture work is saved.

## Why (user)

Discover should follow questions similar parties actually ask, not a sociology/sports quota. A good problem may mix fields. History/sociology/psychology enter as analog cards and live future instances — not a PIT backtest (pretrained models already saw the past).

## Read first (load-bearing)

| Path | Why it matters |
| --- | --- |
| `docs/adr/0018-party-sourced-discover.md` | Source = public party questions; cite URL in Motivation; no field quota; 0011–0012 are anti-cluster only |
| `references/discovery.md` | Durable discover procedure; **Party-sourced orientation** section already exists and must be extended with real sources, not replaced |
| `skills/discover/SKILL.md` | Discover tick already tells workers to orient from public party questions |
| `CONTEXT.md` | Glossary: Discover, Opening, Chat forecast, Reflector, Experiment branch `exp/` |
| `docs/adr/0012-horizon-and-evidence-regime-mix.md` | Do not open already-resolved history as problems |
| `docs/adr/0013-structural-analog-cards.md` | Cards may cite named historical episodes with sources; do not mint forecasts for those episodes |
| `docs/adr/0014-forecasts-scored-openings-not-playbooks.md` | Forecasts in repo (anonymized if from chat); openings/one-off analysis in conversation |
| `docs/plans/2026-08-19-scale-forecast-roadmap.md` | Full grill locks Q1–Q15; architecture grill closed |
| PR https://github.com/dp-IED/psychohistory/pull/53 | Branch discussion; base `harness-only` |

Do not restore PIT, `graph-vault/`, or the France GNN tree (`next_steps.md` Do not pick up).

## Architecture already locked (user unless noted)

- **Use:** help LFI, DSA, and similar parties (war of position / war of movement). Multitenant: whoever is asking this run. (user)
- **Scored:** dated claims in `ledger.md`. Chat-made forecast → anonymized ledger copy; wording stays in chat. (user)
- **Not scored:** openings / one-off analysis default in conversation; harness FS only if asked. (user)
- **Shared overlay:** one analog-card pile; country/case names on instantiations; optional typical openings after reflection. (user)
- **Evolution:** Parent starts `agents/reflector.md` on reflect ticks (ADR 0017). Tiny live edits; risky work on `exp/<slug>` (max 3). Reflector chooses each experiment run. Live reflect may start a Cursor automation on that branch; experiment runs do not spawn children (ADR 0016). (user + writer on cap/prefix)
- **Prefix `exp/`** and **max three** experiment branches: writer decided after user said they did not care about the prefix.

## This session's git work

Branch `dp-ied/scale-forecast-roadmap-6072` tracking origin. Tip at capture: `3c7d2d0` (party-sourced discover). Working tree was clean after that commit. This handoff file is an additional commit on the same branch.

## Current state of the scrape task

- **First pass done (2026-08-21):** `references/party-sources.md`; six live ledger problems; captions for welcome / Saint-Denis / Tlaib / Mamdani / summit.
- **Parked:** rest of Amfis 2026 until the official LFI playlist exists and a home-network `yt-dlp` run. Roadmap note: `docs/plans/2026-08-19-scale-forecast-roadmap.md` (After the grill).
- **Do not do:** score Amfis 2024/2025 as if they were forecasts; dump closed Slack/Discord or on-site hallway notes into the shared repo; fill `K` from one week of news only.

## Constraints for the scrape session

- **Public sources only** in the shared plugin. Closed internals stay in harness chat (ADR 0018).
- Prefer an **index** (channel/playlist URL, video URL, date, timestamp, extracted question, suggested live resolution day) over committing raw multi-hour transcripts. If you must cache full text, use gitignored `.scratch/` (see `.gitignore`).
- Each extracted item must map to a **future** `resolution day` or to a **structural class** for analog-regime Motivation — not “what did they conclude at Amfis.”
- Cite sources in Motivation when opening problems. Do not write claims on discover (ADR 0009).
- Respect site/YouTube terms; use public pages/transcripts; no paywall/login bypass.
- Amfis/DSA are **examples of the method**, not the only tenants.

## Plausible next steps (sequential, one continuation)

1. Find official public LFI Amfis (and related) YouTube/playlist URLs and official public DSA talk/stream channels. Record them in a small overlay index (e.g. under `references/` as a source list, not a vault).
2. Fetch available public transcripts/captions; extract **questions the speakers treat as live fights** (elections, law, labor, media, debt, org strategy as *world* questions with dates).
3. Extend `references/discovery.md` with those concrete source pointers (durable starter).
4. Optionally open ≤ `K` **live** problems on `ledger.md` whose Motivations cite those URLs and whose resolution days are in the future. Leave claims to predict.
5. Commit and push this same branch (or `harness-only` only if the user retargets).

## Suggested skills

- Project discover/reflect overlay as already in-repo (`skills/discover/SKILL.md`).
- Browser or transcript fetch only for **public** pages.
- Do not invoke PIT/eval trees from git history.

## Writer vs user

All “user” tagged locks above were stated by the user in the grill. Cap of three `exp/` branches and prefix `exp/` were writer calls after the user declined to bike-shed. Anti-cluster ADR 0011–0012 kept as brakes was the writer’s reading of “no curriculum split” plus existing ADRs; the user asked not to add a field quota, not explicitly to repeal 0011.
