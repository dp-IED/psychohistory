---
type: entity
tags: [entity, mena, geopolitics, iran, shia]
domain: global
kind: state
title: "Iran"
slug: iran
date_start: 1979-04-01
date_end: null
pit_cutoff: 2026-06-21
status: active
---

# Iran

## Summary

Iran is a Shia theocratic state established after the 1979 Islamic Revolution. Until February 2026, it was governed by Supreme Leader Ali Khamenei (1989-2026) and an elected president. Iran was the leading Shia power in the Middle East and the primary regional rival of Saudi Arabia and Israel, projecting influence through its "Axis of Resistance" network (Hezbollah, Hamas, Houthis, Shia militias in Iraq and Syria).

Iran's nuclear program reached near-weapons-grade enrichment by 2025, triggering the June 2025 Twelve-Day War with the US and Israel — a devastating bombing campaign that destroyed Iran's enrichment infrastructure at Fordow, Natanz, and Isfahan. Khamenei was assassinated on February 28, 2026 (Operation Epic Fury). On June 14, 2026, Iran signed a permanent peace deal with the US, mediated by Pakistan, ending the state of hostilities that began with the February 2026 US/Israeli strikes and subsequent Iranian retaliation.

## Current Status (June 2026)

- **Supreme Leader**: Vacant since Khamenei's assassination. Succession contest between Mojtaba Khamenei (IRGC-backed, 66.45% market probability), Alireza Arafi (~1.7%), Abbas Araghchi (~4.8%). The succession outcome is the single most consequential uncertainty for 2026-2027.
- **President**: Mohammad Mokhber (acting, elevated from VP after Pezeshkian's removal during war)
- **Nuclear program**: Enrichment capacity physically destroyed (June 2025). Status of formal renunciation unresolved — the "$9.9M end enrichment by June 30" market (vault p_yes=0.09, run #52) tests whether Iran makes a formal commitment in the immediate post-peace-deal window. See [[domains/iran/concepts/iran-nuclear-enrichment-post-war/_concept]].
- **Oil exports**: Blocked by US naval blockade imposed during the Twelve-Day War. Blockade-lifting market at 92% YES by June 30. Hormuz traffic normalization at vault p_yes=0.58 by July 15 (run #49).
- **Diplomatic status**: Peace deal signed June 14. Next meeting with US expected by July 31 (vault p_yes=0.89, run #46). Meeting on Iranian soil near-structural NO (vault p_yes=0.008, run #51).

## Significance for Forecasting

- **Leadership succession as master variable**: Who controls Iran post-Khamenei determines nuclear posture, proxy reconstitution speed, and peace deal durability. The Khamenei assassination was the most unpredictable event in recent Iranian history — the vault treats Supreme Leader mortality as a domain-independent variable.
- **Post-war state trajectory**: Iran's reconstitution capacity in the post-war period (proxy network, nuclear program, economic recovery from sanctions) is the key medium-term variable (2026-2030).
- **US-Iran relationship arc**: The fastest enemy-to-peace transition in modern history (~3.5 months from open war to permanent peace deal) creates unusual forecasting conditions — the post-war diplomatic track is unprecedented and lacks historical analogues.
- **Regional power shift**: Iran's military degradation and diplomatic isolation reversal (via peace deal) reshape the entire MENA power balance. Saudi Arabia, Israel, the Gulf states, and Turkey all adjust their postures.

## Related Entities

- [[domains/iran/entities/ali-khamenei]] — Assassinated Supreme Leader
- [[domains/iran/entities/mojtaba-khamenei]] — Succession frontrunner
- [[domains/iran/entities/masoud-pezeshkian]] — Former President (2024-2026)
- [[domains/iran/entities/islamic-revolutionary-guard-corps-irgc]] — Parallel military; controls missiles, nuclear security, proxy network
- [[domains/mena/entities/mustafa-hijri]] — Kurdish opposition candidate in head-of-state market

## Related Threads

- [[domains/iran/threads/us-iran-post-war-diplomacy/_thread]] — Post-peace-deal diplomatic track (RESOLVED YES Jun 14)
- [[domains/iran/threads/iran-protests-2022/_thread]] — Historical protest movement
- [[domains/iran/threads/iran-internet-restrictions/_thread]] — State internet control
- [[domains/mena/threads/iran-israel-escalation/_thread]] — Shadow war to direct conflict
- [[domains/mena/threads/syria-post-assad-transition/_thread]] — Iran's proxy withdrawal from Syria

## Related Concepts

- [[domains/iran/concepts/iran-nuclear-enrichment-post-war/_concept]] — Post-war enrichment status
- [[domains/global/concepts/elderly-leader-mortality-risk/_concept]] — Previously applied to Khamenei (87); now succession context
- [[domains/global/concepts/peace-broker-diplomatic-playbook/_concept]] — Trump's diplomatic methodology that produced the peace deal
- [[domains/global/concepts/short-window-expiration-cluster/_concept]] — June 30 cluster with Iran enrichment as member

## Active Forecasts

| Forecast | p_yes | Volume | Run |
|----------|-------|--------|-----|
| US-Iran meeting by July 31 | 0.89 | $994K | [[runs/20260618-us-iran-diplomatic-meeting-july31]] |
| Hormuz traffic normal by July 15 | 0.58 | $996K | [[runs/20260618-hormuz-traffic-normal]] |
| Iran ends enrichment by June 30 | 0.09 | $9.9M | [[runs/20260619-iran-enrichment-june30]] |
| Next US-Iran meeting in Iran | 0.008 | $995K | [[runs/20260619-us-iran-meeting-in-iran]] |
| Mustafa Hijri head of state end 2026 | 0.002 | $99K | [[runs/20260618-hijri-iran-head-of-state]] |

## Key Events

- **2025 Jun 13-24**: Twelve-Day War — US/Israeli strikes destroy Iran's nuclear facilities; ceasefire
- **2026 Feb 28**: Khamenei assassinated (Operation Epic Fury)
- **2026 Jun 14**: US-Iran permanent peace deal signed (Pakistan-mediated)
- **2026 Jun 30**: Enrichment agreement deadline (predicted: NO)
- **2026 Jul 15**: Hormuz traffic normalization deadline
- **2026 Jul 31**: US-Iran diplomatic meeting deadline

## See Also

- [[domains/iran/_domain]] — Iran domain navigation
- [[domains/mena/_domain]] — MENA domain navigation
- [[domains/mena/entities/saudi-arabia]] — Regional rival
- [[events/us-iran-peace-deal-june-2026]] — Peace deal event file
