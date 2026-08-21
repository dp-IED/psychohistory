# Scale forecast: grill-with-docs session

**Status:** architecture grill closed (Q15). Implementation is separate.  
**Skill:** Matt Pocock `grill-with-docs` (grilling + domain-modeling).  
**Date:** 2026-08-21

## Settled so far

**Q1 — What would “it worked” look like?**  
All four: dated events, material facts, official machines, and what people treat as obvious.  
**Why:** help **LFI, DSA, and similar parties** with a long fight over institutions and common sense, and a short fight when power is contestable.

**Q2 — What should come out for those parties?**  
Forecasts (dated, public, gradeable) plus a **map of openings**. Not a **playbook**. ADR 0014.

**Q3 — Whose openings?**  
**Multitenant:** whoever is asking this run, not one hardcoded party.

**Q4 / Q5 — Shared vs private vs internationalism.**  
Shared analog cards and typical openings on those cards. Private this-run openings. ADR 0015.

**Q6 — Country names on shared notes.**  
Instantiations keep country and case names plus sources so people can look the case up.

**Q7 — One pile or two.**  
One analog-card pile. Optional typical openings on the same card when earned.

**Q8 / chat vs repo (locked; do not re-ask).**  
One-off analysis and openings: conversation history. Forecasts: repo. Chat-made forecast → anonymized copy into `ledger.md`; wording the user saw stays in chat; optional harness FS if they ask.

## Remaining (training and agent structure)

Closed. Reflector may still change sourcing later on `exp/`.

**Q11 — Risky experiments.**  
**C.** Tiny edits on the live plugin. New agents/tools/how-tos on a git **experiment branch**. The **reflect host job** may autonomously create that branch and start a Cursor automation on it. Plugin markdown instructs; it is not a daemon. Reflector writes evolution; harness still starts daily predict/discover/reflect. ADR 0016.

**Q12 — What an experiment run does.**  
The **reflector decides per experiment**. Record it on the branch.

**Q13 — Branch prefix.**  
**`exp/`** (decided). Example: `exp/claim-critic`. List `exp/*` to track. No sidecar file.

**Q14 — Reflector vs Parent.**  
**B.** Parent starts **reflector** on reflect ticks (`agents/reflector.md`, ADR 0017). At most three `exp/` branches (decided).

**Q15 — Problem sourcing (architecture grill closed).**  
No extra field quota; mixed-field problems are good. Discover orients from **public** party questions (Amfis YouTube, public DSA debates, etc.), cited in Motivation. Score only a live future resolution day. Closed internals stay in chat. Anti-cluster mix (ADR 0011–0012) remains a brake on large `K`, not a curriculum. ADR 0018, `references/discovery.md`.
