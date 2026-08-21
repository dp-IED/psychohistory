# Scale forecast: grill-with-docs session

**Status:** in session (not an ADR). One question at a time. Write glossary/ADRs only when a term or trade-off actually settles.  
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

Do not re-ask storage, tenants, Gramsci layers, or “is reflector the daily parent.”

**Q11 — Risky experiments.**  
**C.** Tiny edits on the live plugin. New agents/tools/how-tos on a git **experiment branch**. The **reflect host job** may autonomously create that branch and start a Cursor automation on it. Plugin markdown instructs; it is not a daemon. Reflector writes evolution; harness still starts daily predict/discover/reflect. ADR 0016.

**Spawn bound (locked unless you override):** only a reflect job on the **live** branch starts experiment automations. A run already on an experiment branch does not start children.

Not implemented yet: reflect skill still only commits `harness-only`. Host must actually expose “start automation on branch.”
