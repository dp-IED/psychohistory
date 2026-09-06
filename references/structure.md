# Structural analog cards

How the plugin understands the past so the next forecast is not a news scrape or a weight recall. Procedure for discover, predict, and reflect. Decision: ADR 0013.

The past is a **library of mechanisms**, not a set of already-answered questions to grade. Cards live in `references/` (usually `references/cases/<class-slug>.md`). They are overlay memory. They are not ledger problems.

## Envelope

Keep this shape. Do not add a global type system. Class names are discovered from use; merge later when two names are the same mechanism.

```text
# Class: <discovered name>

- Instantiations: <P-ids>; <named historical episodes + country + URLs, so a later reader can look the case up>
- Mechanism: actors, incentives, constraints, clock
- Base rate: what this class usually does
- Typical openings: <optional; earned on reflection; anonymized, no tenant name>
- Disanalogy: why a live instance might not belong
- Falsifiers: what would kill this analog
```

## Discover

On analog/base-rate problems, Motivation must name the **structural class** this live question instantiates and why understanding that class is the point. Prefer a class that already has a card when one exists. Do not open the historical episode as a problem.

## Predict (claim worker)

1. Read any matching card in `references/` / `references/cases/` before live news. For an administrative tariff or trade clock, that card is `references/cases/tariff-proclamation-deadline-delay.md`. For a delayed majority primary runoff (second round after no one hit the threshold), that card is `references/cases/majority-primary-runoff.md` (named MAGA consolidator outranks a non-consolidator first-round plurality). For a consultative or first-step EU accession-talks referendum in an EEA/fisheries (or resource-periphery) state, that card is `references/cases/eea-fisheries-eu-talks-referendum.md` (capital Yes / coast No; talks≠membership is not a Yes prior). For a junta- or military-transition constitutional yes/no, that card is `references/cases/junta-constitutional-referendum.md` (held vs official-Yes are different clock phases; polling day is not the certified-result date). For a sitting senator (or equivalent statewide incumbent) on a first-past-the-post intra-party primary against a named same-party challenger, that card is `references/cases/incumbent-plurality-primary.md` (stable double-digit incumbent polls outrank age copy and challenger spend; this is not a delayed majority runoff and not an open seat). Then read `references/analog-prior.md`.
2. Put a **Structure** block in every Justification, even for news-now:
   - Class (card name, Motivation’s name, or `none`)
   - Mechanism in one short pass
   - Base rate vs this instance
   - Disanalogy
   - Falsifiers
   - Sources for historical episodes actually fetched this tick (not “I remember”)
3. If a card matches, analog is the default for this **clock phase**. Follow news against that phase only when a listed falsifier has fired this tick (`references/analog-prior.md`). Last-day source-split and remaining bargain gaps are not falsifiers. On a tariff card, a first deadline and an already-extended date are different priors. Name which you followed. If the cheap instrument is a delay to a named next public date, that date is the new **resolution day** of the same problem so the next phase can still be forecast. The claim on that tick is the new phase’s modal, not a recap that the first deadline was paused. `scripts/resolution_clock.py` flags a Claim that already named that next date while Resolution stayed frozen. `scripts/pause_recap.py` flags a live latest Claim that recaps that pause after Resolution already equals the named next date.

## Reflect

Grade the Structure block with the claim series. Grade the **system** that was in force: cards named, skills followed, any `scripts/` or `exp/` tool predict was to use. The prize is still the earliest matching claim; a last-day scrape that ignored a good analog is a miss of method. A tool that pointed at the wrong clock phase is a system miss.

Then grow or cull cards **and** tools:

- Early hit that used the analog → write or deepen the card. When deepening, pull **several** past instantiations of the class from public sources and state the mechanism and base rate. If the series earned it, add or rewrite **typical openings** (no tenant name). That is allowed historical work. Do not mint dated claims for those episodes.
- Miss caused by a bad analog → rewrite or delete the card. Do not keep a false mechanism.
- Load-bearing script or model that missed → rewrite or delete it the same way. Do not leave a false tool loaded.
- No card existed and Structure was empty or parametric → add a card if the series showed a transferable class; otherwise note that nothing earned a file.

Do not write `graph-vault/`.
