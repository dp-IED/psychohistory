# Structural analog cards

How the plugin understands the past so the next forecast is not a news scrape or a weight recall. Procedure for discover, predict, and reflect. Decision: ADR 0013.

The past is a **library of mechanisms**, not a set of already-answered questions to grade. Cards live in `references/` (usually `references/cases/<class-slug>.md`). They are overlay memory. They are not ledger problems.

## Envelope

Keep this shape. Do not add a global type system. Class names are discovered from use; merge later when two names are the same mechanism.

```text
# Class: <discovered name>

- Instantiations: <P-ids>; <named historical episodes + URLs>
- Mechanism: actors, incentives, constraints, clock
- Base rate: what this class usually does
- Disanalogy: why a live instance might not belong
- Falsifiers: what would kill this analog
```

## Discover

On analog/base-rate problems, Motivation must name the **structural class** this live question instantiates and why understanding that class is the point. Prefer a class that already has a card when one exists. Do not open the historical episode as a problem.

## Predict (claim worker)

1. Read any matching card in `references/` / `references/cases/` before live news.
2. Put a **Structure** block in every Justification, even for news-now:
   - Class (card name, Motivation’s name, or `none`)
   - Mechanism in one short pass
   - Base rate vs this instance
   - Disanalogy
   - Falsifiers
   - Sources for historical episodes actually fetched this tick (not “I remember”)
3. If news and analog disagree, say which you followed and why.

## Reflect

Grade the Structure block with the claim series. The prize is still the earliest matching claim; a last-day scrape that ignored a good analog is a miss of method.

Then grow or cull cards:

- Early hit that used the analog → write or deepen the card. When deepening, pull **several** past instantiations of the class from public sources and state the mechanism and base rate. That is allowed historical work. Do not mint dated claims for those episodes.
- Miss caused by a bad analog → rewrite or delete the card. Do not keep a false mechanism.
- No card existed and Structure was empty or parametric → add a card if the series showed a transferable class; otherwise note that nothing earned a file.

Do not write `graph-vault/`.
