# Structural analog cards are how the plugin understands the past

Forecasting improves when the plugin has an explicit, loadable model of **how a class of past cases worked**, not when the model recites news or parametric memory. ADR 0012 already mixes analog-regime live problems into discover. That is not yet understanding. Understanding is a constrained intermediate representation of mechanism, consulted on predict, graded on reflection, rewritten when it fails.

**Loop**

1. **Discover** names a **structural class** on analog-regime problems (in Motivation). The live problem is an instance we will score. The class is what we are trying to understand.
2. **Predict** must reason over overlay analog cards first (`references/structure.md`). Every justification includes a **Structure** block (minimal envelope below). Historical episodes used as evidence are fetched and cited, not recalled from weights. News-now problems still fill the block: either a class applies or the worker says none and why.
3. **Reflect** grades the Structure block with the claim series. If the mechanism earned the early hit (or explains the miss), Parent writes or rewrites a **case card** under `references/` — multiple past instantiations, mechanism, base rate, falsifiers. If the analog was load-bearing and wrong, cull or rewrite the card. Deepening a class from public history while writing the card is allowed. Scoring that history as if it were a forecast is not (ADR 0002, 0003).

**Minimal envelope** (audit/logging, not a global ontology). Discover class names from use; merge cards later when two names are the same mechanism. Do not grow a taxonomy in advance.

- Class (discovered name)
- Instantiations (ledger problem ids, plus named historical episodes with sources)
- Mechanism (actors, incentives, constraints, clock)
- Base rate (what the class usually does)
- Disanalogy (why the live instance might not belong)
- Falsifiers (what would kill the analog)

**Why this beats “read more history.”** Parametric history is shallow, uncited, and not culled. A card is a representation the next worker must use and the next reflection can delete. Transfer reopen (ADR 0012) then falsifies the card on a *new live* instance.

No quality gate to open a problem (ADR 0005). Overlay writes stay Parent’s (ADR 0007). No `graph-vault/`, no PIT epoch.
