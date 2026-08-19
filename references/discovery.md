# Discovery tick

A separate host job (`/discover`). Not predict and not reflect.

Intake: open at most `K` new problems (the knob on `ledger.md`). Each new problem needs **Motivation** and **Resolution** (the **resolution day**). No quality gate; reflection culls later. Open inventory is unbounded. Prefer farther resolution days as the overlay strengthens.

Domain mix: when a tick opens more than one problem, draw from at least three of politics/institutional power, economics/markets/trade, courts/legal, science/tech, conflict/security, sports, culture/entertainment. No single domain may take more than half of the problems opened in that tick. Weigh the domain mix already open (and recently resolved), not just the new batch — prefer an underrepresented domain when motivations are otherwise comparable. This is how a larger `K` still trains a plugin that generalizes instead of one that only gets deeper at trade/geopolitics reasoning (see ADR 0011).

When this tick runs, append problem headings under `## Problems`. Do not add dated claims. **Predict** writes **claim** and **justification** while the problem is live.
