# Discovery tick

A separate host job (`/discover`). Not predict and not reflect.

Intake: open at most `K` new problems (the knob on `ledger.md`). Each new problem needs **Motivation** and **Resolution** (the **resolution day**). No quality gate; reflection culls later. Open inventory is unbounded. Prefer farther resolution days as the overlay strengthens.

Domain mix: when a tick opens more than one problem, draw from at least three of politics/institutional power, economics/markets/trade, courts/legal, science/tech, conflict/security, sports, culture/entertainment. No single domain may take more than half of the problems opened in that tick. Weigh the domain mix already open (and recently resolved), not just the new batch — prefer an underrepresented domain when motivations are otherwise comparable. This is how a larger `K` still trains a plugin that generalizes instead of one that only gets deeper at trade/geopolitics reasoning (see ADR 0011).

Horizon and evidence mix: when a tick opens more than one problem, include at least one **near** resolution day (about ≤30 days) and at least one **far** (about ≥90 days). Include at least one **news-now** problem and at least one whose motivation is **analog/base-rate** (a past class of cases, a structural rate, or a method this plugin already claimed — still scored in the future). Analog-regime Motivation names the **structural class** (`references/structure.md`). After reflection, prefer a new live problem in a class the overlay just claimed to learn, so analog-prior can be falsified (`references/analog-prior.md`). The first graded class is **tariff-proclamation deadline delay** (`references/cases/tariff-proclamation-deadline-delay.md`): open a *new* live administrative tariff/trade clock in that class when one exists (including a later resume of the same 338 annex), and do not reopen P-usca-338. Do not open already-resolved history as problems (ADR 0012, ADR 0013).

When this tick runs, append problem headings under `## Problems`. Do not add dated claims. **Predict** writes **claim** and **justification** while the problem is live.

## Party-sourced orientation

Do not treat domain buckets as a curriculum. One problem may mix economy, institutions, and what people treat as obvious.

Orient new problems from **public** questions similar parties ask: LFI Amfis (YouTube and transcripts), published congresses and platforms, public DSA debates and similar. Put the source URL in **Motivation**. That recording is not the thing we score. Open a **live** question with a **resolution day** after today that the discussion points at (or a live instance of the same class). Analog-regime rows still name the **structural class**.

**Starter index:** `references/party-sources.md` (channel/playlist URLs, extracted questions, suggested future resolution days). Extend that file; do not replace this section. Amfis and DSA are examples of the method, not the only tenants.

Closed internal docs stay in conversation. They are not the shared discover scrape.

Anti-cluster (ADR 0011, 0012) still applies when this tick opens more than one problem, so a large `K` does not collapse onto one headline. Prefer a mixed-field row over splitting one party question into fake pure buckets.
