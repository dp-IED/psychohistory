All four tasks complete. Here's the reflection report.

---

## Per-Question Reflection: Q14 — Trump election interference trial doesn't start before November?

### 1. DIAGNOSIS — Why was the prediction correct?

**Prediction: YES (trial did NOT start before November 2024). Actual: YES. CORRECT.**

This was not a freebie — the vault had substantial structured coverage that supported the forecast:

**What helped:**

1. **trump-criminal-cases thread** (`domains/usa/threads/trump-criminal-cases/_thread.md`): Contains the full timeline of the immunity appeal from Chutkan's December 2023 denial through SCOTUS's July 2024 ruling and remand proceedings. The "Delay as a Coordinated Legal-Political Strategy" key dynamic explicitly documents how the immunity appeal consumed the March-November window. This is the single most valuable vault artifact for this question.

2. **judicial-timing-political-deadline concept** (`domains/usa/concepts/judicial-timing-political-deadline.md`): Provides the structural framework — novel constitutional question + interlocutory appeal with automatic stay + SCOTUS cert grant + remand proceedings = ~11 months of delay. The precise timing table (Step: Date, Duration, Notes) shows exactly how March 4 was pushed past November 5.

3. **Jack Smith entity** (`domains/usa/entities/jack-smith.md`): Documents the special counsel's aggressive pre-indictment timeline (indictments by August 2023) and the structural vulnerability of politically sensitive prosecutions that overlap with election cycles.

4. **Tanya Chutkan entity** (`domains/usa/entities/tanya-chutkan.md`): Documents the pre-appointment schedule (March 4 trial date), the immunity denial (Dec 2023), and the critical insight that trial judges are constrained by appellate timelines they cannot control.

5. **US Supreme Court entity** (`domains/usa/entities/us-supreme-court.md`): Documents the cert grant in February 2024 as a de facto delay signal, the standard 5-month cert-to-decision timeline, and the July 1 immunity ruling.

6. **DOJ OLC entity** (`domains/usa/entities/doj-office-of-legal-counsel.md`): Documents the structural two-track system (federal cases mooted by election victory, state cases proceed independently).

**What was missing — vault gaps despite correct prediction:**

The vault's defense-side analysis was excellent. But it had a systematic gap:

1. **No concept for prosecutorial election-year timing constraints**: The DOJ's "60-day rule" and Merrick Garland's cautious posture were an independent factor that made trial before November structurally impossible — separate from Trump's delay strategy. The judicial-timing-political-deadline concept only analyzes defense delay. A forecaster relying solely on vault content would underestimate the robustness of NO by missing the second independent mechanism.

2. **DC Circuit entity missing its Trump immunity role**: The cert-before-judgment procedural anomaly (SCOTUS bypassing the DC Circuit) paradoxically maximized delay by eliminating a faster intermediate appellate step. The DC Circuit entity focused on crypto/agency review and had no documentation of the Trump immunity bypass — a significant gap for legal-timeline forecasting.

3. **Procedure step 17 lacks symmetrical prosecution-constraints analysis**: The procedure for legal timeline dynamics was entirely defense-focused (identify delay mechanisms, calculate appeal timelines, sum delay budget). No symmetrical step asked: "What constraints does the prosecutor face independent of the defense?" This created a one-sided analytical framework.

**Causal chain summary:**

The correct YES prediction relied on this chain:

- Judge Chutkan denied immunity (Dec 2023) → Trump filed interlocutory appeal (automatic stay) → SCOTUS granted cert (Feb 2024) → standard SCOTUS timeline consumed Feb-July 2024 → remand proceedings consumed July-Nov 2024 → **Election passed without trial** (defense-delay mechanism)

- **Independent parallel mechanism**: DOJ's 60-day rule rendered trial after early September 2024 practically impossible regardless of the immunity timeline — Garlands DOJ would not have pushed a trial into the post-Labor Day window even if SCOTUS had ruled faster

Both mechanisms independently produce NO. The combined effect made the outcome near-deterministic.

### 2. IMPROVEMENTS MADE

| File | Change | Rationale |
|------|--------|-----------|
| `domains/usa/concepts/prosecutorial-election-year-timing.md` | **Created** (9,712 bytes) | Documents DOJ 60-day rule, Comey effect, Garland's posture, state exception — the prosecutor-side mirror to judicial-timing-political-deadline. Previously absent from vault entirely. |
| `domains/usa/entities/dc-circuit-court-of-appeals.md` | **Enriched** — Added "Trump immunity bypass" section with timeline table | Documents the cert-before-judgment procedural anomaly and the forecasting implication (SCOTUS bypass maximized timeline consumption). |
| `_procedure.md` step 17 | **Enriched** — Added "Map the prosecutor's timing constraints" sub-step between state/federal distinction and framework application | Creates symmetrical analysis: forecaster must check BOTH defense delay AND prosecutor constraints. References the new concept. |
| `domains/usa/_domain.md` | **Enriched** — Added `prosecutorial-election-year-timing` to subjects list | Integrates the new concept into the Elections domain graph. |
| `_index.md` | **Enriched** — Added May 20 Recent Additions section | Documents the new and modified files for downstream navigation. |

### 3. KEY LESSON

The most important vault lesson from this question: **defense-side delay analysis is necessary but not sufficient for legal-timeline forecasting.** The prosecutor faces independent institutional constraints (DOJ 60-day rule, AG posture, Comey effect) that can make pre-election trial impossible even if the defense's delay strategy fails. A forecaster who only maps the defendant's delay mechanisms will systematically underestimate the probability that a trial won't occur before an election. The correct approach requires **dual-constraint analysis**: defense delay + prosecutor constraint = combined timeline ceiling.

The new concept formalizes this for any future question about federal prosecutions near an election — and the updated procedure step ensures the check cannot be skipped.