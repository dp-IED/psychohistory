## Per-Question Reflection Report: gold_12 (Biden dropout miss)

### 1. Diagnosis: Why was this prediction wrong?

The NO prediction for "Biden drops out of presidential race?" failed due to three distinct biases:

**a) Wrong frame applied:** I applied a persistence frame ("incumbents don't drop out, party unifies behind the nominee") instead of a vulnerability frame. At the time, I saw a unified Democratic party, no visible trigger, and Biden's repeated public statements. I didn't apply the historical comparison to Truman 1952 and LBJ 1968 — both incumbents who looked stable right up until they withdrew.

**b) Overweighted stated intentions:** Biden's public denials and his inner circle's confidence were treated as evidence when they're actually a structural feature of the Stage 0 denial pattern exhibited by all three canonical withdrawers.

**c) Underweighted cumulative trigger probability:** Over a 10-month horizon for an 81-year-old, the probability of at least one trigger event (debate failure, health scare, primary challenge, gaffe cascade) was ~40-55% — not a low-probability tail. I assessed the static state (no trigger today) rather than the dynamic risk (trigger over horizon).

### 2. Vault Improvements Made

| File | Change | Rationale |
|------|--------|-----------|
| **domain/usa/entities/george-clooney.md** (new) | Created entity stub documenting his role as donor/surrogate whose NYT op-ed was the cascade inflection point | George Clooney was referenced as a wikilink from 3+ locations but had no file — a dangling graph node |
| **incumbent-withdrawal-cascade.md** (3 patches) | Fixed broken wikilinks: `[[Nancy Pelosi]]` -> `[[domains/usa/entities/nancy-pelosi\|Nancy Pelosi]]`, `[[George Clooney]]` -> `[[domains/usa/entities/george-clooney\|George Clooney]]` | Violated Spec Rule 36 (no dangling concept/entity references). Concept named key actors but wikilinks didn't resolve |
| **_procedure.md** (2 patches) | Fixed broken `[[George Clooney]]` wikilink in step 16 cascade velocity section + Added new **step 6a** (Content-loaded entity sweep) requiring entity stub verification for all named actors in loaded concept/thread/procedure content | The existing step 6 (named entity sweep) only covers question text. Concept files reference entities that never appear in question text (Pelosi, Clooney, Schumer, etc.) — these were systematically missed |
| **_spec.md** (new Rule 40) | Added Rule 40: "Donor/surrogate defection as mandatory cascade-signal tracking" with timing rule (3-7 day leading indicator before institutional engagement, P(withdrawal within 21 days) > 80% when donor defects with proximate-witness credibility) and entity stub requirements for donor/surrogates | The donor defection pattern was documented in the concept and procedure but had no spec-level mandate. Without this rule, future forecasts could still miss this signal |

### 3. Key Insight from This Miss

The gold_12 error was not just about getting the probability wrong — it was about the **framing error** of treating an incumbent as structurally similar to a non-incumbent. The vault now enforces candidate-type classification (Spec Rule 35, Procedure Step 16) as the FIRST step before any withdrawal analysis. This ensures that no future withdrawal question can accidentally apply a persistence frame to an incumbent without first checking the 6-signal vulnerability inventory from the aging-incumbent early warning procedure.

The specific code-level gap revealed by this reflection is: **named entities in concept files are not captured by the existing question-text entity sweep.** The Pelosi/Clooney entities were referenced in the cascade concept but not in the question "Biden drops out?" — so step 6 missed them entirely. Step 6a closes this gap.