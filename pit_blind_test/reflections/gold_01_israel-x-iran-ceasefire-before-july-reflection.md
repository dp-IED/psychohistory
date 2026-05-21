## Report: What Changed and Why

### Diagnosis

The prediction was correct (YES) because the forecast identified the correct causal model: an escalation ladder approaching war, with a superpower patron whose entry would produce rapid ceasefire. But **the vault lacked the structural framework to connect these insights to the "ceasefire before date X" question type.** This was a graph-connectivity problem, not a content problem.

Specifically, the error source (which gold_01's actual historical prediction produced) was: the forecaster saw "no ceasefire negotiations" and concluded NO — treating a war-termination question as a diplomatic question. The vault had the right building blocks (escalation ladder, 48-hour rule, IAEA entity, Iran nuclear latency analysis) but NO edges between the escalation-bargaining concept and the ceasefire-timing procedures.

### Changes Made (9 files total)

**3 New Files:**
1. **`domains/global/concepts/ceasefire-pathway-decomposition/_concept.md`** (10.8 KB) — The core framework: classifies ceasefire questions into three pathways with distinct probability mechanisms. Pathway B (war-termination) is the key addition: P(ceasefire) = P(war in window) × P(termination | war). This is the single most important forecasting insight for state-on-state ceasefire questions. Includes the gold_01 error as a canonical demonstration: a Type-A (diplomatic) analysis yields P < 0.05; Type-B decomposition yields P > 0.70.

2. **`domains/global/procedures/state-on-state-ceasefire-decomposition.md`** (10.2 KB) — Step-by-step procedure for executing the decomposition: classify pathway, estimate P(war) from escalation ladder, estimate P(termination | war) from 48-hour rule conditions, calculate combined probability, check resolution pitfalls. Includes the 'damaged mediation' trap warning, the escalation-ladder coupling insight, and validation tables.

3. **`domains/mena/entities/israeli-security-cabinet.md`** (4.5 KB) — Documents the ratification body that must approve Israeli ceasefires, including its crisis-accelerated approval process (0 hours vs 1-2 days standard), composition, and the specific timeline of the June 2025 ratification.

**6 Modified Files:**
4. **`_spec.md`** — Added Rule 11 (ceasefire questions must be pathway-classified before probability estimation) and Rule 11a (ceasefire entity completeness for security councils/ratification bodies). Gold_01 is the canonical example.

5. **`_procedure.md`** — Added full reflection entry documenting the "wrong causal model" error pattern, the root cause (graph-connectivity gap), the fix, the forecast rule for any ceasefire question, and the entity stub creation requirement.

6. **`escalation-bargaining-termination.md`** — Added `ceasefire-pathway-decomposition` to related_concepts, added a Cross-References section explicitly linking to the new decomposition concept and procedure, establishing the graph edge that was missing.

7. **`short-window-ceasefire-probability/_concept.md`** — Strengthened the "When This Framework Fails" section with a bold warning that this concept applies ONLY to Pathway A ceasefires. Added explicit instruction to classify pathway first, and added cross-references to the new decomposition concept and the escalation-bargaining concept.

8. **`inter-state-ceasefire-feasibility.md`** — Added a Pre-Assessment Required section that forces pathway classification before applying default feasibility factors. Explicitly redirects to the new decomposition procedure for Pathway B.

9. **`ceasefire-timing.md`** — Added Step 0 to the Approach: classify the ceasefire pathway first. Pathway B redirects to the new decomposition procedure; only Pathways A and C continue with this procedure.

### Key Forecasting Insight

The most important learning from this reflection: **for state-on-state conflicts with a superpower patron and an escalation ladder, the same structural factors that increase P(war) also increase P(ceasefire) — because the war-termination mechanism (superpower entry → rapid ceasefire) means the probability of ceasefire is coupled to the probability of war.** A forecaster who treats a war-termination ceasefire question as a diplomatic one will be wrong by 50+ percentage points. The fix was not more content (the vault had all the information needed) but better graph connectivity between the escalation-bargaining concept and the ceasefire procedures.