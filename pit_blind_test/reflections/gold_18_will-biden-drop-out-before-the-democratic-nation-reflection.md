Report summary.

## Report: What changed and why

### Diagnosis

The question "Biden drops out before the Democratic convention?" is a **deadline-constrained** variant of a withdrawal question. The vault had extensive content on the withdrawal cascade (the incumbent-withdrawal-cascade concept at 306 lines, the Biden entity with vulnerability signals, multiple procedures) but **zero explicit methodology for handling the "before [deadline]" compound probability structure**.

The deadline matters because it transforms the forecast from P(withdrawal) into P(withdrawal before DNC) = P(trigger by effective_deadline) × P(cascade completes in time | trigger). The DNC was August 19, the cascade takes ~24 days, so the effective trigger deadline is ~July 26. If a trigger occurred (it did — the June 27 debate), and 53 days remain to the convention > 24-day cascade, **the constraint is not binding** and P(before DNC) ≈ P(withdrawal | trigger) ≈ 85%+. At a pre-debate cutoff, the scheduled debate itself within the effective window dramatically increased per-period trigger probability.

The error was not assessing whether the deadline constraint was binding — which would have immediately shown it wasn't (post-debate) or that the effective window still contained the debate trigger risk (pre-debate).

### Files changed

1. **`_procedure.md`** — Added "DEADLINE-CONSTRAINED WITHDRAWAL" sub-step to step 16, with a 4-step protocol: identify deadline + cascade clock, compute effective trigger deadline, assess constraint binding, and compound probability calculation. Includes the Biden-DNC canonical example.

2. **`domains/usa/concepts/incumbent-withdrawal-cascade.md`** — Added "Deadline-Constrained Withdrawal Forecasting" section with compound probability model, three scenarios (A: post-trigger, not binding; B: post-trigger, binding; C: pre-trigger), cascade-acceleration factor, canonical case table for Biden-DNC, and an 8-item forecasting checklist.

3. **`_spec.md`** — Added Rule 55 mandating deadline-constrained analysis for ANY "before [deadline]" withdrawal question, with required documentation for deadline date, effective trigger deadline, constraint binding assessment, compound probability calculation, and cascade acceleration factor.

4. **`meta/reflections/_reflection-2026-05-20-per-q18-biden-before-dnc.md`** — Full per-question reflection documenting the error, the compound probability model, what was missing at forecast time, what was created, and the cross-domain transfer failure (legal + ceasefire "before deadline" patterns existed but were not applied to withdrawal).

5. **`_index.md`** — Added reflection entry logging the gap analysis and all fixes.