# Forecasting strategy

Obsidian-side **memory** for the synthesis agent. **Machine knobs** (`blind_spot_checks`, `max_steps`, `convergence_epsilon`, `shrinkage`) live in `.harness/policy.md` YAML; this file is **operating procedure + postmortem routing**.

This harness runs a two-phase loop:

## Phase 1 — Guess

Given a question and cutoff date, the synthesis agent:
1. Researches via PIT-filtered search (only information available ≤ cutoff)
2. Checks past analogues for similar resolved questions
3. Produces a `p_yes` forecast with reasoning

Tools available at synthesis time:
- **PIT search**: `python -m harness.tools.pit_search 'query' --cutoff YYYY-MM-DD`
- **Dataview**: `python -m harness.tools.dataview_query --category X --horizon N`
- **Policy**: `cat .harness/policy.md` (canonical YAML + procedure; root `policy.md` is a pointer)

### Strategic priorities

1. **Fixture detection beats upstream labels**: Club/nation + calendar date + win/lose/draw ⇒ **sports workflow** always. Do not wait for `market_family=sports`.
2. **Prose triad is non-negotiable**: Calendar, league priors, availability for **every** fixture answer — even when telemetry shows sports checks skipped or YAML stayed `[]`.
3. **YAML profile for eval**: Default empty `blind_spot_checks` preserves step budget on mixed batches. **Sports-heavy calibration runs**: temporarily paste the three sports IDs into `.harness/policy.md` YAML so dashboards show **fired** (then revert). Long-term: family-conditional planner (delegated).
4. **Break the generic attractor**: Thin evidence + shrinkage pulls toward ~0.5 — lethal for misrouted fixtures. Lock **odds-implied** or **league 1X2 role** marginal first, then adjust.
5. **Lexical / epithet markets** (*Will X say "Y" by date?*): Corpus token frequency or market price **before** a tight `p_yes`. Missing counts ⇒ **widen upward**; sparse insult tails punish confident lows.
6. **Twin coherence**: Same fixture/market, divergent `p_yes` without evidence ⇒ cite anchor explicitly or widen — signals missing consensus line.
7. **Step budget honesty**: `max_steps: 1` caps refinement. If odds, lineups, or phrase-frequency are still missing after the hop, **state that** and widen — do not fake precision.

## Phase 2 — Reflect

After each batch, the reflection loop:
1. Reads all resolved episodes, computes per-category Brier
2. Reads the vault (strategy + approach notes)
3. Updates `.harness/policy.md` — machine config + lessons
4. Updates `_strategy.md` and approach notes with findings, wikilinks
5. Archives old policy to `.harness/policy_history/`

## Vault map

- [[approaches/sports-fixtures]] — playbook for club/date win questions
- [[runs/batch-postmortem-2026-05]] — calibration table, worst episodes, per-check telemetry, lexical checklist
