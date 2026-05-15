# Batch postmortem — 2026-05-15 (4-batch Polymarket run)

Current-run metrics only. Previous epochs (stub-era episodes with degenerate 0.565 prior) are excluded from these numbers.

## Calibration snapshot

| Metric | Value |
|--------|--------|
| Overall Brier | **0.2064** |
| General | 0.1978 |
| Crypto | 0.2407 |

## Worst episodes (this run)

| job_id | family | question (abridged) | final_p_yes | Brier | lesson |
|--------|--------|---------------------|-------------|-------|--------|
| job-154215d1f222 | general | Trump says "Sleazebag" by Feb 28 | 0.2791 | 0.5197 | Lexical: corpus anchors needed — same question as previous worst episode; inconsistent across draws |
| job-9b17de1f7044 | general | Trump says "Sleazebag" by Feb 28 | 0.2983 | 0.4923 | Twin-run dispersion without external line |
| job-62ce98573af2 | general | Universitatea Craiova CS win 2025-10-23 | 0.5864 | 0.3439 | Fixture as general: sports workflow needed, odds-implied anchor missing |

## Lexical / exact-phrase playbook

1. **Quantitative anchor first**: PIT-respecting counts of exact token or tight variants **or** a cited market price.
2. **Exposure model**: remaining time × expected public remarks — how many rhetorical "draws" remain before resolution?
3. **Asymmetric tail risk**: for insult / epithet prompts, missing counts ⇒ widen **up**; **do not** land on sharp low `p_yes` from "he rarely says that" alone.

## Per-check telemetry (this run only)

No blind_spot checks were configured (`blind_spot_checks: []`). All PIT research runs via `pit_search.py` in the synthesis prompt, not via the check registry. For fixture-heavy eval batches, paste the sports triad into YAML temporarily to fire retrieval.

## Sports trio — YAML snippet (fixture-heavy eval only)

```yaml
blind_spot_checks:
  - sports_match_calendar_check
  - base_rate_league_prior_check
  - injury_suspension_lineup_check
```

Revert to `blind_spot_checks: []` for mixed-category work.

## Next-run checklist

1. Regex / intent: fixtures → sports workflow regardless of upstream `market_family`.
2. Every sports answer: three short sections (calendar, league rates, injuries) matching registry prompts — regardless of telemetry.
3. Do not settle a forecast with **fewer than three evidence items** **or** without a **quantitative** anchor where the question implies one (sports, lexical).
4. Exact-phrase: retrieval for verbatim usage **and** similar markets; wide uncertainty if still blind after `max_steps`.
5. Rotate YAML sports trio only when batches are fixture-heavy; push family-conditional injection to harness code when possible.

## Related

- [[approaches/sports-fixtures]]
- [[_strategy]]
