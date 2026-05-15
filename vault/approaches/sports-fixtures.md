# Approach: sports fixtures (win / draw / lose on a date)

Use when the question matches: **named team(s)** + **outcome type** (win, draw, advance, cover, etc.) + **calendar context** (explicit date or matchday).

## Preconditions before a number

1. **Identify the fixture**: competition, round, home vs away, **named opponent** if discoverable via PIT search at cutoff — a date + club alone is not enough to anchor odds if the league week has multiple matches.
2. **League priors**: cite typical home/draw/away or moneyline-structure baselines for that league or tournament stage; adjust for cup vs league if relevant.
3. **Lineup / availability**: note injuries, suspensions, international break fatigue, or *unknown* availability with implied width on the interval.
4. **Numeric anchor**: decimal odds (convert to implied prob and remove vig mentally if multiple books agree), or model/Elo win expectancy — **not** vibes.

## Harness alignment

- **Default**: `.harness/policy.md` may keep `blind_spot_checks: []` so non-fixture questions do not burn `max_steps` on irrelevant templates.
- **Fixture-heavy evaluation**: list `sports_match_calendar_check`, `base_rate_league_prior_check`, and `injury_suspension_lineup_check` in YAML — keys **must** match `TEMPLATE_REGISTRY` in `harness/query_templates.py` exactly (copy-paste from `.harness/policy.md` § *Valid blind_spot_checks identifiers*).
- **Always**: the same three sections **in prose**, whether hooks fired, skipped, or absent from the episode record.

## Failure modes we observed

- **Label lag**: `market_family=general` on clear fixtures → generic shrinkage path instead of odds-anchored probability (Man U 2025-05-25 twins **job-aa1fb95dd0e5 / job-61b67fb6ab37**; Inter row **job-065883c872a2** in [[runs/batch-postmortem-2026-05]]).
- **Hooks skipped, brain skipped**: Telemetry `skipped` or empty check lists must not correlate with omitting those paragraphs in text.
- **Twin-run instability**: Same question + different `p_yes` without distinct evidence ⇒ no shared external line — cite the numeric anchor or widen explicitly.
- **Overprecision on thin evidence**: High Brier on favorites and dogs — when anchors are missing, prefer honest spread toward league role marginal + wide CI.

## Links

- Policy: `.harness/policy.md` — family routing, minimum evidence gate
- Postmortems: [[runs/batch-postmortem-2026-05]]
