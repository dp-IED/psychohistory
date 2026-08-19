# Horizon mix and analog evidence, not historical scoring

Domain mix (ADR 0011) stops a large `K` from clustering in one topic. A second clustering risk remains: every new problem sitting in the current news week. That trains overlay that reads today’s feed, not overlay that uses structure from earlier cases. Psychohistory is the latter.

**Yes, there is a benefit**, if the mix is this:

- **Horizon mix.** When a discover tick opens more than one problem, include at least one **near** resolution day (within about 30 days of today — fast series for reflection) and at least one **far** resolution day (90 days or more — time for analog and base-rate methods to be the skill, not a last-day scrape). Do not fill `K` with only this week’s deadlines or only distant dates with no near score.
- **Evidence-regime mix.** At least one new problem should be **news-now** (deadline, live talks, a scrape can move the claim). At least one should be **analog/base-rate**: motivation says the interesting method is a past class of cases, a structural rate, or a method this plugin already claimed after a prior reflection — still scored in the future.
- **Transfer reopen.** After reflection grows the overlay, prefer a *new live* problem in that structural class so the next series can falsify transfer. That is learning from this plugin’s past. It is not replaying gold.

**Not this mix** (ADR 0002, 0003; `next_steps.md`): do not open problems whose resolution day is already past; do not use historical cutoffs, PIT, or gold Brier as the epoch; do not ask the model to “forecast” what the world already answered. A 2023 replay cannot grade this LLM. Learning from the past belongs in **justification methods**, **analog cards** (ADR 0013), and in **which live problems we open**, not in the scoring clock.

No quality gate (ADR 0005). Discover still does not write claims (ADR 0009).
