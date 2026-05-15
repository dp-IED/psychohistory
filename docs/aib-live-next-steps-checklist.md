# AIB Live Next Steps — Branch Scope (`dp-IED/aib-live-next-steps`)

Status baseline: 58 passing; context compression gate met; §§1–5 complete.

## PR 1 — AskNews `web_search` tool (current)
- [ ] Add `harness/tools/web_search.py` with `AskNewsSearchTool`
- [ ] PIT mapping: `as_of_date -> publishedBefore`, default `publishedAfter = as_of_date - window_days`
- [ ] PIT guard: fail closed if any result has `published_at > as_of_date`
- [ ] Add `tests/test_web_search.py` (mocked HTTP only)
- [ ] Wire AskNews into `competition_runner.py` `AgentToolset` build path
- [ ] Env var contract: `ASKNEWS_API_KEY` optional; missing key keeps stub behavior

## PR 2 — First live AIB runs
- [ ] Add `scripts/run_aib_batch.py`
- [ ] Fetch 10 open questions, run loop, post forecasts
- [ ] Log run outputs to `runs/YYYY-MM-DD.jsonl`
- [ ] No new harness internals (wiring only)
- [ ] Success: 10 `EpisodicRecord`s in `JsonlMemoryStore`

## PR 3 — Hand-authored pattern seeds
- [ ] Add `data/patterns/seed_patterns.json` (5–10 `ConceptualPattern`s)
- [ ] Add `scripts/load_seed_patterns.py` one-shot loader
- [ ] Derived from observations of first live runs

## PR 4 — Policy Self-Improvement (§6)
- [ ] Add `harness/policy_patch.py` (`PolicyPatchProposal`, gates, accepted patch -> `ConceptualPattern`)
- [ ] Add `tests/test_policy_patch.py`
- [ ] Hard gate in code: requires 10+ resolved episodes

## Ordering rationale
1. AskNews evidence improves forecast quality immediately.
2. Live runs create real episodes/Brier signal.
3. Seed patterns bootstrap planning prior to enough auto-learned data.
4. Policy self-improvement after resolved-episode threshold is met.
