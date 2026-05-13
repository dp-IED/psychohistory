# Next Steps — Phase 1 Gate Snapshot

**Status:** `58 passing` — **Context Compression gate is met**.

This is a checkpoint summary of where the full spec stands, what is unblocked now, and what should happen next.

## Spec Gate Status

| Gate | Criterion | Status |
|---|---|---|
| Memory Schema | Three-layer schema, all protocol tests | ✅ |
| Memory Store | `JsonlMemoryStore` passing all protocol tests | ✅ |
| Agent Loop | Runs end-to-end, writes `EpisodicRecord` on exit | ✅ |
| Brier Scoring | Computed + written to episode on resolution | ✅ |
| Context Compression | Graph + timeline skills, token budget enforced | ✅ |
| Policy Self-Improvement | Requires 10+ resolved episodes | 🔜 |
| Graph Builder (Step E) | `graph_query` tool callable | 🔜 |
| Assumption Layer (Step F) | `assumption_check` tool callable | 🔜 |
| GNN Score Tool (Step G) | `gnn_score` tool callable | 🔜 |
| WM v0 (Step H) | After agent loop + memory green | 🔜 |

## Phase 1 Interpretation

You have completed **§§1–5 of the spec**.

**Policy Self-Improvement (§6) is now the only remaining Phase 1 gate**, and it cannot fire until memory contains **10+ resolved episodes**. That requires running live competition questions now.

## Practical Meaning

You can **go live on AIB today** with the current harness chain:

```text
fetch question → run_agent_loop → post_forecast → poll resolution → resolve_market → Brier in memory
```

This chain is tested end-to-end. Remaining stubs (`web_search`, `gnn_score`, `graph_query`) still allow valid forecast posting, but forecast quality will be constrained until evidence tools are wired.

## Recommended Next Session Order

1. **Wire `web_search` to AskNews**  
   Highest immediate ROI for forecast quality; supports PIT via `as_of`.

2. **Run 5–10 live AIB questions**  
   Start accumulating resolved episodes and real Brier signal in memory.

3. **Hand-author 3–5 `ConceptualPattern`s**  
   Bootstrap pattern library from observed live runs.

4. **Implement Policy Self-Improvement (§6)**  
   Do this once the 10+ resolved-episode gate is satisfied.

## Prioritization Note

Steps **E/F/G** (graph builder, assumption layer, GNN tool) remain important but are lower immediate ROI than starting live runs and capturing real scoreboard feedback.

## Reference

- Source planning note: `next_steps.md` input provided in session
- AIB context + AskNews/PIT relevance: Effective Altruism Spring 2026 AIB announcement
