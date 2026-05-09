# Next Steps — Agentic Reframe (review handoff)

**Audience:** Reviewers and executor models (Hermes/Codex) with full-repo context.
**Status:** Planning document; supersedes the GNN-pipeline sequencing in the previous `next_steps.md`.
**Alignment:** [`project.md`](project.md), [`roadmap.md`](roadmap.md), [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md), [`forecast_charter.md`](forecast_charter.md), PR #30 (`dp/polymarket-agentic-harness`).

---

## 0. Architectural reframe — what changed and why

The previous `next_steps.md` sequenced work as a deterministic pipeline: graph builder → encoder → assumption layer → forecast head → world model → Polymarket. That framing treated Track M (markets) as a downstream product and gated it behind WM v0.

This version reflects a fundamental reframe:

**The system is a memory-driven analyst agent, not a GNN forecasting pipeline.** The graph builder, GNN encoder, and assumption layer are tools the agent can invoke — not the loop itself. Polymarket Brier scores are the primary feedback signal that drives self-improvement, not a late-stage validation artifact. The GNN is valuable not because neural message-passing is the right architecture for political forecasting, but because the graph is the right *representation* for compressing relational political history into agent-readable form.

Three corollaries:

1. **Track M moves up.** Markets are now parallel with Steps D–G, not gated behind H. The agent loop needs a scoring signal from day one.
2. **Step D changes character.** The "training loop skeleton" is now the *agent loop state machine*, not a `train.py` with gradient flow. Brier score replaces loss as the primary feedback signal.
3. **Memory is the core asset.** A cold-start agent that re-derives every pattern from scratch cannot build the kind of long-horizon analytical insight (Luxemburg's economic-political fusion thesis, etc.) that makes forecasts interesting. The memory system must be designed before the loop is implemented.

What does *not* change: PIT discipline, cutoff policy, adversarial harness requirements, France as validation smoke, Iran as shadow/red-team only, and all risk mitigations from the previous version.

---

## 1. Program goals (this phase)

### 1.1 Primary outcomes

1. **Memory system design and interface** — before any agent loop code.
   Three-layer schema (`EpisodicRecord`, `ConceptualPattern`, `StructuralFact`) + typed `MemoryStore` protocol. Backend-agnostic. `NullMemoryStore` for stateless testing. `JsonlMemoryStore` as first real backend.

2. **Agent loop — core state machine** — replaces the GNN pipeline as the central loop.
   Orchestrates: memory read → planning → research loop (tool calls) → synthesis → memory write. GNN, graph builder, web search, and analogues are all *tools* this loop calls. Stop conditions driven by `ConstructionPolicy`.

3. **Blind spot → query mapper** — the narrowest entry point to retrieval.
   Maps `blind_spot_check` strings + `MarketFrame` → `WebSearchRequest`. Template registry first; LLM fallback. Cutoff discipline enforced in tests.

4. **Brier scoring + episode resolution** — closes the feedback loop.
   On market resolution: find `EpisodicRecord`, compute Brier, infer which skipped checks would have helped, update memory. This is what enables Option C.

5. **Context compression skills** — graph and chart compression for agent context.
   40 events → compact graph < 500 tokens. Fixed edge taxonomy. Deterministic. This is what lets long-running, long-context jobs stay token-efficient.

6. **Policy self-improvement (Option C)** — after Brier scoring is live.
   Bad Brier score + episode → `PolicyPatchProposal`. Gate: tests pass + Brier improves on held-out split + rationale references evidence episodes. Accepted patches write a `ConceptualPattern` with `source="policy_patch"`.

7. **Graph builder + assumption layer** — steps E and F from the previous plan, now as agent tools.
   Still valid work. Now wired as callables into `AgentToolset` rather than as standalone pipeline stages. Sequencing is unchanged: E before F.

### 1.2 Explicit non-goals for this phase

- Productising the agent as a user-facing API before memory + loop + scoring are stable.
- Over-optimising the forecast head before retrieval and assumption interfaces are ablation-stable.
- Using Iran as the primary benchmark before France smoke + held-out eval contracts exist.
- Blocking WM v0 on perfect builder specs — WM is still in scope (Step H) but sequenced after the agent loop + memory system are green.
- Re-proving France on every unrelated change.

### 1.3 Success criteria

| Gate | Criterion |
|------|-----------|
| **Memory** | Three-layer schema committed, `JsonlMemoryStore` passing all protocol tests |
| **Loop** | Agent loop runs end-to-end with stub tools, writes `EpisodicRecord` on exit |
| **Scoring** | Brier computed and written to episode on resolution; miss inference runs without crash |
| **G4** (markets) | Masked runs + coverage reporting; non-market slices explicit |
| **G5** (discovery) | Held-out time/region + ablations across builder vs encoder vs WM as credited |

---

## 2. Memory system (start here)

Memory is the architectural primitive that separates a cold-start pipeline from a continuously learning analyst. Design it before the loop so hooks are built in, not retrofitted.

### 2.1 Three-layer schema

| Layer | Type | Written by | Read by | Purpose |
|-------|------|-----------|---------|---------|
| `EpisodicRecord` | Per-job record | Agent loop on exit; updated on resolution | Loop planning phase; policy patch evaluator | What happened in past runs; what was missed |
| `ConceptualPattern` | Named analytical pattern | Hand-authored; policy patch on acceptance | Loop planning phase (context injection) | Patterns that proved predictive across jobs |
| `StructuralFact` | Graph fact | Graph query tool | Graph builder; loop planning | Facts that don't need re-fetching |

Key field on `ConceptualPattern`: `source: Literal["hand_authored", "agent_proposed", "policy_patch"]`. Do not collapse this — it is what distinguishes hand-authored `ConstructionPolicy` checks from agent-discovered patterns. The self-improvement loop depends on it.

### 2.2 Memory store protocol

`MemoryStore` is a typed Protocol. Backend is swappable. Required methods:

```python
class MemoryStore(Protocol):
    def write_episode(self, record: EpisodicRecord) -> None: ...
    def update_episode_brier(self, job_id: str, brier_score: float, misses: list[str]) -> None: ...
    def read_recent_episodes(self, market_family: str, n: int) -> list[EpisodicRecord]: ...
    def write_pattern(self, pattern: ConceptualPattern) -> None: ...
    def read_patterns(self, market_family: str) -> list[ConceptualPattern]: ...
    def write_fact(self, fact: StructuralFact) -> None: ...
    def read_facts(self, subject: str) -> list[StructuralFact]: ...
```

Provide `NullMemoryStore` (no-ops) immediately. `JsonlMemoryStore` (flat JSONL per layer) as first real backend — sufficient for local dev and Hermes runs.

### 2.3 Deliverables

| File | Contents |
|------|----------|
| `harness/memory_schema.py` | Three frozen dataclasses, validated, with docstrings on every field |
| `harness/memory_store.py` | `MemoryStore` Protocol + `NullMemoryStore` + `JsonlMemoryStore` |
| `tests/test_memory_schema.py` | Construction, validation, serialisation round-trips |
| `tests/test_memory_store.py` | Protocol conformance for both backends |

---

## 3. Agent loop (after memory schema)

The loop is the central loop the system runs per market question. It is not a training loop in the gradient-descent sense — it is a policy-driven research loop scored by Brier.

### 3.1 Loop contract

```python
def run_agent_loop(
    question: str,
    cutoff_date: date,
    resolution_date: date,
    policy: ConstructionPolicy,
    memory: MemoryStore,
    tools: AgentToolset,
) -> AgentLoopResult:
    ...
```

`AgentToolset` is a dataclass of callables: `web_search`, `graph_query`, `gnn_score`, `analogues`, `market_context`. The loop calls them — it does not implement them.

### 3.2 Loop phases

1. **Load memory** — read recent episodes for this market family; read applicable `ConceptualPattern`s. Inject into planning context.
2. **Plan** — given question + cutoff + patterns, determine which `blind_spot_checks` to run and in what order. LLM call here.
3. **Research loop** — iterate over checks; call tools; update `AgentLoopState`. Stop when: Δ`gnn_score` < ε on last 2 calls, max steps reached, or planner declares convergence.
4. **Synthesise** — reconcile competing hypotheses, apply calibration, produce final `p_yes` + confidence interval.
5. **Write episode** — commit `EpisodicRecord`; `brier_score` field left `None` until resolution arrives.

### 3.3 Blind spot → query mapper

The narrowest entry point. Lives in `harness/query_mapper.py`. Template registry in `harness/query_templates.py` (at least 5 check strings with deterministic templates on first commit). LLM fallback for unrecognised checks. Cutoff discipline: tests must assert no query can be formulated that leaks post-cutoff information.

### 3.4 Deliverables

| File | Contents |
|------|----------|
| `harness/agent_loop.py` | Core state machine, `AgentLoopState`, `AgentLoopResult` |
| `harness/agent_toolset.py` | `AgentToolset` dataclass with typed callable signatures |
| `harness/query_mapper.py` | `blind_spot_to_query()`, `WebSearchRequest` |
| `harness/query_templates.py` | Template registry (5+ deterministic templates) |
| `tests/test_agent_loop.py` | End-to-end with `NullMemoryStore` + stub tools |
| `tests/test_query_mapper.py` | Template matching, LLM fallback, cutoff assertion |

---

## 4. Brier scoring + resolution listener

Closes the feedback loop. Implement after the loop runs end-to-end with stub tools.

```python
def resolve_market(
    job_id: str,
    outcome: bool,
    memory: MemoryStore,
    tools: AgentToolset,
) -> BrierUpdateResult:
    ...
```

Brier score: BS = (p_yes − outcome)². Miss inference: for each skipped check in the episode, test whether firing it would have shifted `p_yes` toward the correct answer by > 0.05. Log as heuristic — not ground truth. `update_episode_brier()` called exactly once per resolution.

Deliverable: `harness/resolution.py` + `tests/test_resolution.py` (integration test: run loop → resolve → verify episode updated).

---

## 5. Context compression skills

Long-running, long-context jobs require token efficiency. Compression skills convert raw evidence into agent-readable compact representations.

### 5.1 Graph compression skill

Input: `list[EventRecord]` (date, actors, event_type, description, source_url).
Output: `CompressedGraph` — nodes (deduplicated actors/entities), edges (typed: `escalated`, `de-escalated`, `mediated`, `sanctioned`, `allied`, `opposed`), summary stats (event count, date range, dominant relation types, most central actors).

Target: 40 events → < 500 tokens when serialised as structured text for agent context. Deterministic. Alias registry for name deduplication (not NER).

### 5.2 Chart/timeline compression skill

Temporal sequences → event timeline representation for agent context. Particularly for macro indicator history and base rate sequences. Output is structured text, not a rendered image.

Deliverables: `harness/skills/graph_compression.py`, `harness/skills/timeline_compression.py`, tests.

---

## 6. Policy self-improvement (Option C)

Implement after Brier scoring is live and at least 10 resolved episodes exist in memory.

### 6.1 PolicyPatchProposal

```python
@dataclass
class PolicyPatchProposal:
    proposal_id: str
    triggered_by_job_id: str
    brier_score: float
    proposed_check: str
    proposed_family_scope: list[str]
    rationale: str               # must reference at least one evidence_episode
    evidence_episodes: list[str]
    confidence: float
    status: Literal["pending", "accepted", "rejected"]
```

### 6.2 Gate conditions (all must pass for acceptance)

1. Full test suite passes (425+).
2. Brier improves by > 0.02 on held-out validation split (last 10 resolved markets in family).
3. `proposed_family_scope` is non-empty and specific — `["all"]` is rejected.
4. `rationale` is > 50 words and references at least one `evidence_episode` job ID.

Accepted patches write a `ConceptualPattern` with `source="policy_patch"`. Rejected patches are logged but not deleted.

Deliverables: `harness/policy_patch.py`, `tests/test_policy_patch.py`.

---

## 7. Existing pipeline steps (D, E, F, G, H) — status and reframing

These steps from the previous plan remain valid. Their character changes slightly: they now produce *agent tools*, not pipeline stages.

| Step | Description | Reframe |
|------|-------------|---------|
| **D** | Training loop skeleton | Now: agent loop state machine (`harness/agent_loop.py`). Brier replaces loss. |
| **E** | Graph builder v1 | Now: `graph_query` tool callable in `AgentToolset`. ANN top-100 → rerank ≤50 nodes. |
| **F** | Assumption layer v0 | Now: soft gates modulate agent attention; callable as `assumption_check` tool. |
| **G** | Forecast head | Now: `gnn_score` tool callable. Lightweight head on induced subgraph. |
| **H** | WM v0 + multi-step | Unchanged. Sequenced after agent loop + memory are green on pinned subgraph contract. |

Data prerequisites (A, B, C, C′) and world-model ablation matrix (GRU-only vs GNN-only vs GRU+GNN) from the previous plan remain unchanged.

---

## 8. Track M — Markets (now parallel with D–G, not gated behind H)

The agentic reframe promotes Track M. Polymarket Brier scores are the primary feedback signal for the self-improvement loop — they cannot wait until after WM v0.

| Phase | Deliverable | Timing |
|-------|-------------|--------|
| M1 | Schema + ingestion: quotes, resolutions, metadata; versioned raw; loader tests | Parallel with Steps D–E |
| M2 | Synthetic adversarial tests before real labels: future resolution, retroactive correction — pipeline must fail closed or alarm | Before M3 |
| M3 | Coverage audit: table of domains/eras with no market coverage vs eval geography | Before first real Brier computation |
| M4 | Label contracts: resolution head first; short-horizon later; masking ablations; baselines | After M2 + M3 |

PIT discipline unchanged: adversarial tape tests before trusting production labels.

---

## 9. Iran stress-test lane (unchanged from previous plan)

Iran remains a shadow/red-team slice after Step H. Do not use as primary benchmark before France smoke + held-out eval contracts are stable. See §2.3 of the previous plan for full audit checklist — that section is still operative.

---

## 10. Sequencing summary

```
Memory schema          ──►  Memory store interface
                                    │
                                    ▼
Blind spot mapper ──►  Agent loop state machine  ◄──  Graph builder (E)
                                    │                  Assumption layer (F)
                                    │                  GNN score tool (G)
                                    ▼
                          Brier scoring + resolution
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
           Context compression            Policy self-improvement
           skills (graph, chart)          (Option C, after 10+ episodes)
                                                   │
                                                   ▼
                                          WM v0 + ablation matrix (H)
```

Track M (markets) runs parallel to the agent loop, providing the resolution signals that feed Brier scoring.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory retrofitting | Hooks added late are brittle | Design schema before loop — mandatory sequencing |
| Cold-start episodes | No Brier history → self-improvement cannot fire | Hand-author 5–10 synthetic `EpisodicRecord`s for early testing |
| Policy patch overfitting | Patches tune to gold set, not generalisable patterns | Held-out split gate + family scope specificity requirement |
| Context compression information loss | Compressed graph drops signal needed for forecast | Lossiness tests: compare `gnn_score` on raw vs compressed subgraph; flag divergence > 0.05 |
| Label leakage | Invalidates Brier scores | Adversarial harness on synthetic tapes before real resolution labels |
| Credit assignment (WM) | Temporal + MP fusion opaque | Time-then-space architecture; separate forward blocks; multi-seed ablations |
| Contract drift | Runs mix training and validation corpora | Tag all runs by training corpus; France = validation/smoke unless explicitly bridged |

---

## 12. Executor checklist

- Re-read [`roadmap.md`](roadmap.md) gates for the current change.
- Memory schema before loop — do not start `agent_loop.py` before `memory_schema.py` is committed and tested.
- Any Brier claim: attach `job_id`, `market_id`, `cutoff_date`, `resolution_date`.
- Compression skills: test lossiness against raw subgraph before deploying as default context representation.
- Markets: adversarial tape tests before trusting production resolution labels.
- Policy patches: all four gate conditions must pass; log rejections.
- Iran: shadow/red-team only after Step H; keep France as ablation-controlled harness.

---

## 13. Motion task index (new tasks — agentic reframe)

| Task | Suggested deadline | Priority | Dependencies |
|------|--------------------|----------|--------------|
| [Harness] Memory Schema — Three-Layer Design | 2026-05-22 | High | None |
| [Harness] Memory Store Interface — Typed Protocol | 2026-05-23 | High | Memory Schema |
| [Harness] Blind Spot → Query Mapper | 2026-05-28 | High | ConstructionPolicy |
| [Harness] Agent Loop — Core State Machine | 2026-05-30 | High | Memory Schema, Memory Store, Query Mapper |
| [Harness] Brier Scoring + Episode Update on Resolution | 2026-06-05 | Medium | Agent Loop, Memory Store |
| [Memory] Context Compression — Graph Skill | 2026-06-10 | Medium | EventRecord schema |
| [Policy] Self-Improvement — Policy Patch Protocol | 2026-06-25 | Low | Brier Scoring, Memory Store, Agent Loop |

---

## 14. References

- [`roadmap.md`](roadmap.md) — stages, gates, execution reality
- [`forecast_charter.md`](forecast_charter.md) — metrics, markets tier
- [`docs/graph-builder-contract-v0.1.md`](docs/graph-builder-contract-v0.1.md) — locked builder, assumptions, supervision, compute, training contexts
- [`docs/research/architecture.md`](docs/research/architecture.md) — target layers + implementation status
- [`docs/research/outputs/perplexity.md`](docs/research/outputs/perplexity.md)
- PR #30 `dp/polymarket-agentic-harness` — agentic harness scaffold (typed contracts, 425 passing tests)
