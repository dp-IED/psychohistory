# GNN + Agent Architecture: Sprints & Specs

Status: spec, not yet building
Last updated: 2026-05-22

## Architecture (resolved)

```
  Vault (threads, entities, concepts, timeline, tags)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  ┌─────────┐                          ┌──────────┐
  │   GNN   │                          │  Agent   │
  │         │                          │          │
  │ Learns: │    ┌──────────┐         │ Reasons: │
  │ struct  │◄───│ features │─────────│ explicit │
  │ patterns│    │ (tags,   │         │ mechs,   │
  │ from    │    │  domain, │         │ vault    │
  │ vault + │    │  window) │         │ context  │
  │ labels  │    └──────────┘         │          │
  │         │                          │          │
  │ p_gnn   │                          │ p_agent  │
  └────┬────┘                          └────┬─────┘
       │                                    │
       └──────────┬─────────────────────────┘
                  │
                  ▼
         |p_gnn - p_agent| > threshold?
              │            │
         HIGH-VALUE    NORMAL
         forecast      weight by
                       track record
```

Key decisions:
- GNN reads vault features (tags v1, DSL v2), not Polymarket tags alone
- Agent is in charge — decides features, decides adjustments
- Disagreement = measurement opportunity, not error
- DSL bridges agent reasoning → GNN-understandable vocabulary (deferred to Sprint 5)
- Retraining: periodic (weekly/monthly), not per-forecast

---

## Sprint 1: GNN MVP — tag-based baseline with interactions

**Goal:** A GNN that beats the current Beta-Binomial tag model by learning
tag interactions. Train on 1,277 resolved Polymarket markets.

**Input:** `resolved_markets.jsonl` (1,277 markets, 415 tags, resolution labels)
**Output:** `p_gnn = GNN(tags)` with per-market predictions + embeddings

**Graph structure:**
- Market nodes: 1,277, with resolution labels (YES/NO)
- Tag nodes: 415, with one-hot type features
- Edges: market ↔ tag (if market has that tag)
- Edge weights: normalized co-occurrence frequency

**Model:** 2-layer GraphSAGE or GAT
- Layer 1: tags aggregate signal from connected markets
- Layer 2: markets aggregate from connected tags
- Readout: MLP → sigmoid → P(YES)

**Training:**
- 80/20 time-series split (train on earlier markets, test on later)
- Binary cross-entropy loss
- Evaluate: Brier score, calibration error, AUC

**Deliverables:**
- `harness/gnn_tag.py` — model definition + training
- `scripts/train_gnn.py` — training loop + eval
- `data/polymarket/gnn_tag_model.pt` — saved model
- Comparison: GNN vs Beta-Binomial on same test split

**Success criteria:** GNN Brier ≤ Beta-Binomial Brier on test set
**Risk:** 1,277 markets may not be enough for meaningful interaction learning.

---

## Sprint 2: Agent ↔ GNN integration

**Goal:** Agent queries the GNN as part of its forecast pipeline.

**Integration point:**
```python
# Agent's forecast pipeline
vault_context = read_vault(question)
tags = identify_tags(question, vault_context)  # agent's job
features = encode_features(tags, domain, window)

gnn_result = gnn.predict(features)
# → {"p_yes": 0.34, "ci": [0.22, 0.50], "similar_markets": [...]}

# Agent reasons about the gap
if gnn_result["p_yes"] > 0.8:
    # GNN is very confident — agent must justify any deviation
    ...
elif len(gnn_result["similar_markets"]) < 5:
    # GNN has few analogs — agent's vault reasoning matters more
    ...

# Agent outputs final prediction + reasoning trail
```

**Deliverables:**
- `harness/forecaster.py` — updated with GNN query step
- Agent instructions updated: how to interpret GNN output, when to override
- Test: agent forecasts on 20 resolved markets with and without GNN

**Open question:** How does the agent decide when to trust the GNN vs override?
  - GNN confidence (narrow CI = trust, wide CI = agent's turn)
  - Number of similar training markets (few analogs = agent must do the work)
  - Domain coverage (if vault covers this domain well, agent may know more)

---

## Sprint 3: Disagreement tracking

**Goal:** When GNN and agent diverge, record both predictions and measure who
was right. This is the core measurement loop.

**Disagreement threshold:** |p_gnn - p_agent| > 0.15

**Recording:**
```json
{
  "question": "Will Iran strike Israel by June 30, 2026?",
  "cutoff": "2026-05-22",
  "p_gnn": 0.34,
  "p_agent": 0.62,
  "agent_reasoning": "escalation-ladder active, commitment trap from US...",
  "features_passed": ["Iran", "Geopolitics", "Middle East", "TIME_WINDOW(<30d)"],
  "gnn_confidence": 0.22,  // CI width
  "resolution": null,       // filled when resolves
  "gnn_correct": null,      // filled when resolves
  "agent_correct": null
}
```

**Deliverables:**
- `harness/disagreement.py` — tracking + reporting
- `data/polymarket/disagreements.jsonl` — historical record
- Dashboard: agent win rate vs GNN, by domain, by mechanism type

**Success criteria:** After 20+ disagreements with known resolutions,
  can answer: "in what situations does the agent beat the GNN?"

---

## Sprint 4: Vault-aware GNN

**Goal:** GNN reads vault structure, not just tags. Entity nodes, concept
nodes, thread nodes, event nodes all participate in message passing.

**Graph expansion:**
- Entity nodes (from vault): countries, people, institutions
- Concept nodes: mechanisms, frameworks
- Thread nodes: topical storylines
- Event nodes: timestamped occurrences
- Edges: wikilinks, temporal (event → event), membership (entity → thread)

**Training:** Self-supervised pretraining on vault structure (link prediction,
node type classification), then fine-tune on resolution labels.

**Feature encoding:** Each node gets:
- Type embedding (entity, concept, thread, event)
- Domain embedding (mena, east-asia, global, usa)
- Text embedding (LLM embedding of title/description)

**Deliverables:**
- `harness/gnn_vault.py` — heterogeneous GNN
- Pretrained embeddings for all vault nodes
- Updated agent integration: agent specifies which vault nodes are relevant

**Risk:** Vault coverage is uneven. Some domains have rich threads, others are
  stubs. The GNN must learn to discount thin domains.

---

## Sprint 5: DSL — agent → GNN vocabulary

**Goal:** A closed vocabulary of primitives that the agent uses to describe
novel situations. The GNN was trained on these primitives, so it understands
new compositions.

**Primitive types (draft):**
```
TEMPORAL:
  TIME_WINDOW(days)        — resolution horizon
  BEFORE(event)            — must resolve before X
  WITHIN(days)             — within N days of cutoff

CAUSAL:
  CAUSAL_CHAIN(a→b→c)     — chain of causal links
  COMMITMENT_TRAP(patron)  — patron's public commitment binds local actor
  ESCALATION_LADDER        — tit-for-tat cycle with threshold

STRUCTURAL:
  FPTP_SYSTEM              — first-past-the-post electoral rule
  INSTITUTIONAL_VETO(body) — body can block outcome
  COALITION_REQUIRED       — no single actor can decide

EVIDENCE:
  PUBLIC_ANNOUNCED         — has been publicly stated
  PRIVATE_NEGOTIATION      — behind closed doors
  SCHEDULED(event)         — on a known calendar
```

**Agent describes a novel concept:**
```
trump-tariff-escalation-spiral:
  CAUSAL_CHAIN(
    tariff_announcement →
    retaliation →
    escalation →
    economic_crisis
  )
  WITHIN TIME_WINDOW(days=90)
  GATED_BY INSTITUTIONAL_VETO(congress)
```

**GNN sees this as a feature vector** of primitive activations — same encoding
it was trained on. No retraining needed.

**Deliverables:**
- `harness/dsl.py` — primitive definitions + composition
- Agent instructions: how to encode novel concepts using DSL
- Updated GNN training: includes DSL primitives as features
- Test: agent describes a novel mechanism, GNN pools it correctly

**Open question:** Who defines the primitives? Agent discovers them from existing
  concepts? Human-curated? Starts with a seed set and agent proposes additions?

---

## Sprint 6: Training loop + blindspot detection

**Goal:** Periodic retraining, domain expansion, and systematic detection of
where the GNN (and agent) are weakest.

**Retraining cadence:** Weekly
- Ingest new resolved markets from pmxt
- Retrain GNN on expanded dataset
- Recalibrate per-mechanism diagnostics
- Update agent's mechanism library

**Blindspot detection:**
- For each Polymarket tag cluster, compute GNN calibration error
- Tags with high error + wide CI → "blindspot"
- Agent is notified: "GNN is unreliable for tag cluster X — you must do the work here"
- Agent can propose new vault content (threads, concepts) and the GNN retrains

**Continuous measurement:**
- Every forecast: record p_gnn, p_agent, resolution
- Rolling window: last 50 forecasts by domain
- Per mechanism: does invoking mechanism X improve over GNN?

**Deliverables:**
- `scripts/retrain_gnn.py` — scheduled retraining
- `harness/blindspot.py` — detection + reporting
- Cron job: weekly `fetch → retrain → report`

---

## Current state (end of Sprint 0)

| Component | Status | Path |
|---|---|---|
| Data fetcher (pmxt) | ✅ | `scripts/fetch_calibration_data.py` |
| Tag calibration (Beta) | ✅ | `harness/tag_calibration.py` |
| PIT training harness | ✅ | `scripts/train_calibration.py` |
| Reasoning trail parser | ✅ | `harness/reasoning_trail.py` |
| Mechanism graph | ✅ | `harness/mechanism_graph.py` |
| Graph calibration | ✅ | `harness/graph_calibration.py` |
| GNN (tag-based) | 🔲 Sprint 1 | — |
| Agent ↔ GNN | 🔲 Sprint 2 | — |
| Disagreement tracking | 🔲 Sprint 3 | — |
| Vault-aware GNN | 🔲 Sprint 4 | — |
| DSL | 🔲 Sprint 5 | — |
| Training loop | 🔲 Sprint 6 | — |

---

## Open questions (unresolved)

1. **GNN model choice:** GraphSAGE vs GAT vs simple linear? Depends on how much
   interaction signal exists in 1,277 markets.

2. **Agent trust heuristic:** What formula decides whether the agent trusts the
   GNN or overrides? CI width? Number of similar training markets? Domain coverage?

3. **DSL primitives:** Who defines them and how do they evolve? Agent-discovered
   from existing concepts? Human-seeded?

4. **Vault coverage:** The GNN can only learn from domains with vault content
   AND resolution labels. What's the minimum vault coverage needed for a domain
   to be trainable?

5. **Non-binary resolutions:** How to handle markets that resolve to a scalar
   (not YES/NO)? Regression head on the GNN? Separate model?
