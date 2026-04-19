# Graph Builder Contract v0.1

**Status:** Locked program decisions for builder, assumptions, and training context. Implementation choices (embeddings, ANN library, MLP shapes) are **out of scope** until this contract is stable.

**Related:** [`project.md`](../project.md), [`roadmap.md`](../roadmap.md), [`next_steps.md`](../next_steps.md), [`docs/research/architecture.md`](research/architecture.md), [`forecast_charter.md`](../forecast_charter.md).

---

## Purpose

The graph builder is the **core research contribution** of this system. It is a **query-conditioned sparse retriever** that proposes a **fixed-budget subgraph** from a historical warehouse. The forecaster is a **downstream consumer** of this subgraph. All architectural decisions below are **compute-honest** for a **MacBook Pro** class development environment.

---

## Retrieval unit (locked)

**Three-tier hierarchy** — all three types are first-class nodes in the subgraph (architecture reserves slots from day one; **v1 implements actor-states only**).

| Tier | Definition | Primary role |
|------|------------|--------------|
| **Actor-state** | A typed entity (person, group, institution, region) paired with a **behavioural state** at a time slice | **Current-query relevance** |
| **Trend thread** | A named **slow-moving dynamic** active across multiple time windows | **Long-running signal** |
| **Historical analogue** | A **structural template** from past history that rhymes with the current query context | **Assumption priming** |

---

## Graph builder role (locked)

The builder is **not** a full graph generator. It is a **two-stage retriever**:

1. **ANN retrieval** — given a query vector, fetch **top-100** candidate nodes from a **precomputed warehouse index**.
2. **Reranker + edge enricher** — score query–candidate and candidate–candidate pairs, select **top-50** nodes and assign edge weights.

**Output:** a **fixed-budget induced subgraph**: **max 50 nodes**, **max 200 edges**. Shape is **static** at inference. **No dynamic graph construction** at query time.

---

## Supervision (locked)

**Two signals**, trained **in stages**:

| Stage | Mode | Description |
|-------|------|-------------|
| **Stage 1** | **Self-supervised** | Train from **GDELT + ACLED tape alone**. Rewards retrieval of nodes that **co-evolve**, **persist**, or **causally precede** each other. Signals: temporal co-occurrence, Granger-style lead–lag, cross-source corroboration. |
| **Stage 2** | **Weak supervised** | **Forecast improvement** as retrieval label: a candidate is a **positive** if including it in the subgraph **measurably improves** forecast loss on **held-out** slices. **No human labels.** |

**Stage 3** — **Polymarket / Kalshi** (and peers) enter for **forecaster calibration** and late-stage gate validation. They **do not supervise** the graph builder.

---

## Assumption layer (locked)

Assumptions are **soft gates**, not class labels and not free-form latent vectors.

**Five named gates**, each a scalar in \([0,1]\), from a lightweight MLP over the retrieved subgraph:

| Gate | Meaning |
|------|---------|
| **Persistence** | This actor-state or trend is **ongoing**, not transient |
| **Propagation** | This state is **spreading** to adjacent actors or regions with a lag |
| **Precursor** | **Weak early signal** before a larger event |
| **Suppression** | One actor-state is **masking or dampening** detection of another |
| **Coordination** | Multiple actors are moving **non-independently** |

- **Training:** gates are **soft**.
- **Inference:** **hard-thresholded**.
- The forecaster’s attention is **modulated** by gate values.
- **No explicit labels** on gates — supervision is **forecast improvement when a gate fires** (together with the rest of the stack).

The five types are an **architectural prior**, not a hand-labeled schema; the model learns **when** to invoke each.

---

## Data sources (locked)

| Source | Role | Stage |
|--------|------|--------|
| **GDELT** | Trend-thread signal path, self-supervised builder training, historical-analogue retrieval | **1** |
| **ACLED** | Actor-state grounding; weak-supervision positives for retriever | **1–2** |
| **Polymarket / Kalshi** | Forecaster calibration; late-stage gate validation | **3** |

---

## Compute constraints (locked)

Hard limits for laptop-class development:

| Constraint | Value |
|------------|--------|
| Subgraph | **≤ 50** nodes, **≤ 200** edges |
| Node embedding dim | **128** or **256** |
| ANN pool | **Top-100** retrieved → rerank to **top-50** |
| Node embeddings | **Precomputed offline**, **memory-mapped** |
| Query-time graph | **No** dynamic graph construction; **fixed-budget** shapes only |

---

## Training contexts (locked)

| Context | Role |
|---------|------|
| **France protests** | **Validation harness only** — not a primary **training** context (too seasonal / too easy for builder learning claims). Still allowed as **regression smoke** when pipelines change (`next_steps.md`). |
| **Arab Spring (2010–2013)** | **Primary training** — cross-regional propagation, high outcome variance |
| **European sovereign debt crisis (2009–2015)** | **Primary training** — institutional actors, long slow buildup |
| **One Latin American commodity-politics sequence** | **Primary training** — geography/period **TBD** |

All three primary sequences are preprocessed into **offline node embedding stores**. The builder trains **across** them with **fixed-budget** constraints active from day one.

---

## Encoder path (implementation): dual eval

The **primary** v0 stack is **query-conditioned retrieval → fixed-budget subgraph** (≤50 nodes, ≤200 edges) encoded by a **bag encoder** consuming the tensor contract in code (`schemas/graph_builder_retrieval.py`: `RetrievedGraphBatch`).

For **ablations and reviewer-facing comparisons**, keep an **optional parallel branch** on the **same** \((t, \text{labels})\) rows: the existing deterministic **full weekly snapshot** path (`ingest/snapshot_export` → `baselines.gnn.build_graph_from_snapshot` → `HeteroGNNModel`). That branch is **not** the primary research encoder once the learned builder is active; it exists so subgraph-policy vs encoder effects can be separated without conflating a new retrieval interface with legacy graph construction.

---

## Not decided yet (implementation)

These follow after the contract is stable:

- Node embedding scheme (vector contents from GDELT + ACLED)
- Specific ANN library and index format
- Reranker architecture (MLP vs cross-attention)
- Gate MLP width/depth
- Forecaster head architecture

---

## Revision history

| Version | Date | Notes |
|---------|------|--------|
| v0.1 | 2026-04-18 | Initial locked decisions |
| v0.1.1 | 2026-04-18 | Dual eval: bag encoder vs legacy full-graph baseline (`schemas/graph_builder_retrieval`, `baselines/gnn`) |
