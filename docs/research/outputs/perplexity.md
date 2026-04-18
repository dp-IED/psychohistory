# Deep research: Psychohistory forecasting system

> Synthesis aligned with [`../architecture.md`](../architecture.md), [`../research.md`](../research.md), and program docs.

**Problem summary:** relational situation graphs + latent world model (multi-step dynamics, mixture/hypothesis competition), not a generic RAG. Markets = training curriculum and optional context; many questions are not listed contracts. Outputs: material forecasts + epistemological ranked hypotheses + evidence + HITL. Discovery via held-out tests, ablations, stability—not fixed narrative priors. **Architecture (pipeline):** evidence store → situation $S_t$ → query lens → graph encoder → world model → task heads → constrained Q&A.

---

The core thesis is that **relational situation structure** (heterogeneous graph) and **latent temporal dynamics** (world model) are **composable** layers—with prediction markets as dense optional training signal and epistemological outputs as a **separate** evaluation tier.[^1][^2]

## 1. Graph encoder and world model fusion

The design problem is wiring $h_t$ (encoder output) into $z_t$ (world-model state) over time without gradient interference or mis-assigned credit.

**Graph-RSSM (“Graph Dreamer”, NeurIPS 2025)** — Berkes et al., *Graph Dreamer: Temporal Graph World Models for Sample-Efficient and Generalisable Reinforcement Learning* — is a direct blueprint: **time-then-space** (per-node temporal GRU, then relation-aware message passing), permutation-equivariant dynamics, variational per-node latents.[^3]

- **Separate recurrence axes:** temporal (GRU/SSM) before message passing—not interleaved.
- **Multi-horizon losses:** Dreamer-style $\lambda$-returns; auxiliary SSL (masked nodes, temporal contrast) when labels are thin.[^1]
- **Exogenous channels:** pooled graph summary plus exogenous inputs (e.g. market quotes) for prediction heads—analogous to Layer 4 emissions.[^3]

**Identifiability:** Manenti et al. show that minimizing **point-prediction** losses alone does not guarantee learning of latent relational structure and its uncertainty; **stochastic** losses on model outputs are needed for joint calibration of structure and prediction.[^4]

## 2. Learned graph structure under PIT constraints

Layer 1 builds $S_t = \mathrm{Build}(E_{\leq t})$ with no leakage; learned soft edges can still complete sparse graphs.[^1]

**Risk:** fully learned adjacency (MTGNN-, NRI-style) $\rightarrow$ **correlation soup** and shortcut learning.

**Mitigations:**

- **GTGIB** (Xiong & Sakellariou, arXiv:2508.14859): temporal graph information bottleneck—regularize edges and features; compatible with PIT (only information up to $t$).[^5]
- **Stochastic structure** (DGM, NRI; theory in Manenti et al.): $q(A \mid X_{\leq t})$, ELBO—propagates structural uncertainty.[^4]
- **Regularization:** sparsity, MI bottlenecks, temperature annealing for discrete relaxations.

**PIT contract:** `EvidenceStore.query(as_of=t)` stops raw leakage; learned completion must use **targets masked to $t$** so soft edges do not encode post-$t$ resolutions.[^1]

## 3. Query-conditioned graph views (lens)

$S_t^{(q)} = \mathrm{Lens}(q, S_t)$: **no invented facts**, **no hidden evidence dropping**.[^1]

| Method | Faithfulness | Evidence-drop risk | Latency |
|--------|----------------|--------------------|--------|
| Embedding similarity subgraph (G-Retriever) | Moderate | High (recall gaps on hard queries) | Fast |
| GNN score ranking (GNN-RAG) | Better | gaps vs. ground truth on some settings | Medium |
| Gated message passing + query attention | High if structure kept | Low if **masks logged** | Slower |

**Recommendation:** GNN-RAG-style **scored retrieval**[^8] plus **budgeted expansion** with a **log of masked nodes/edges**. **Iterative expansion:** if uncertainty under $S_t^{(q)}$ stays high, widen the lens—related to unified retrieval$\rightarrow$generation limits in **Reasoning by Exploration (RoE)** on graphs.[^6]

## 4. Prediction market integration

Markets = **dense PIT training** (belief paths, structure, resolutions), not mandatory inference oracles.[^7][^2]

- **Resolution labels:** only when `resolved_at` and paths respect cutoff; no conditioning on future belief when building $S_t$.
- **Short-horizon head:** e.g. $p_{t+1}$ from $(z_t, h_t)$ only—separate head, explicit non-leakage guard.[^1]
- **Cross-market edges** in the hetero graph enrich context without requiring a market at inference.

**Baselines:** third-party reports cite aggregate Polymarket Brier on the order of **0.19** and stronger scores on restricted subsets—treat as **external** baselines; claim lift in **multi-entity** and **masked (non-market)** settings.[^9][^10]

**Selection bias:** listed topics $\neq$ random; **masking ablations** and train-with/without-market comparisons isolate generalization.[^7] **Coverage:** explicitly report eval domains or regions with **no** listed markets—do not let the model **silently** underperform where supervision was never present (`forecast_charter.md` markets tier, `next_steps.md` Track M).

## 5. Epistemological outputs without full hand-labeling

**Mixture over experts:** $\pi_t \in \Delta^K$ over world experts $\approx$ competing hypotheses; **online mixture of experts** with no-regret aggregation (Liu & Etesami, NeurIPS 2025) is a formal reference for dynamic expert weighting.[^11]

**Probes:** linear probes on frozen $z_t$ vs. constructs; validate on **held-out** time/regions.[^2]

**HITL:** top-$K$ hypotheses + node citations; rubrics on groundedness, competing framings, calibration; annotator disagreement $\rightarrow$ uncertainty signal.

## 6. Bias, ethics, multi-community time

GDELT/ACLED **coverage bias:** mitigate with **source metadata**, provenance weighting, explicit `unknown` where coverage is thin.[^2]

**Clocks:** weekly snapshots privilege one rhythm; **multi-scale** break tests (e.g. BOCPD) and compare learned boundaries to **community-specific** vs. **national** calendars.[^2]

**Teleology:** discover boundaries first; **name** on held-out probes; avoid baking shocks into core priors.[^2]

---

## Architecture integration map

| Layer | Integration |
|-------|----------------|
| **L0 Evidence** | Source diversity in schema; PIT via `as_of=t`; edge targets capped at $t$ |
| **L1 Builder** | GTGIB-style regularization; stochastic adjacency (ELBO) |
| **L2 Lens** | GNN-RAG scores + budget + mask log; iterate on high uncertainty |
| **L3 Encoder** | Hetero GNN v1; optional graph transformer; feed L4 time-then-space |
| **L4 World model** | Graph-RSSM pattern; $z_t$; $\pi_t$ for hypotheses; market channel optional + masked |
| **L5 Heads** | Material; short-horizon market head (isolated); epistemic via mixture + probes + HITL |
| **L6 Q&A** | Routes to citations + heads; no solo forecasting; exclusions logged |

---

## Evaluation protocol

1. **Held-out time:** train on $[t_1, t_2)$, evaluate on $[t_2, t_3)$; Brier and baselines per horizon.
2. **Ablations:** encoder vs. encoder+WM; full graph vs. lens; markets on vs. masked; learned vs. fixed adjacency.
3. **Multi-seed:** $\geq 5$ seeds; mean $\pm$ std; report mixture assignment variance.
4. **Covariate checks:** learned switch times vs. hand-coded events (elections, spikes).
5. **Shared-code smoke (optional):** if the world model shares snapshot/backtest code with the France pipeline, rerun France to catch integration bugs—**not** as proof the WM is correct, only as wiring check.

---

## References

[^1]: [architecture.md](../architecture.md)

[^2]: [research.md](../research.md)

[^3]: Berkes, Vakalis, Bengio, Rolnick (2025). *Graph Dreamer: Temporal Graph World Models for Sample-Efficient and Generalisable Reinforcement Learning.* NeurIPS 2025. [OpenReview](https://openreview.net/forum?id=pHmgNUZixd) · [NeurIPS poster](https://neurips.cc/virtual/2025/poster/133755)

[^4]: Manenti, Zambon, Alippi (2024/2025). *Learning Latent Graph Structures and their Uncertainty.* arXiv:2405.19933 · ICML 2025. [OpenReview](https://openreview.net/forum?id=TMRh3ScSCb) · [arXiv](https://arxiv.org/abs/2405.19933) · [Code](https://github.com/allemanenti/Learning-Calibrated-Structures)

[^5]: Xiong, Sakellariou (2025). *Graph Structure Learning with Temporal Graph Information Bottleneck for Inductive Representation Learning.* arXiv:2508.14859. [arXiv](https://arxiv.org/abs/2508.14859)

[^6]: Han et al. (2025). *Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs.* arXiv:2510.07484. [arXiv](https://arxiv.org/abs/2510.07484) · [PDF](https://arxiv.org/pdf/2510.07484)

[^7]: [project.md](../../../project.md)

[^8]: Mavromatis, Karypis (2025). *GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs.* ACL Findings 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.856/) · [PDF](https://aclanthology.org/2025.findings-acl.856.pdf)

[^9]: Secondary calibration digest (use with care): [Scribd-hosted summary](https://www.scribd.com/document/1006343517/Prediction-markets-calibration)

[^10]: Third-party Polymarket analysis: [Fensory](https://www.fensory.com/intelligence/predict/polymarket-accuracy-analysis-track-record-2026)

[^11]: Liu, Etesami (2025). *Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making.* NeurIPS 2025. [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117331)

**Further reading:** Longa et al., *GNNs for temporal graphs* — [arXiv:2302.01018](https://arxiv.org/abs/2302.01018); TGL workshop — [2025](https://sites.google.com/view/tgl-workshop-2025/); graph + LLM survey (industry) — [Siemens](https://blogs.sw.siemens.com/thought-leadership/topics-in-ai-systems-part-i-gnn-and-llm-fusion/); ICML graph reading list — [GitHub](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024).
