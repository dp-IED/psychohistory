# Optimization target (graph IR autoresearch)

You are **searching** over a **graph intermediate representation (IR) schema** for a psychohistory-style forecasting system. The schema is **not** the final GNN; it is the typed vocabulary and constraints that graphs must satisfy before projection and training.

## Primary objective (what the harness actually scores)

The runner runs `evals` and computes a **composite score** in `[0, 1]` from:

- **Structural gate** — All probe-declared `must_represent` node/edge type strings are registered in the schema.
- **Functional gate** — Retrieval-task answer shapes reference node types that exist (lightweight check).
- **Constraints gate** — Projection validity, epistemic anti-collapse rules, persistence-hook consistency when probes require it.
- **Stub block** — Placeholder metrics (GDELT, traversal QA, Polymarket, persistence ablation) — **currently near zero** until wired.

**Your job is exploratory: propose schema changes that maximize this composite and keep the IR honest.** When boolean gates are already satisfied, the score can plateau until stub metrics are implemented; exploration should still improve **constraint quality**, **typing discipline**, and **extensions/metadata** that future metrics can consume.

## Secondary objectives (design intent — do not sacrifice for a cheap numeric win)

1. **Expressiveness** — Causal, institutional, ideological, contested structure without collapsing nuance.
2. **Fixed projection** — Explicit finite-dimensional projection rules for downstream GNNs.
3. **Retrieval** — Probe `probe_tasks` shapes remain representable as typed structure.
4. **Epistemic honesty** — Confidence, contention, provenance, perspective stay first-class.

## Anti-objective

Do not optimize by **collapsing** contested claims into a single cluster or untyped blobs. Do not delete or rename probe-required type strings to “pass” checks — that is **cheating the benchmark**, not improving the IR.

## Benchmarks

- Probes under `probes/` are **frozen**. Do not edit probe YAML to fit your schema.
- The seed registry already includes a large type union; your exploration is mostly **how** types are specified (layers, epistemics, temporal defaults, `extensions`), not whether to delete required names.

## Exploration strategy (recommended)

- Read **`autoresearch/experiments/log.jsonl`** and any **`eval_report.json`** under `candidates/` — learn what failed (`failure_class`, constraint issues).
- **Vary hypotheses** across iterations: e.g. tighten `EpistemicSpec` on `Claim`, adjust `ProjectionSpec` documentation or per-type hints, refine layer assignments, add structured `extensions` for retrieval or provenance without breaking pydantic.
- Prefer **measurable** rationale: “expect constraints_ok to stay true while improving X” rather than vague prose.
- If composite is flat at ~0.85 with all gates true, state that in `rationale` and explore **quality / robustness** (e.g. anti-collapse, persistence clarity) anyway.

## Deliverable each iteration

Emit **one** JSON proposal (see `autoresearch/agent_interface.py`): `rationale`, `file_patch_plan`, `expected_metric_impact`, `risks`. The harness applies it in a sandbox, runs eval, logs scores, keeps the best candidate.
