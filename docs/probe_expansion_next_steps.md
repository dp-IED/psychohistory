# Probe Expansion Next Steps

## Purpose

Expand strict-objective coverage only where it improves the training signal.
Probe expansion should follow the graph artifact/export contract, not precede
it.

## Priority Order

1. Keep the current six-probe strict set stable.
2. Add one more long-run historical family with seed entities and persistence
   tasks.
3. Add one more contemporary event-stream family after Wikidata grounding is
   stable.
4. Only then expand to the full probe pack.

## Long-Run Candidate Probes

Good next candidates:

- Roman recovery/reconfiguration
- Mughal expansion
- Mughal fragmentation
- British India / anticolonial partition
- Chile rise and fall

Each should receive:

- `seed_entities`
- `golden_tasks`
- `designated_ablations`
- `complexity_budget`
- at least one persistence or transmission task if long-run legacy is part of
  the probe’s purpose

## Contemporary Candidate Probes

Good next candidates:

- DRC resource extraction and external constraint
- Burkina Faso / Sankara rollback
- Burkina post-2014 Sahel crisis

Each should receive:

- external actor seed entities
- conflict/protest/extraction seed entities
- event-stream-ready task labels
- a counterweight ablation

## Acceptance Checks

- New probe has non-zero `source_G`.
- New probe has at least one path whose `task_path_grounding_rate` is non-zero.
- Adding the probe does not break strict eval for the existing pilot set.
- The probe contributes a distinct training task, not just more entity names.

## Expansion Anti-Patterns

- Adding seed entities with no golden-task path that uses them.
- Adding strict fields only to raise coverage counts.
- Adding future-outcome targets without explicit cutoff dates.
- Modeling long-run influence as one direct causal edge.
