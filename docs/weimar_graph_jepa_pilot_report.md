# Weimar Graph-JEPA pilot report

Worktree: `/Users/darenpalmer/conductor/workspaces/psychohistory-v2/weimar-graph-jepa-pilot`
Branch: `dp-IED/weimar-graph-jepa-pilot`

## Purpose

Fast diagnostic for whether a Graph-JEPA-style masked-domain objective is immediately promising for a PIT historical representation layer over Weimar Republic data spanning three domains:

- Economic: inflation/currency stress, unemployment/industrial stress, foreign-credit/fiscal fragility.
- Cultural: avant-garde/media intensity, nationalist cultural backlash, public-sphere polarization.
- Socio-political: street violence, coalition fragility/emergency rule, extremist electoral strength.

This is not a final Weimar data product. It is a deliberately small, replayable pilot using curated ordinal yearly states to test representation behavior before investing in a larger ingestion/training path.

## Implementation

Added:

- `baselines/weimar_graph_jepa_pilot.py`
- `tests/test_weimar_graph_jepa_pilot.py`

Artifacts written by the run:

- `data/runs/weimar_graph_jepa_pilot/summary.json`
- `data/runs/weimar_graph_jepa_pilot/eval_rows.csv`
- `data/runs/weimar_graph_jepa_pilot/weimar_states.jsonl`

The pilot constructs yearly PIT states for 1919-1933. For each as-of year and target domain, it masks that domain in the current-year graph and trains a small non-contrastive JEPA-like context encoder to predict a stopped latent projection of the next-year target-domain state.

Train/eval split:

- Train as-of years: 1919-1929, 33 examples.
- Eval as-of years: 1930-1932, 9 examples.
- Eval targets: 1931-1933, intentionally stressing late collapse years not represented as a train target label.

Baselines:

- Persistence: next-year domain state = current same-domain state.
- Ridge: linear predictor from masked graph context to target-domain vector.
- Graph-JEPA pilot: masked graph context -> latent prediction; evaluated by cosine to stopped target latent and nearest-neighbor regime retrieval from train target bank.

## Commands run

```bash
PYTHONPATH=. pytest -q tests/test_weimar_graph_jepa_pilot.py
PYTHONPATH=. python -m baselines.weimar_graph_jepa_pilot --out-dir data/runs/weimar_graph_jepa_pilot
```

Test result:

- `2 passed`

## Results

Overall latent cosine to actual next-year target latent:

| model | cosine |
|---|---:|
| Persistence | 0.9985 |
| Ridge | 0.9891 |
| Graph-JEPA pilot | 0.9815 |

By domain:

| domain | Graph-JEPA | Ridge | Persistence | JEPA train-bank regime match |
|---|---:|---:|---:|---:|
| economic | 0.9981 | 0.9988 | 0.9999 | 0.00 |
| cultural | 0.9533 | 0.9701 | 0.9961 | 0.00 |
| socio-political | 0.9931 | 0.9983 | 0.9995 | 0.00 |

By target year:

| target year | Graph-JEPA | Ridge | Persistence | JEPA train-bank regime match |
|---|---:|---:|---:|---:|
| 1931 | 0.9914 | 0.9958 | 0.9992 | 0.00 |
| 1932 | 0.9852 | 0.9930 | 0.9993 | 0.00 |
| 1933 | 0.9679 | 0.9783 | 0.9970 | 0.00 |

JEPA loss converged cleanly from `0.1796` to `0.00125`, so the negative result is not a training failure. The objective fits the training examples but does not beat simpler baselines on the held-out collapse slice.

## Interpretation

The pilot argues against replacing the current architecture wholesale with Graph-JEPA right now.

Main finding:

- On this small Weimar tri-domain setup, Graph-JEPA learns a smooth latent trajectory, but the historical signal is dominated by strong temporal persistence and monotonic crisis escalation. A simple persistence baseline beats both Graph-JEPA and ridge on latent cosine.

Domain-specific reading:

1. Economic

Graph-JEPA is close to ridge and persistence on economic stress, but it does not add much. The trajectory from 1930 to 1932 is highly autocorrelated: unemployment/credit/currency stress stay high or intensify. Persistence is therefore an unusually strong baseline.

2. Cultural

This is the weakest Graph-JEPA domain. The 1933 cultural target changes directionally: avant-garde/media intensity collapses while nationalist backlash and polarization saturate. The masked-domain JEPA objective smooths this transition and retrieves stabilization-era cultural neighbors, which is historically wrong for 1933. This directly mirrors the existing project concern that culture extraction/reasoning is weaker and more prone to unsupported background priors.

3. Socio-political

Graph-JEPA captures broad escalation but not the regime transition. Nearest-neighbor retrieval tends to map late Weimar crisis to earlier crisis/reentry or stabilization patterns rather than recognizing collapse as structurally new. That is a warning against relying on latent similarity alone for political regime-break forecasting.

## Important caveats

- The Weimar states are curated ordinal intensities, not externally measured source-tier data.
- The train/eval split is tiny.
- Cosine over a fixed random stopped target projection is a representation diagnostic, not a forecast metric.
- The nearest-neighbor regime match is harsh because collapse labels are absent from the training target bank; however, that is also exactly the hard problem: recognizing novel regime breaks rather than analogizing them away.
- This pilot does not test a full graph neural network over entity/event nodes. It tests the core JEPA-style masked-domain predictive representation idea under a minimal setup.

## Recommendation

Do not migrate to Graph-JEPA as the primary approach yet.

Use Graph-JEPA only as an ablatable candidate inside `R(q, h_t)`, and only after stronger deterministic baselines are in place. For Weimar-style cases, the next useful version would need:

1. Real PIT source-tier construction instead of hand-curated yearly intensities.
2. A regime-break objective or evaluation probe, not only smooth next-state latent prediction.
3. Domain-conditioned masking, especially for culture, where smooth latent similarity fails hardest.
4. Explicit comparison against persistence, ridge/linear, and deterministic retrieval baselines every time.
5. Evidence-carrying output, not just latent vectors, so the LLM can audit why 1933 is not merely another high-stress year.

Bottom line:

Graph-JEPA remains strategically interesting for learned `R(q, h_t)`, but this pilot supports a conservative path: keep the current evidence-constrained architecture, add Graph-JEPA as a sidecar retrieval/representation ablation, and require it to beat simple temporal baselines before promoting it.
