# Reflector is a named agent Parent starts on reflect ticks

Open-ended evolution needs a dedicated evolution role. **Parent** still means: this host job runs exactly one tick (ADR 0001, one job per run).

**Predict / discover:** Parent follows those skills (and may spawn claim-workers).

**Reflect:** Parent starts **reflector** (`agents/reflector.md`). The reflector grades the series and writes overlay: tiny live edits, `exp/` experiments, analog cards (ADR 0007, 0013, 0016).

The reflector does not become the daily traffic cop. The harness still starts the three jobs.

Does not add a fourth host job. Does not restore an in-repo orchestrator.
