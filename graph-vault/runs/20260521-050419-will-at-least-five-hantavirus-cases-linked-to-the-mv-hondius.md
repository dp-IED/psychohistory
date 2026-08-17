---
timestamp: 20260521-050419
p_yes: 0.2
reasoning: As of cutoff (May 21, 2026) there are zero confirmed off-ship cases; only
  contacts under monitoring in Scotland/Wales and 12 Dutch HCWs in quarantine. The
  Andes strain's documented H2H transmission and the 6-week incubation window mean
  some off-ship cases may still emerge, but the WHO's explicit 'no sign of larger
  outbreak' assessment, aggressive global contact tracing and 42-day quarantines,
  and hantavirus's requirement for close/prolonged contact make 5+ secondary cases
  by Aug 1 unlikely.
cutoff: '2026-05-21'
source: metaculus-cup
question_id: test-hondius-offship
pit_context: "=== PIT RESEARCH BRIEF (librarian sub-agent) ===\nCutoff: 2026-05-21\n\
  Conjuncture: The vault has ZERO coverage of the MV Hondius hantavirus outbreak \u2014\
  \ a critical gap. The health domain covers respirator"
---

Will at least five hantavirus cases linked to the MV Hondius outbreak be reported in people who were not aboard the vessel, before August 1, 2026?

Resolution criteria: This question will resolve as Yes if, before August 1, 2026, at least five hantavirus cases are publicly reported in people who were not aboard the MV Hondius during its March-April 2026 voyage, and the cases are linked to the MV Hondius outbreak by a public health authority.

Fine print: A qualifying case must be either laboratory-confirmed or described as a probable/suspected hantavirus case linked to the MV Hondius outbreak. The case does not need to be conclusively proven to result from human-to-human transmission.

As of cutoff (May 21, 2026) there are zero confirmed off-ship cases; only contacts under monitoring in Scotland/Wales and 12 Dutch HCWs in quarantine. The Andes strain's documented H2H transmission and the 6-week incubation window mean some off-ship cases may still emerge, but the WHO's explicit 'no sign of larger outbreak' assessment, aggressive global contact tracing and 42-day quarantines, and hantavirus's requirement for close/prolonged contact make 5+ secondary cases by Aug 1 unlikely.

## PIT Context

=== PIT RESEARCH BRIEF (librarian sub-agent) ===
Cutoff: 2026-05-21
Conjuncture: The vault has ZERO coverage of the MV Hondius hantavirus outbreak — a critical gap. The health domain covers respiratory diseases (COVID-19, influenza, RSV, H5N1) but has no hantavirus-specific content, entity stubs, or outbreak threads. The timeline ends at 2026-Q1 (March 31, 2026) and does not mention Hondius or hantavirus. Since the MV Hondius voyage occurred in March-April 2026, the outbreak timeline is entirely within 2026-Q2 — a quarter with no timeline file. At cutoff (May 21, 2026), the outbreak would be in its early detection phase: hantavirus incubation periods range from 1-6 weeks (strain-dependent: Andes virus ~2-4 wks, Seoul virus ~1-2 wks, Sin Nombre virus ~2-4 wks) plus diagnostic confirmation lag, meaning the index cluster on the vessel was likely only recently confirmed, and any secondary transmission events (if occurring) would be even more recent and difficult to detect.
Key events (≤ cutoff):
  - No vault file covers the MV Hondius voyage or hantavirus outbreak.
  - The 2026-Q1 timeline ends March 31, 2026 — before the MV Hondius voyage concluded.
  - No 2026-Q2 timeline exists in the vault.
  - No hantavirus entity, thread, or concept file exists in the health domain.
Active threads:
  - None directly relevant; the closest structural analogs are:
  - domains/global/threads/h5n1-avian-influenza-outbreak/_thread (agricultural spillover plateau model, but designed for respiratory/avian pathogens, not rodent-borne hantavirus)
  - domains/health/threads/respiratory-season-2025-26/_thread (respiratory season tracker, not applicable to hantavirus)
  - domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept (plateau model, but hantavirus has different ecology: rodent reservoir, aerosolized excreta transmission)
  - domains/health/concepts/outbreak-escalation/_concept (sporadic→local→regional→pandemic framework)
Mechanisms / concepts:
  - domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept — agricultural spillover plateau model; limited transferability because hantavirus is rodent-borne not agricultural, and incubates differently
  - domains/health/concepts/outbreak-escalation/_concept — staged outbreak escalation framework; useful for situating the outbreak along a progression spectrum but assumes respiratory transmission
  - domains/global/concepts/forecast-resolution-criteria-gotchas — relevant for checking resolution criteria ambiguity: the question defines 'linked by a public health authority' which resolves the attribution question but leaves ambiguity about what constitutes 'probable/suspected' in different jurisdictions
  - domains/global/procedures/outbreak-case-threshold-forecast — procedure for case count thresholds; designed for US CDC-tracked outbreaks with established case counters; may need adaptation for a multinational cruise ship outbreak
Still uncertain at cutoff (do not treat as resolved):
  - Zero vault signal: no baseline case count from MV Hondius voyage exists in any file
  - Hantavirus strain involved is unknown from vault content (Andes virus has known H2H potential; Seoul virus is rare; Sin Nombre virus is North American — strain determines secondary transmission risk)
  - Number of passengers and crew aboard the MV Hondius is undocumented in the vault
  - Number of onboard cases (index cluster size) is undocumented
  - Whether any public health authority has issued a statement linking secondary cases to the outbreak as of May 21, 2026, is undocumented
  - Geographic distribution of port calls and potential exposure sites is undocumented
  - The outbreak's timing (voyage March-April 2026, cutoff May 21, 2026) means it was likely in Weeks 3-8 since first detection — still in the early amplification phase for a zoonotic spillover event, but hantavirus incubation and diagnostic lag could mean the outbreak was only recognized very recently at cutoff
  - Which specific publication or authority has confirmed the 9 index cases is undocumented

## Cross-References

- [[domains/health/entities/hantavirus]] — Pathogen entity with strain table and Andes virus H2H risk assessment
- [[events/mv-hondius-hantavirus-outbreak-2026]] — MV Hondius outbreak event file (same outbreak)
- [[domains/health/concepts/transport-vector-outbreak]] — Shipborne outbreak dynamics concept (canonical case)
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting]] — Agricultural spillover plateau model (limited transferability to hantavirus, noted in run)
- [[runs/20260521-044449-will-a-hantavirus-outbreak-with-over-100-confirmed-cases-be-]] — Related run: 100+ cases in 2026 (p_yes=0.07)
- [[runs/20260521-184716-will-at-least-5-non-passengers-be-linked-to-the-mv-hondius-h]] — Sibling run: same question from metaculus-tournament (p_yes=0.24, qid=43465)