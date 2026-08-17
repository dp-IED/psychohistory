---
timestamp: 20260521-044449
p_yes: 0.07
reasoning: 'A Hantavirus outbreak reaching 100 confirmed cases in 2026 is unlikely:
  the MV Hondius outbreak (Andes virus) produced only 9 lab-confirmed cases, the international
  response has been aggressive, WHO risk assessment is low, and no single defined
  hantavirus outbreak has ever reached 100 confirmed cases in the Americas or Europe
  (largest: Four Corners 1993 with 48 nationwide; Yosemite 2012 and MV Hondius both
  ~9 cases). The resolution criteria ambiguity (cumulative global totals vs. single
  outbreak) adds ~2% tail risk, and ~5% reflects the possibility of larger secondary
  transmission from the dispersed MV Hondius contacts or a separate South American
  outbreak in the remaining 7 months.'
cutoff: '2026-05-21'
source: metaculus-cup
question_id: test-hantavirus
pit_context: "=== PIT RESEARCH BRIEF (librarian sub-agent) ===\nCutoff: 2026-05-21\n\
  Conjuncture: The vault has zero Hantavirus-specific content \u2014 no entity, thread,\
  \ or case count data exists for this pathogen. This i"
---

Will a Hantavirus outbreak with over 100 confirmed cases be reported in 2026? 
The outbreak must be confirmed by WHO or a national health authority.
Resolution criteria: At least 100 laboratory-confirmed cases of Hantavirus reported cumulatively in 2026 by an official source.

A Hantavirus outbreak reaching 100 confirmed cases in 2026 is unlikely: the MV Hondius outbreak (Andes virus) produced only 9 lab-confirmed cases, the international response has been aggressive, WHO risk assessment is low, and no single defined hantavirus outbreak has ever reached 100 confirmed cases in the Americas or Europe (largest: Four Corners 1993 with 48 nationwide; Yosemite 2012 and MV Hondius both ~9 cases). The resolution criteria ambiguity (cumulative global totals vs. single outbreak) adds ~2% tail risk, and ~5% reflects the possibility of larger secondary transmission from the dispersed MV Hondius contacts or a separate South American outbreak in the remaining 7 months.

## PIT Context

=== PIT RESEARCH BRIEF (librarian sub-agent) ===
Cutoff: 2026-05-21
Conjuncture: The vault has zero Hantavirus-specific content — no entity, thread, or case count data exists for this pathogen. This is partly a vault coverage gap (the health domain is respiratory/agricultural-spillover focused: H5N1, flu, RSV, COVID) and partly a signal: the 2026-Q1 timeline, which covers geopolitics, economics, tech, and public health events, records no Hantavirus outbreak or unusual case activity. The vault's applicable frameworks (zoonotic outbreak case count forecasting, outbreak escalation stages, government confirmation requirement) all predict that without human-to-human transmission — which is absent for most Hantavirus strains — case counts in a single spillover event would plateau at tens to low hundreds. However, Hantavirus is endemic across multiple continents with annual case loads that can reach 100+ cumulatively in endemic regions without constituting a single 'outbreak,' creating a definitional ambiguity in the resolution criteria.
Key events (≤ cutoff):
  - 2026-Q1 timeline records no Hantavirus outbreak or unusual epidemiological signal
  - Vault contains zero files mentioning 'hanta' or 'Hantavirus' across all 150+ indexed files
  - Health domain scope is respiratory (FluView, ILINet, H5N1) — Hantavirus falls outside primary surveillance coverage
Active threads:
  - domains/health/concepts/outbreak-escalation/_concept (general escalation stages framework)
  - domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept (plateau model for spillover outbreaks)
  - domains/global/concepts/government-confirmation-requirement/_concept (Type C technical-fact confirmation dynamic for notifiable diseases)
  - domains/global/procedures/outbreak-case-threshold-forecast (calibration procedure)
Mechanisms / concepts:
  - Zoonotic spillover plateau model: Without H2H transmission, case counts in spillover events plateau at tens to low hundreds — directly applicable to most Hantavirus strains (Andes virus is the exception with limited H2H)
  - Government confirmation requirement: WHO or national health authority confirmation of notifiable diseases like Hantavirus is Type C (technical fact with high confirmation probability if cases occur)
  - Outbreak escalation stages: Hantavirus is a known endemic pathogen, not a novel detection — Stage 1 'sporadic detection' base rate of P(escalation) < 0.10 does not apply directly
  - Global endemic baseline: Hantavirus circulates at low levels across multiple continents (Americas, Europe, Asia) with periodic outbreak years — China alone reports thousands of HFRS cases annually in non-outbreak years, creating a different case count profile from novel agricultural spillover pathogens like H5N1
Still uncertain at cutoff (do not treat as resolved):
  - No vault data on Hantavirus surveillance or case counts for any period in 2026 (Q1 timeline lacks Hantavirus mention; no Q2 timeline exists)
  - Definitional ambiguity: 'an outbreak with over 100 confirmed cases reported cumulatively in 2026' could mean a single geographic cluster, or the sum of all Hantavirus cases reported globally in 2026 by any official source — the vault provides no guidance on how resolution criteria handle endemic pathogens with routine low-level cases
  - No epidemiological baseline data in vault: absence of Hantavirus annual case rate data by region makes it impossible to assess whether 100 cases is above the normal seasonal baseline or within it for any specific endemic region
  - Hantavirus surveillance quality and reporting lags vary significantly by country — the vault has no entity files for PAHO, ECDC, or China CDC that might track Hantavirus surveillance systems
  - Whether the 'WHO or national health authority' confirmation criterion requires a specific outbreak declaration or merely cumulative reporting of notifiable disease cases is not specified
Excluded or truncated (post-cutoff leakage prevented):
  - Any

## Cross-References

- [[domains/health/entities/hantavirus]] — Pathogen entity with strain table and historical base rates
- [[events/mv-hondius-hantavirus-outbreak-2026]] — MV Hondius outbreak event file
- [[domains/health/concepts/transport-vector-outbreak]] — Shipborne outbreak dynamics concept
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting]] — Spillover plateau model
- [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius]] — Related run: off-ship cases (p_yes=0.2)
- [[domains/global/concepts/endemic-pathogen-criteria-ambiguity/_concept]] — General framework for this ambiguity class
