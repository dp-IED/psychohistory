---
timestamp: 20260521-184716
p_yes: 0.24
reasoning: "[cognitive] p_yes=0.240 Base rate anchor: 36% (appointment/health category,\
  \ weakly matched). Three sub-agents converged at 0.18-0.20, but per protocol, their\
  \ close agreement (within 0.02) may indicate shared bias from over-indexing on the\
  \ '0 confirmed non-passenger cases as of May 21' signal, so the base rate receives\
  \ increased weight. My synthesis: Andes virus H2H transmission is documented but\
  \ very inefficient \u2014 the ship itself produced only ~7 cases from the index\
  \ case among 147 people in extremely confined quarters over weeks. Non-passenger\
  \ contacts have far less exposure intensity. ZERO non-passenger cases confirmed\
  \ after 17+ days of active multinational surveillance (12+ countries testing contacts),\
  \ and several high-profile potential cases (KLM attendant, Singapore residents,\
  \ US passenger) tested negative. However, ~30 passengers who left at Saint Helena\
  \ on April 24 had ~10 days in the community before outbreak detection, creating\
  \ the highest-risk window for household transmission. Incubation period (1-7 weeks)\
  \ means cases from April exposures could still emerge into June. Expected secondary\
  \ cases: ~2-4 from household contacts (Saint Helena group), ~0-1 from healthcare\
  \ workers, ~0-1 from flight contacts \u2014 below the 5-case threshold but close\
  \ enough that tail risk is non-trivial. Weighted blend: 0.35*0.36 (base) + 0.25*0.18\
  \ (causal) + 0.15*0.20 (analogical) + 0.25*0.18 (narrative) + ~0.02 disconfirmation\
  \ adjustment = 0.24."
cutoff: '2026-05-21'
source: metaculus-tournament
question_id: '43465'
pit_context: '=== PIT RESEARCH BRIEF (librarian sub-agent) ===

  Cutoff: 2026-05-21

  Conjuncture: Three forces interact at cutoff: (1) Andes virus is the only hantavirus
  with confirmed human-to-human transmission, mak'
---

Will at least 5 non-passengers be linked to the MV Hondius hantavirus outbreak before August 2026?

In early May 2026, WHO [reported](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599) a cluster of hantavirus infections associated with the cruise ship [MV *Hondius*](https://en.wikipedia.org/wiki/MV_Hondius_hantavirus_outbreak), with seven confirmed or suspected cases and three deaths. The vessel carried [147 passengers and crew](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599), and illness onset among reported cases ranged from April 6 to April 28, 2026. WHO noted that hantavirus is usually acquired from infected rodents, but limited human-to-human transmission has previously been reported for Andes virus. Public health authorities in multiple countries began monitoring passengers and contacts after the outbreak.

Resolution criteria: This question will resolve as **Yes** if, before August 1, 2026, at least five hantavirus cases are publicly reported in people who were not aboard the *MV Hondius* during its March-April 2026 voyage, and the cases are linked to the *MV Hondius* outbreak by a public health authority.

Fine print: A qualifying case must be either laboratory-confirmed or described as a probable/suspected hantavirus case linked to the *MV Hondius* outbreak. The case does not need to be conclusively proven to result from human-to-human transmission.

[cognitive] p_yes=0.240 Base rate anchor: 36% (appointment/health category, weakly matched). Three sub-agents converged at 0.18-0.20, but per protocol, their close agreement (within 0.02) may indicate shared bias from over-indexing on the '0 confirmed non-passenger cases as of May 21' signal, so the base rate receives increased weight. My synthesis: Andes virus H2H transmission is documented but very inefficient — the ship itself produced only ~7 cases from the index case among 147 people in extremely confined quarters over weeks. Non-passenger contacts have far less exposure intensity. ZERO non-passenger cases confirmed after 17+ days of active multinational surveillance (12+ countries testing contacts), and several high-profile potential cases (KLM attendant, Singapore residents, US passenger) tested negative. However, ~30 passengers who left at Saint Helena on April 24 had ~10 days in the community before outbreak detection, creating the highest-risk window for household transmission. Incubation period (1-7 weeks) means cases from April exposures could still emerge into June. Expected secondary cases: ~2-4 from household contacts (Saint Helena group), ~0-1 from healthcare workers, ~0-1 from flight contacts — below the 5-case threshold but close enough that tail risk is non-trivial. Weighted blend: 0.35*0.36 (base) + 0.25*0.18 (causal) + 0.15*0.20 (analogical) + 0.25*0.18 (narrative) + ~0.02 disconfirmation adjustment = 0.24.

## PIT Context

=== PIT RESEARCH BRIEF (librarian sub-agent) ===
Cutoff: 2026-05-21
Conjuncture: Three forces interact at cutoff: (1) Andes virus is the only hantavirus with confirmed human-to-human transmission, making this a structurally different risk from non-H2H hantavirus strains; (2) MV Hondius is the first documented shipborne hantavirus outbreak, creating a transport-vector two-phase pattern where the index cluster (7 confirmed/suspected cases, 3 deaths among 147 passengers/crew) has ended but the secondary dispersion window (days 14-90) is still open given hantavirus incubation up to 6 weeks; (3) WHO risk assessment at cutoff found 'no sign of larger outbreak' and zero Phase 2 cases had been publicly reported, but multinational contact tracing across 10+ countries was ongoing with variable detection capacity.
Key events (≤ cutoff):
  - WHO DON599 published early May 2026 reporting 7 confirmed/suspected hantavirus cases on MV Hondius, 3 deaths
  - Illness onset among index cases ranged April 6-28, 2026
  - Vessel carried 147 passengers and crew on March-April 2026 Antarctic voyage
  - Confirmed pathogen: Andes virus (ANDV) — the only hantavirus with documented limited human-to-human transmission
  - WHO risk assessment at cutoff: 'no sign of larger outbreak'
  - Zero Phase 2 (off-ship) secondary cases publicly reported as of cutoff, May 21, 2026
  - International contact tracing initiated under IHR mechanisms across multiple countries where passengers disembarked
Active threads:
  - respiratory-season-2025-26 (health) — active but tangential, covers only influenza/RSV/COVID surveillance not hantavirus
  - hantavirus (health entity) — newly created May 21, 2026 with historical base rates, strain table, and forecasting rules
  - transport-vector-outbreak (health concept) — framework specifically written for MV Hondius outbreak, two-phase pattern
Mechanisms / concepts:
  - transport-vector-outbreak — two-phase pattern: index cluster (Phase 1, days 1-21) followed by secondary dispersion (Phase 2, days 14-90); Phase 2 detection depends on H2H capability, incubation period relative to voyage, contact tracing capacity, and cruise line cooperation
  - zoonotic-outbreak-case-count-forecasting — plateau model for non-H2H outbreaks; Andes virus's limited H2H breaks the plateau partially but not to pandemic scale (historical Andes outbreaks peaked at ~30 cases)
  - outbreak-escalation — Stage 2 (local cluster) with ≥3 linked cases from common exposure; without sustained H2H transmission, clusters burn out; MV Hondius at Stage 2 at cutoff
  - Andes virus specific: only hantavirus with confirmed human-to-human transmission; documented in 1996 El Bolsón (16 cases) and 2018-19 Epuyén (30+ cases) outbreaks
  - cruise ship enclosed environment amplification risk (Diamond Princess/Grand Princess COVID-19 precedent) but mitigated for inefficient H2H pathogens
Still uncertain at cutoff (do not treat as resolved):
  - Whether the ~6 week hantavirus incubation window will produce secondary cases detected after cutoff (voyage ended late April, incubation extends to early June)
  - Effectiveness of multinational contact tracing across 10+ countries with variable public health capacity
  - Whether cruise line (Oceanwide Expeditions) passenger manifest sharing with health authorities is complete enough to identify all close contacts
  - Whether healthcare workers treating index patients in multiple countries could become secondary cases (household/close contact transmission pattern in Epuyén)
  - Whether household contacts of the 147 passengers/crew who disembarked across multiple countries will develop symptoms within the incubation window
  - WHO's 'no sign of larger outbreak' assessment could be a leading indicator that Phase 2 suppression is working, or could reflect detection lags in countries with weaker surveillance
  - How resolution criteria will handle 'linked by a public health authority' — epidemiology reports linking cases to the outbreak vs forma

## Cross-References

- [[domains/health/entities/hantavirus]] — Pathogen entity with strain table and Andes virus H2H risk assessment
- [[events/mv-hondius-hantavirus-outbreak-2026]] — MV Hondius outbreak event file (same outbreak)
- [[domains/health/concepts/transport-vector-outbreak]] — Shipborne outbreak dynamics concept (canonical case)
- [[runs/20260521-050419-will-at-least-five-hantavirus-cases-linked-to-the-mv-hondius]] — Sibling run: same question from metaculus-cup (p_yes=0.20, id=test-hondius-offship)
- [[runs/20260521-044449-will-a-hantavirus-outbreak-with-over-100-confirmed-cases-be-]] — Related run: 100+ cases in 2026 (p_yes=0.07)
