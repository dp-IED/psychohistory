## Live Forecast — Will the next diplomatic US-Iran meeting be in Iran?

### Market Data
- Contract: Yes: 0.4% | No: 99.6% | Volume: $994,568
- Ends: Not specified in scan (likely event-driven — resolves when next meeting location is announced)
- Source: Polymarket

### Vault Files Read
- `domains/iran/threads/us-iran-post-war-diplomacy/_thread.md` — diplomatic track, meeting venues, Switzerland mediation
- `events/us-iran-peace-deal-june-2026.md` — June 14 peace deal signed via Pakistan mediation
- `runs/20260615-us-iran-switzerland.md` — previous meeting in Switzerland
- `domains/global/entities/iran.md` — Iran entity with US diplomatic relations context
- `domains/global/entities/donald-trump.md` — Trump travel patterns and diplomatic preferences
- `_forecast_instructions.md` — structural reasoning procedure

### Forecast Instructions Check
- Rule 1 (Central Bank): Not triggered
- Rule 2 (Domestic Politics): Not triggered
- Rule 11 (Public Event Before Cutoff): Not triggered
- Structural Reasoning Procedure: Applied — time dimension (diplomatic sequencing), chain dimension (peace deal → implementation meetings), anchor dimension (historical meeting venues)

### Vault Knowledge Summary
The vault documents that the June 14, 2026 US-Iran peace deal was brokered by Pakistan. Previous diplomatic meetings occurred in Switzerland (per run notes). The vault confirms zero US diplomatic presence in Iran since 1979 (embassy hostage crisis → severed relations). The peace-broker-diplomatic-playbook concept shows Trump's preference for neutral-venue summits (Alaska Summit for US-Russia, Switzerland/Geneva for US-Iran). No US president has visited Iran since Carter (1977, pre-revolution).

### Vault Usage Score: HIGH
The vault provides specific historical precedent (Switzerland as meeting venue), diplomatic pattern analysis (neutral venues for adversarial diplomacy), and structural context (no US embassy in Iran since 1979, Pakistan as mediator). This is not general knowledge the model would reliably have about the specific US-Iran post-war diplomatic track.

### Counterfactual
Without the vault, the forecast would still be NO with high confidence (general knowledge: US-Iran meetings happen in neutral venues). But the vault adds specificity: it confirms the Switzerland pattern, documents the Pakistan mediation role, and provides the peace deal context that makes "next meeting in Iran" structurally impossible rather than just improbable.

### Forecast
**Prediction:** NO
**Confidence:** 0.992
**Reasoning:**

1. **No US diplomatic presence in Iran since 1979**: The US has not had an embassy or diplomatic mission in Iran since the 1979 hostage crisis. Diplomatic communications flow through the Swiss embassy (US Interests Section). A US-Iran meeting on Iranian soil would require either: (a) a temporary diplomatic mission with security guarantees unprecedented in 47 years, or (b) a presidential/senior official visit requiring Secret Service-level security coordination with a government the US was bombing 16 months ago. Neither is remotely plausible as the FIRST post-peace-deal meeting.

2. **Neutral-venue precedent is overwhelming**: All US-Iran diplomatic meetings since 1979 have occurred in neutral third countries (Switzerland, Oman, Qatar, Iraq, UN venues in New York). The June 2026 peace deal was mediated by Pakistan. The meeting documented in `runs/20260615-us-iran-switzerland.md` was in Switzerland. The diplomatic pattern is so established that a first meeting in Iran would represent a complete break with 47 years of precedent.

3. **Security and optics**: A US official traveling to Tehran would face: (a) physical security risks (IRGC hardliners opposed to the peace deal), (b) domestic political backlash in the US (meeting on "enemy soil" 16 months after the US bombed Iran), (c) Iranian domestic politics (hardliners would frame it as US capitulation or spy mission). Neutral venues solve all three problems.

4. **Sequencing logic**: The peace deal was signed 5 days ago. The diplomatic sequence is: (a) technical working groups on sanctions relief, (b) blockade lifting (92% YES by June 30), (c) verification framework, (d) higher-level political meetings. A meeting IN Iran is a capstone event that would only occur after multiple successful lower-level meetings — it is not the NEXT meeting.

5. **Market calibration**: 0.4% YES ($994K volume) is effectively a near-consensus NO. The 0.4% represents either noise traders or a technical interpretation where a virtual/Zoom meeting with Iranian officials "in Iran" could theoretically trigger YES. Even accounting for loose resolution criteria, the probability is below 1%.

**Key uncertainty**: Resolution criteria — if "meeting" includes virtual meetings and "in Iran" means the Iranian delegation is physically in Iran, this could trigger at non-trivial probability. But such interpretations are unlikely given Polymarket's resolution norms.
