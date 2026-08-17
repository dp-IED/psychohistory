---
type: procedure
tags: [procedure, antitrust, divestiture, forecast-methodology]
title: "Antitrust Divestiture Timeline Forecast"
domain: usa
subdomain: antitrust
related_concepts:
  - antitrust-divestiture-timeline
  - comprehensive-tech-regulation-gridlock
  - national-security-tech-ban
---
# Antitrust Divestiture Timeline Forecast

## When to Use

Use this procedure when a forecasting question asks whether a specific company, asset, or business unit will be **forced to sell or divest** through **antitrust litigation** (not national security legislation) within a specified timeframe.

## Procedure Steps

### Step 1: Identify the Legal Vehicle

Distinguish which forced-divestiture framework applies:

| Vehicle | Example | Timeline | Probability Profile |
|---------|---------|----------|-------------------|
| Sherman Act §2 monopolization litigation | FTC v. Meta (Instagram/WhatsApp) | 5-10+ years from filing | Low in <3yr window |
| Clayton Act §7 merger challenge (post-consummation) | FTC v. Meta (Instagram) | 5-10+ years from filing | Very low in <3yr window |
| Clayton Act §7 merger challenge (pre-consummation) | Meta/Within (Beat Saber) | 12-18 months from challenge | Moderate |
| National security legislation | TikTok divest-or-ban | 6-18 months from bill | Moderate-high |
| Executive order (national security) | WeChat restrictions | 1-6 months | Variable, legally vulnerable |
| Regulatory agency rulemaking | FCC media ownership rules | 2-5 years | Low |

If the vehicle is antitrust litigation (Sherman Act §2 or Clayton Act §7 post-consummation), proceed below. Otherwise, use the concept appropriate to the vehicle (e.g., [[domains/global/concepts/national-security-tech-ban]]).

### Step 2: Map the Case Timeline

Determine the current stage and estimate remaining time:

1. **Has the case been filed?** If no, add 0-12 months for filing
2. **Has the complaint survived motion to dismiss?** 
   - If dismissed and refiled pending: add 12-24 months to stage 2
   - If dismissed with prejudice: case is effectively over (p_yes ≈ 0)
3. **Is discovery complete?** (typically 18-36 months)
4. **Has summary judgment been decided?**
5. **Has trial occurred?**
6. **Has liability been found?**

Estimated remaining time = sum of remaining phases:
- Motion to dismiss: 6-12 months
- Discovery: 18-36 months
- Summary judgment: 6-12 months
- Trial: 1-6 months
- Liability ruling: 1-6 months
- Remedy briefing/hearings: 6-12 months
- Appeals: 2-5 years

### Step 3: Assess Divestiture Probability

Apply the default priors from the [[antitrust-divestiture-timeline]] concept:

**Base probability**: For antitrust litigation seeking post-consummation divestiture within N years of filing:
- <1 year: p_yes < 0.01
- <2 years: p_yes < 0.03
- <3 years: p_yes 0.03-0.05
- <5 years: p_yes 0.10-0.20
- <10 years: p_yes 0.30-0.50

**Adjustments**:
- Consummated merger (already owned): -50%
- Enforcement-first administration (Khan-era FTC): +50%
- Permissive administration (Ferguson-era FTC): -30%
- Previous liability finding in same case: +2x multiplier
- Parallel EU enforcement (DMA, EC competition): +20% (increases regulatory pressure, may create remedies that US can adopt)
- Deep technical integration (Instagram/WhatsApp into Meta): -25% (divestiture feasibility lower)
- Strong economic evidence of harm: +25%
- Weak evidence / novel legal theory: -25%

### Step 4: Consider the Remedy Phase

Even if liability is found, the remedy phase has its own barriers:

1. **Structural remedy (divestiture) requires showing:**
   - The merger caused anticompetitive harm (causal link, not just market power)
   - Divestiture is feasible (buyer exists, assets can be separated)
   - Divestiture would restore competition (not make things worse)
   
2. **Appeals court will review:**
   - Market definition (de novo)
   - Liability findings (deferential)
   - Remedy (abuse of discretion)
   
3. **Remedy alternatives to divestiture:**
   - Behavioral remedies (firewall, non-discrimination, data access) — more likely than structural
   - Injunctive relief (prohibiting future conduct) — most likely
   - Consent decree with limited conditions — most common resolution

4. **The AT&T/Microsoft precedent**: 
   - Microsoft's breakup was REVERSED on appeal (DC Circuit, 2001)
   - AT&T was settled via consent decree, not fully litigated
   - No court has ordered a tech platform divestiture post-consummation that survived appeal

### Step 5: Political Environment Check

Assess the current administration's antitrust posture:

| Factor | Enforcement-First (Biden/Khan) | Permissive (Trump/Ferguson) |
|--------|-------------------------------|-----------------------------|
| Remedy aggressiveness | High (seek structural) | Low-Moderate (seek behavioral) |
| Appeal defense posture | Aggressive | May settle on weaker terms |
| Agency resources | Increased | Stable or decreased |
| Legislative pressure | Strong (bills advancing) | Weak (bills stalled) |
| Industry engagement | Adversarial | Cooperative |

If the question timeframe spans an election year, factor in the possibility of a regime change and its impact on remedy posture.

### Step 6: Final Calibration

1. Check the case's current docket — has the judge set a trial date? Post-trial briefing schedule?
2. Check for parallel legislative action — is Congress considering sectoral antitrust legislation that could create a separate path to divestiture?
3. Check for changes in agency leadership — are new commissioners being appointed who could shift enforcement posture?
4. Check settlement negotiations — have the parties engaged in serious settlement discussions?

## Example Application: "Will Meta be forced to sell Instagram or WhatsApp in 2025?"

Applied to the FTC v. Meta case:

1. **Vehicle**: Sherman Act §2 monopolization + Clayton Act §7 post-consummation → antitrust framework
2. **Timeline**: Filed Dec 2020. In 2025, case was in pre-trial phase (summary judgment pending). Trial not yet scheduled. Even if liability found in 2025, remedy phase + appeals would extend to 2028+.
3. **Divestiture probability**: Base prior for <5 years from filing = 10-20%. Adjustments: consummated merger (-50%), permissive 2025 administration (-30%), deep integration (-25%) → ~2.5-5% adjusted. Further reduced by Fe
4. **Short horizon**: Question asks about 2025 specifically, not "ever." Even a fast-track liability ruling would not produce a divestiture order in 2025.
5. **Result**: p_yes ≈ 0.01-0.03 (consistent with NO resolution).

## Validation

| Question | Prediction | Actual | Notes |
|----------|-----------|--------|-------|
| Meta forced to sell Instagram/WhatsApp in 2025? | NO | NO (correct) | Correctly identified antitrust divestiture timeline (5-10 years) not aligned with 2025 horizon; administration shift reduced remedy pressure |

## Wikilinks

- [[domains/usa/concepts/antitrust-divestiture-timeline/_concept]]
- [[domains/usa/threads/us-big-tech-antitrust-enforcement/_thread]]
- [[domains/usa/entities/federal-trade-commission]]
- [[domains/usa/entities/meta-platforms]]
- [[domains/global/concepts/national-security-tech-ban]]
- [[domains/usa/concepts/comprehensive-tech-regulation-gridlock/_concept]]
