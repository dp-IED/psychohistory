---
type: thread
tags: [thread, technology, regulation, usa]
title: "US Federal AI Safety Regulation"
slug: us-ai-safety-regulation-federal
inception: 2023-01-01
conclusion: null
status: active
domain: usa
subdomain: technology-regulation
---

# US Federal AI Safety Regulation

## Overview

Tracks the legislative journey of federal AI safety regulation in the United States — the bills introduced, the political dynamics that blocked them, and the structural barriers to comprehensive federal AI legislation. The core pattern is one of **sustained legislative effort producing zero enacted comprehensive regulation**, despite high public salience and repeated bipartisan working groups.

The canonical question this thread answers: **Why has the US not passed comprehensive AI safety legislation, and under what conditions might it ever do so?**

## Key Actors

| Actor | Position | Interest |
|-------|----------|----------|
| [[domains/usa/entities/donald-trump]] | Executive (2025-) | Deregulation, AI innovation priority, revocation of Biden EO |
| [[domains/usa/entities/joe-biden]] | Executive (2021-2025) | Precautionary approach via AI EO (Oct 2023) |
| [[domains/global/entities/openai]] | Industry | Self-regulation preference, opposes mandatory safety testing |
| [[domains/global/entities/anthropic]] | Industry | Supports regulation (differs from OpenAI), advocate for safety frameworks |
| [[domains/global/entities/google-deepmind]] | Industry | Mixed: research safety advocacy vs. competitive pressure |
| [[domains/global/entities/meta]] | Industry | Open-source advocate, opposes compute-based regulation |
| [[entities/chuck-schumer]] | Senate Majority Leader | SAFE Innovation Framework architect |
| [[entities/maria-cantwell]] | Senate Commerce Chair | AI Research, Innovation, and Accountability Act sponsor |

## Timeline

### 2023: Agenda-Setting Phase

- **2023-Q2** — Senator Chuck Schumer launches the SAFE Innovation for AI framework, announcing bipartisan working groups to develop AI legislation. The framework identifies five key areas: security, accountability, foundations of democratic values, explainability, and innovation.
- **2023-Q3** — Bipartisan AI Insight Forums begin, bringing together senators, AI company CEOs, researchers, and civil society. Nine closed-door forums held through Q4.
- **2023-Q4, Oct 30** — President Biden signs Executive Order 14110 on Safe, Secure, and Trustworthy Development and Use of AI. The EO invokes the Defense Production Act to require safety testing reports from frontier AI model developers and establishes the AI Safety Institute at NIST. This is the most significant federal AI action to date but is an executive order — reversible by the next president.

### 2024: Legislative Proliferation Without Convergence

- **2024-Q1-Q2** — Multiple AI bills introduced across committees:
  - **SAFE Innovation Act** (Thune, Klobuchar) — Creates AI security requirements for critical infrastructure
  - **AI Research, Innovation, and Accountability Act** (Cantwell, Moran, Beyer) — Establishes transparency requirements, AI Office at NTIA
  - **CREATE AI Act** (Young, Booker) — Creates national AI research resource
  - **Algorithmic Accountability Act** (Wyden, Booker, Clarke) — Requires impact assessments of automated decision systems
- **2024-Q3** — No comprehensive bill reaches the floor. Committee jurisdictions overlap; partisan disagreements over liability frameworks and preemption of state laws remain unresolved. California's SB 1047 veto (Sept 29-30) demonstrates the political difficulty of AI safety regulation even in the most pro-regulation state.
- **2024-Q4** — Election absorbs legislative attention. AI regulation is not a top-tier voter issue in the presidential campaign. Lame-duck session produces no AI legislation.

### 2025: Deregulatory Turn and Complete Gridlock

- **2025-01-20** — Trump is inaugurated. On his first day, he signs an executive order revoking Biden's AI EO (Oct 2023) and replacing it with a lighter-touch framework emphasizing innovation, removing safety testing requirements and the AI Safety Institute's mandatory reporting authorities.
- **2025-Q1** — Republican-controlled Congress shows no appetite for comprehensive AI safety regulation. Leadership prioritizes tax cuts, tariffs, and budget reconciliation. Several bipartisan AI bill reintroductions (SAFE Innovation Act, AI Research Act) are referred to committee and never marked up.
- **2025-Q2** — Liberation Day tariffs (April 2) consume congressional bandwidth. The Iran-Israel Twelve-Day War (June 13-24) absorbs executive branch attention. AI legislation receives zero floor time.
- **2025-Q3** — Alaska Summit (Aug 15), GPT-5 launch (Aug 7), and the World Humanoid Robot Games dominate the AI narrative but not the legislative calendar. No AI bill reaches a floor vote in either chamber.
- **2025-Q4** — Lame-duck session focuses on government funding and appropriations. AI safety bills do not advance. **No AI safety bill is signed into law in 2025**, confirming the prediction market resolution of NO.

### 2026 (Through June)

- **2026-Q1** — Continued gridlock. State-level AI regulation (California, Colorado, New York) advances as substitute for federal action. Industry lobbying against federal preemption intensifies.
- **2026-Q2 (Through May)** — Midterm election campaigning begins in earnest; legislative productivity declines further. Federal AI safety legislation remains stalled. Polymarket YES price: ~41.5%.
- **2026-Q2 Swing Event (June)** — The Polymarket "US enacts AI safety bill before 2027" market **crashed 24.5pp from 41.5% → 17.0%** (delta -24.5pp, $994K volume at time of detection). Three working hypotheses:
  - **H1 (Most likely):** A specific AI safety bill (Blumenthal-Hawley framework or successor) failed in committee, was withdrawn, or collapsed. In the compressed 2026 midterm calendar, a June bill failure has essentially zero revival chance before Jan 2027.
  - **H2:** Trump administration signaled formal veto threat or leadership announced AI safety won't reach the floor. The Trump 2025 AI EO established light-touch framework; comprehensive legislation threatens that approach.
  - **H3:** State-level preemption deal collapsed. Industry had been negotiating federal preemption of state AI laws in exchange for federal safety standards. Deal collapse flipped industry lobbying from neutral to opposition, killing the bill.
- **Vault assessment**: Structural NO at p_yes=0.17 (market 17%, vault 85% NO). The swing confirms the six-barrier gridlock framework. The 17% residual captures rider-legislation risk (attaching AI safety to must-pass NDAA/budget) and catastrophic AI incident scenarios.

## Structural Barriers to Federal AI Safety Legislation

### 1. Partisan Polarization on Tech Regulation

AI regulation does not break cleanly on party lines. Internal Democratic splits (safety advocates vs. innovation Democrats) are as consequential as partisan divisions. Republicans broadly oppose regulation that would "coddle" innovation to China. The result is a coalition that cannot form a majority for any particular regulatory approach.

### 2. Industry Lobbying Power

Major AI companies collectively spend hundreds of millions on federal lobbying. Their messaging strategy is sophisticated: support "voluntary commitments" and "responsible AI" frameworks while opposing binding legislation with liability provisions. The industry's geographic concentration (California headquarters, employees as a key Democratic donor constituency) gives them leverage across party lines.

### 3. Competing Regulatory Approaches

At least four distinct regulatory visions compete simultaneously:
- **Precautionary/safety-first**: Mandatory safety testing, model licensing, developer liability (supported by AI safety advocates, some of Anthropic's positions)
- **Innovation-first**: Minimal regulation, voluntary standards, export controls only for national security (Trump administration position)
- **Rights-based**: Algorithmic accountability, anti-discrimination, transparency (civil rights organizations, Algorithmic Accountability Act approach)
- **Sectoral/focused**: Regulation limited to specific high-risk domains (healthcare AI, autonomous vehicles, critical infrastructure)

No coalition has been able to merge these into a majority framework.

### 4. First-Mover Disadvantage Fear

The US fears that comprehensive AI regulation would cede AI leadership to China (where no comparable safety regulation exists). This argument is particularly potent in the post-DeepSeek environment (Jan 2025), where China's cost-efficient AI development has already created a Sputnik-moment narrative. The "regulate and lose" framework is used to block any binding legislation.

### 5. Preemption Deadlock

States, particularly California, are actively legislating AI. Industry strongly prefers federal preemption (a single national standard over 50 state regimes). But progressives oppose preemption that would weaken state-level protections. This deadlock prevents either path from advancing.

### 6. Executive Order as Substitute

Both Biden and Trump used executive orders as a substitute for legislation. Biden's EO (2023) created safety testing requirements via executive authority; Trump's EO (2025) revoked it. The reversibility of executive action means each administration can undo the previous one's AI policy, making durable regulation impossible without legislation.

## Forecasting Significance

This thread is the primary node for any question about:
- US federal AI safety legislation (probability, timeline, content)
- AI executive orders and their durability
- The relationship between state and federal AI regulation
- Industry influence on AI policy
- Congressional AI caucuses and working groups

**Key forecasting rule**: Federal AI safety legislation is the hardest level of AI regulation to achieve — harder than executive orders (which are unilateral), harder than state legislation (which has unified political control), and harder than international frameworks (which are non-binding). Assume a structural probability floor of ~5-10% for any comprehensive federal AI safety bill in any given Congress, absent a major AI incident (a "Sputnik event" for AI safety) that shifts the Overton window. The existing pattern (2023-2026: zero enacted comprehensive bills) validates this baseline.

## Wikilinks

- [[domains/usa/threads/state-level-ai-regulation/_thread]] — state-level AI regulation, California as bellwether
- [[domains/global/entities/openai]]
- [[domains/global/entities/anthropic]]
- [[domains/usa/entities/donald-trump]]
- [[domains/usa/entities/gavin-newsom]]
- [[domains/global/concepts/national-security-tech-ban]]
- [[domains/usa/concepts/state-level-tech-regulation-bellwether/_concept]]
- [[domains/usa/concepts/governor-veto-tech-bill-dynamics/_concept]]
- [[concepts/comprehensive-tech-regulation-gridlock/_concept]]
- [[timeline/2025-Q1]]
- [[timeline/2025-Q2]]
- [[timeline/2025-Q3]]
- [[timeline/2025-Q4]]
