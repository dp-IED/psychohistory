---
type: entity
tags: [entity]
kind: organization
title: "Centers for Disease Control and Prevention (CDC)"
slug: centers-for-disease-control-and-prevention
date_start: 1946-07-01
date_end: null
pit_cutoff: 2025-06-30
---

## Summary

The Centers for Disease Control and Prevention (CDC) is the United States' national public health agency, headquartered in Atlanta, Georgia, and operating under the Department of Health and Human Services (HHS). Founded in 1946 as the Communicable Disease Center (originally focused on malaria control), the CDC is the primary federal authority for infectious disease surveillance, outbreak investigation, and public health risk assessment. Its mission includes protecting the US from health threats, conducting epidemiological research, and maintaining national health statistics.

## Significance for Forecasting

The CDC is the authoritative source for US infectious disease case counts, risk assessments, and outbreak declarations — making it a critical entity for any forecast question involving US disease outbreaks, public health emergencies, or epidemiological milestones. Understanding the CDC's institutional incentives and constraints is essential for interpreting its public statements and calibrating the probability of specific outbreak milestones.

## Key Dynamics for Outbreak Forecasting

1. **Case count authority**: The CDC's official case counter is the definitive source for US disease outbreak case counts. Polymarket and other prediction markets resolve against CDC-confirmed case data. When a question asks about case count thresholds, the CDC counter is the reference standard.

2. **Risk assessment as leading indicator**: The CDC maintains a public health risk level for each disease outbreak (low/moderate/high). Changes in this assessment are the single most important leading indicator for outbreak trajectory. The agency has a strong institutional conservatism in upgrading risk levels — it takes documented evidence of changing transmission dynamics (H2H transmission, virulence shift, vaccine escape) to trigger a risk upgrade. This means the "low" assessment can persist longer than seems reasonable to media or political observers, and its maintenance is a signal that the internal epidemiological models do not predict escalation.

3. **Institutional incentives**:
   - **Public health reputation**: The CDC's credibility was damaged by its COVID-19 response (testing delays, evolving mask guidance, political interference allegations). This creates a bias toward more cautious, data-justified public communications — a "trust but verify" posture that may delay risk upgrades until evidence is overwhelming.
   - **Outbreak fatigue**: The COVID-19 pandemic created a high baseline of public attention to outbreaks. The CDC may be reluctant to declare novel outbreaks or upgrade risk assessments for zoonotic spillover events that lack pandemic potential, knowing that outbreak declarations trigger economic and political consequences.
   - **Agricultural industry sensitivity**: Outbreak declarations affecting the livestock industry (dairy, poultry) are politically sensitive given the economic stakes. The CDC must coordinate with USDA and state agricultural departments, and its public posture may reflect inter-agency compromise.

4. **Data products relevant to forecasting**:
   - **H5N1 Situation Summaries**: Weekly updates with case counts, state distribution, and risk assessment
   - **FluView**: Weekly influenza surveillance report with novel influenza A detections
   - **MMWR (Morbidity and Mortality Weekly Report)**: Official outbreak reports with detailed epidemiological data
   - **CDC Case Counter**: Online dashboard showing confirmed human cases by state

5. **Pre-forecast audit questions**:
   - Has the CDC published an official case count for the relevant outbreak? If yes, this is the PIT baseline.
   - What is the current CDC risk assessment for this outbreak? "Low" = bounded case counts; any upgrade requires evidence.
   - Has the CDC documented any human-to-human transmission? If no, the plateau model holds.
   - Is there evidence of inter-agency friction (CDC vs. USDA) that could affect reporting?

## Related Files

- [[domains/global/threads/h5n1-avian-influenza-outbreak/_thread]] — H5N1 outbreak thread
- [[domains/global/concepts/zoonotic-outbreak-case-count-forecasting/_concept]] — Forecasting concept
- [[domains/global/procedures/outbreak-case-threshold-forecast]] — Forecasting procedure
