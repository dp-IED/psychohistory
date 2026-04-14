# Reformation To US Foreign Policy Path

## Question

How has the Protestant Reformation impacted modern American foreign policy?

## Modeling Principle

Do not encode this as:

```text
Protestant Reformation -> caused -> US foreign policy
```

Encode it as contested, time-scoped transmission paths:

```text
origin -> transmission channel -> legacy carrier -> policy frame or institution
```

## Current Probe Task

The Protestant Reformation probe now includes:

```text
reformation_legacy_to_us_foreign_policy
Event --transmits_through--> TransmissionChannel --leaves_legacy_via--> Policy
```

This is a retrieval/training scaffold, not a final historical claim.

## Candidate Path Families

1. Scriptural authority and conscience:
   - Reformation authority claims
   - dissenting Protestant traditions
   - religious liberty language
   - American political theology
   - foreign-policy moral framing

2. Puritan and settler inheritance:
   - English Reformation
   - Puritan migration
   - New England institutions
   - American exceptionalism
   - mission/covenant language in foreign policy

3. Evangelical missions:
   - Protestant missions
   - nineteenth-century missionary networks
   - humanitarian and reform movements
   - international religious freedom advocacy

4. Secularized institutional legacy:
   - confessional state formation
   - literacy, education, discipline, public argument
   - civic religion and liberal institutionalism
   - Wilsonian or rights-based internationalism

5. Counter-paths:
   - Enlightenment and republican traditions
   - Catholic, Jewish, secular, and pluralist influences
   - strategic/geopolitical material interests
   - Cold War institutional incentives

## What A Good Graph Must Preserve

- Multiple rival inheritance paths.
- A gap between theological origin and modern policy carrier.
- Transformation, not simple survival.
- Counterclaims and non-Protestant explanations.
- Time windows for each path segment.
- Grounding for actors, movements, texts, institutions, and policy frames.

## Next Concrete Work

1. Add more precise intermediate nodes for Reformation-to-US paths:
   - `Puritan migration to New England`
   - `American exceptionalism`
   - `Evangelicalism in the United States`
   - `international religious freedom`
   - `Wilsonianism`
2. Add contested claim nodes:
   - Protestant legacy thesis
   - secularized institutional legacy thesis
   - geopolitical-materialist counterclaim
3. Add at least one negative/counterweight path so the GNN cannot learn a
   one-edge ideology shortcut.
4. Export the resulting graph artifact and inspect whether the path is learnable
   from typed structure alone before adding text embeddings.
