# Scale forecast: grill-with-docs session

**Status:** in session (not an ADR). One question at a time. Write glossary/ADRs only when a term or trade-off actually settles.  
**Skill:** Matt Pocock `grill-with-docs` (grilling + domain-modeling).  
**Date:** 2026-08-20

## Settled so far

**Q1 — What would “it worked” look like?**  
All four: dated events, material facts, official machines, and what people treat as obvious.  
**Why:** help **LFI, DSA, and similar parties** with a long fight over institutions and common sense, and a short fight when power is contestable.

**Q2 — What should come out for those parties?**  
**A + B, not C as a scored guess.** Forecasts (dated, public, gradeable) plus a **map of openings**. Not a **playbook**. ADR 0014.

**Q3 — Whose openings?**  
**Multitenant:** whoever is asking this run (C), not one hardcoded party. Openings are from that **tenant’s** point of view. Not a nameless “the left,” and not “LFI-only.” Where tenant id is passed, and what is shared vs private, not locked.

---

## Answer these (one line is enough)

Skip any that feel premature. “I don’t know yet” is a valid answer.

### What you are trying to get right

1. In five years, if this worked, what would you be proud you predicted? Give 3–5 concrete examples (not “the world”).
2. Are those mostly **one-day events** (an election, a war starting, a rate decision) or **slow things** (a country getting poorer, a party slowly losing the public)?
3. If you had to pick one: would you rather be good at **many small dated bets**, or at **a few big slow pictures** that might never have a clean score date?
4. Name some events you care about that are **not** elections. If the list is empty, say so.

### Money, factories, ships vs laws, parties, media vs “what people believe”

You said you want three kinds of “structure”:

- **Material:** food, energy, factories, ports, money, weapons, labor.
- **Official:** laws, courts, parties, armies, schools, newspapers, churches.
- **Mental:** what a person or a public treats as obvious, fair, or unsayable.

5. For a prediction you care about, which of those three actually **moves the outcome** most often in your head: material, official, or mental?
6. Do you want the system to **predict those structures themselves** (“this public no longer believes X”), or only use them as **reasons** for an event (“they will vote no **because** they no longer believe X”)?
7. How would a stranger check that a “belief” or “mood” claim was right, without asking you? Poll? Strike? Turnout? Something people said on TV? If you have no check, should we still try to store it?

### Whose thoughts

8. When you say “mental structures,” do you mean **the people in the event** (voters, a prime minister, a general), **the forecasting plugin’s own reasoning**, or both?
9. If both: should we keep those **in different places** so we don’t treat “the model wrote a clever paragraph” as “Israelis think X”?

### Time

10. Are you more interested in things that resolve in **days/weeks** (so we can learn fast) or **months/years** (closer to “how a society works,” slower to score)?
11. Is “the mood of a country” allowed as a scored question, or only things with a **public date and a clear yes/no or who-won**?

### What would change your mind

12. If we keep scoring US primaries and sports and we get good at those, does that count as progress toward your goal, or is that a distraction?
13. What is an example of a question about **ports, energy, strikes, debt, or factories** you would actually want on the list?
14. What is an example of a question about **what a government can openly say** (or cannot say) that you would want on the list?

### How the pieces connect

15. When something happens, do you think the usual story is: **money/stuff runs out → officials act → the public’s story changes**, or some other order? Write the order you believe in, even roughly.
16. For one real case you care about (tariffs, an election, a war): what **physical or money limit** could have changed the outcome? If you can’t name one, that’s useful too.

### What “at scale” means to you

17. Does “at scale” mean **more questions**, **longer into the future**, **more countries**, or **combining several reasons on one question** (e.g. “ships + parliament”)?
18. Would you rather have **20 reusable “this kind of situation usually goes like this” write-ups** that survived being wrong sometimes, or **a big map of everything**?
19. Are you trying to **forecast** what will happen, or also to **advise** what someone should organize? (Those grade differently.)

### Using the plugin runs (no new invention)

20. After a question resolves, what do you want to **keep** from the write-up: the guess, the reasons, a short “this is a type of situation,” or the method (“always look up the law first”)?
21. If a type-of-situation write-up keeps failing on the next similar question, should we **delete it** even if it sounds true?
22. Should we **wait until some questions have actually resolved** before adding new kinds of notes about “society,” or start those notes now?

---

## Tiny glossary (only if a question used a word you don’t use)

- **Ledger / problem:** a question with a date when the world answers.
- **Claim:** one dated guess on that question.
- **Plugin / overlay:** the instruction files we keep after scoring; not the list of guesses.
- **Analog card:** a short reusable note: “this *kind* of situation usually works like this.”
- **Reflection:** after the date, we grade the guesses and change those files.

---

## Appendix (jargon map; skip unless you want it)

Events stay what we score. Material / official / mental are reasons that must help those scores. Extra objects (who can actually do the thing; how a fact becomes a law becomes a speech; what we can observe) live in those reusable notes only when a scored miss or hit needed them. No big taxonomy up front. No restored graph. Simulators only if a scored type of situation repeatedly needs them.

Proposed locks after you answer: see git history of this file before the plain-language rewrite, or chat answers copied below when you lock them.

### Answers (paste here)

_Empty until you reply._
