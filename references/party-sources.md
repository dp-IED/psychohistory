# Party-sourced discover index

Public channels and extracted questions that **orient** discover. Not a transcript vault. Do not score the past recording. Open a **live** problem whose **resolution day** is still ahead, and cite the URL in Motivation (ADR 0018).

Refresh this index on later discover ticks from the same public pages. Closed Slack/Discord stays in harness chat.

## How to scrape (public only)

1. Start from the channel/playlist URLs below, plus `https://amfis.fr/programme` while an edition is live.
2. Prefer titles, official programme rows, published platforms, and **available** captions. YouTube timedtext from datacenter IPs is often blocked; WebFetch of the watch page sometimes still returns a caption dump. Cache full text only in gitignored `.scratch/`.
3. Extract the **world question** the speakers treat as a live fight (election, law, labor clock, media, debt, org strategy with a date). Map each row to a future resolution day or to a **structural class** for analog-regime Motivation.
4. Amfis and DSA are **examples of the method**, not the only tenants.

## Channel index

| Tenant example | What | URL | Notes |
| --- | --- | --- | --- |
| LFI | Official site | https://lafranceinsoumise.fr/ | News + programme pointers |
| LFI | Amfis site / 2026 programme | https://amfis.fr/ · https://amfis.fr/programme | 20–23 Aug 2026, Valence / O Lac |
| LFI | YouTube channel | https://www.youtube.com/channel/UCKHKSD-yanY2ZwwU_4Tgf0w | RSS: `feeds/videos.xml?channel_id=UCKHKSD-yanY2ZwwU_4Tgf0w` |
| LFI | 2027 campaign | https://melenchon2027.fr/ | Citizen parrainages; not the 500 elected endorsements |
| DSA | National site | https://www.dsausa.org/ | NPC notes, public talks |
| DSA | YouTube | https://www.youtube.com/@DSAdemsocialists (`UC6Bm7YK7z8Y2rGAmJx8Qaug`) | RSS: `feeds/videos.xml?channel_id=UC6Bm7YK7z8Y2rGAmJx8Qaug` |
| DSA | 2025 convention | https://convention2025.dsausa.org/ | Biennial; next convention ~2027 |
| DSA | NEC write-up | https://electoral.dsausa.org/reflecting-on-the-2025-national-convention/ | Electoral resolutions |
| DSA | Workers Deserve More | https://program.dsausa.org/ | 2026–27 program; endorsement yardstick |

### LFI Amfis playlists (official channel, via [fr.wikipedia.org/wiki/Les_Amfis](https://fr.wikipedia.org/wiki/Les_Amfis))

| Edition | Playlist / video |
| --- | --- |
| 2017 | https://www.youtube.com/playlist?list=PL49hPUZTPboWVAxzbehRKo0eJK8KO-TQ3 |
| 2018 | https://www.youtube.com/playlist?list=PL49hPUZTPboUnsNnqw6eiOe5_-J5MFLQQ |
| 2019 | https://www.youtube.com/playlist?list=PL49hPUZTPboXKX_5inU76F69unQTq62qA |
| 2020 | https://www.youtube.com/playlist?list=PL49hPUZTPboV3My_wDfaSALmDuwWMl6NH |
| 2021 | https://www.youtube.com/playlist?list=PL49hPUZTPboV8XhdNioUvq_SfKoyMG7tF |
| 2022 | https://www.youtube.com/playlist?list=PL49hPUZTPboV4Lvad9GWg-SPpAr9en7uY |
| 2023 | https://www.youtube.com/playlist?list=PL49hPUZTPboUQNJoinG2chvPrSmrYzX4n |
| 2024 | https://www.youtube.com/playlist?list=PL49hPUZTPboWbL4l9vHGyXAn-wlAsatpk |
| 2025 | https://www.youtube.com/watch?v=dqCa8r6TOLw (wiki lists a video, not a playlist id) |

## Extracted questions (2026-08-21 scrape)

Each row is orientation, not a scored forecast of the talk. Suggested resolution days are **after 2026-08-21**.

| Source | Date of source | Extracted question (live fight) | Suggested resolution day | Class / regime | Opened? |
| --- | --- | --- | --- | --- | --- |
| Amfis 2026 welcome, captions on https://www.youtube.com/watch?v=HW03wZDi6Tk (Bompard, Guetté, Lachaud, Pouille) | 2026-08-20 | LFI niche abrogate the agricultural-law pesticide provisions (acétamipride / flupyradifurone) | 2026-10-29 | news-now; **opposition niche-day repeal of a just-passed majority law** | `P-fr-pest-niche` |
| Same welcome + LFI tribune https://lafranceinsoumise.fr/2026/08/19/tribune-le-29-octobre-pesticides/ | 2026-08-19 | Street start of the pesticide / climate rentrée: a nationally reported demonstration on 15 Sep | 2026-09-15 | news-now | `P-fr-pest-street` |
| Same welcome (Pouille as Gironde insoumise Senate head of list) | 2026-08-20 | Does that list take ≥1 of Gironde’s 6 PR Senate seats? Official clock: https://www.interieur.gouv.fr/actualites/actualites-du-ministere/elections-senatoriales-27-septembre-2026 | 2026-09-27 | news-now | `P-fr-gir-sen` |
| Amfis programme https://amfis.fr/programme — « Rechercher les 500 parrainages »; Saint-Denis launch captions https://www.youtube.com/watch?v=Z5YIa1QD178 | 2026-06-07 / 2026-08-21 | Is Mélenchon on the Conseil constitutionnel official first-round list? (500 elected endorsements; citizen parrainages on melenchon2027.fr are a different clock) | 2027-03-26 | analog: **two-round presidential ballot-access endorsements** | index only (politics already half of this tick) |
| Same welcome + programme « Comment appliquer un programme économique de rupture ? » https://www.youtube.com/watch?v=hefmL-vkxKw ; official dates https://www.service-public.gouv.fr/particuliers/actualites/A15053 | 2026-08-20 | Who wins the most votes in the 2027 presidential **first round** (not the Élysée yet)? | 2027-04-18 | analog: **two-round presidential first-round plurality** | `P-fr-pres-t1` |
| Programme « Septembre : coup d’envoi d’une année de lutte ? » (CGT / FSU / Solidaires) | 2026-08-22 slot | Fonction publique intersyndicale strike/manifestation day | 2026-09-29 | news-now labor | index only (avoid a third France-September row this tick) |
| Programme « Faut-il interdire les milliardaires ? » https://www.youtube.com/watch?v=anVooQj8LlE | 2026-08-19 upload | EU / FR wealth-tax or billionaire-cap legislative clock | none dated in the title | analog: **progressive wealth taxation** | class only |
| Programme « Une nouvelle République… 6e République » | 2026-08-21 programme | Constituent process after a 2027 win | no independent date | analog: **post-election constituent assembly** | class only |
| DSA Tlaib keynote captions https://www.youtube.com/watch?v=-0BYHdRe2b8 | 2025-08-10 | Stop US weapons funding for Israel; midterms as the reckoning | 2026-11-03 (House) | news-now politics; analog **arms-embargo legislative rider** for a later NDAA row | House opened as `P-us-house-26` |
| DSA convention + NEC https://convention2025.dsausa.org/ · https://electoral.dsausa.org/reflecting-on-the-2025-national-convention/ | 2025-08 | 2026 midterms; 2028 Congress slate (Carnation: 5 House candidates); independent ballot lines | 2026-11-03 / 2028 | news-now / analog | House opened |
| DSA R30 / co-chair May Day 2028 (public reports of the convention labor resolution; Fain Big Three expiration 2028-04-30) | 2025-08 | Coordinated May Day strike when auto contracts expire | 2028-05-01 | analog: **aligned-contract May Day general strike** | `P-us-mayday-28` |
| DSA program https://program.dsausa.org/ · launch video https://www.youtube.com/watch?v=GzhPYE_OWW0 | 2026-07 | Medicare for All, abolish ICE, Green New Deal, Free Palestine as endorsement tests | various 2027–28 | analog cards later | class only |
| DSA Socialist Summit 2026 https://www.youtube.com/watch?v=kuKLtHkS20s | 2026-08-17 | Same 2026–28 fights in a public assembly | n/a | orientation | not scored |

Other Amfis 2026 watch URLs on the LFI RSS at scrape time (several still `length=0` placeholders, no captions yet): `RqQQ_cDyYGQ` Guetté; `Uqe5vvsXsZ4` La Boétie *L’État et la révolution citoyenne*; `KJvF-KUAFqE` Palestine solidarity criminalization; `gIg8h2TEiRE` ecological planning (Panot / Chatelain); `zfSZ82IaRZg` Lordon; `ZoRWhmtiWsc` Pigasse. Closing meeting Sunday 23 Aug 2026 is a campaign event, not a resolution day.

## Anti-patterns

- Do not open “what did they conclude at Amfis 2024/2025.”
- Do not fill `K` from one week of headlines even if every talk names it.
- Do not dump closed internals into this file.
