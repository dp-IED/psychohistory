---
type: index
tags: [meta, case-library, dataview]
date: 2026-05-21
purpose: "Structured case library for outside-view base rate anchoring"
---

# Case Library

Resolved Polymarket cases with structured metadata. Queryable via Dataview.
**100 cases** across 19 reference classes.

## Reference Class Summary

| Event Type | Domain | N | Base Rate | YES | NO |
|------------|--------|---|-----------|-----|----|
| appointment | politics | 2 | 0.0% | 0 | 2 |
| ceasefire | geopolitics | 11 | 63.6% | 7 | 4 |
| court_ruling | politics | 3 | 0.0% | 0 | 3 |
| court_ruling | technology | 1 | 0.0% | 0 | 1 |
| election | geopolitics | 1 | 0.0% | 0 | 1 |
| election | politics | 22 | 40.9% | 9 | 13 |
| government_action | economics | 1 | 0.0% | 0 | 1 |
| government_action | politics | 4 | 25.0% | 1 | 3 |
| macro_release | economics | 2 | 0.0% | 0 | 2 |
| macro_release | geopolitics | 4 | 25.0% | 1 | 3 |
| macro_release | politics | 2 | 50.0% | 1 | 1 |
| military_strike | geopolitics | 1 | 0.0% | 0 | 1 |
| other | economics | 6 | 50.0% | 3 | 3 |
| other | geopolitics | 2 | 0.0% | 0 | 2 |
| other | politics | 16 | 37.5% | 6 | 10 |
| other | technology | 6 | 16.7% | 1 | 5 |
| rate_decision | economics | 8 | 37.5% | 3 | 5 |
| regulatory_approval | economics | 6 | 50.0% | 3 | 3 |
| regulatory_approval | technology | 2 | 50.0% | 1 | 1 |

## All Cases

```dataview
TABLE resolution, event_type, domain, time_horizon_days, reference_class_base_rate
FROM "cases"
SORT domain ASC, event_type ASC
```

## Economics (23 cases)

- ❌ [[cases/gold_39_debt-ceiling-raisedsuspended-before-trump-inaugu|Debt ceiling raised or suspended by inauguration?]] `government_action` 30d
- ❌ [[cases/gold_55_will-us-gdp-growth-be-less-than-2-in-q1-2025|Will U.S. GDP growth be less than -2% in Q1 2025?]] `macro_release` 56d
- ❌ [[cases/gold_59_will-us-gdp-growth-be-greater-than-2-in-q1-2025|Will U.S. GDP growth be greater than 2% in Q1 2025?]] `macro_release` 56d
- ✅ [[cases/gold_48_european-central-bank-cuts-rates-in-oct-meeting|European Central Bank cuts rates in Oct meeting?]] `other` 34d
- ✅ [[cases/gold_71_will-the-next-recession-in-the-us-happen-by-q3-2|Will the next recession in the US happen by Q3 2022?]] `other` 287d
- ✅ [[cases/gold_75_will-the-next-recession-in-the-us-happen-by-q4-2|Will the next recession in the US happen by Q4 2022?]] `other` 379d
- ❌ [[cases/gold_92_btc-price-50000-1-hour-after-etf-approval|$BTC price >$50,000 1 hour after ETF approval?]] `other` 5d
- ❌ [[cases/gold_93_btc-price-between-42500-45000-1-hour-after-etf-a|$BTC price between $42,500-45,000 1 hour after ETF approval?]] `other` 4d
- ❌ [[cases/gold_95_sec-to-not-approve-spot-bitcoin-etf-by-jan-10|SEC to NOT approve spot Bitcoin ETF by Jan 10?]] `other` 1d
- ❌ [[cases/gold_19_fed-decreases-interest-rates-by-25-bps-after-jul|Fed decreases interest rates by 25 bps after July 2024 meeting?]] `rate_decision` 48d
- ❌ [[cases/gold_27_fed-increases-interest-rates-by-25-bps-after-jul|Fed increases interest rates by 25+ bps after July 2024 meeting?]] `rate_decision` 48d
- ✅ [[cases/gold_44_will-the-turkish-central-bank-raise-interest-rat|Will the Turkish Central Bank raise interest rates at its June meeting?]] `rate_decision` 15d
- ❌ [[cases/gold_74_will-the-fed-raise-rates-again-in-2023|Will the Fed raise rates again in 2023?]] `rate_decision` 116d
- ❌ [[cases/gold_78_will-the-fed-decrease-interest-rates-by-25-bps-a|Will the Fed decrease interest rates by 25 bps after its January meeting?]] `rate_decision` 53d
- ✅ [[cases/gold_79_will-the-fed-raise-interest-rates-by-0-bps-after|Will the Fed raise interest rates by 0 bps after its December meeting?]] `rate_decision` 49d
- ❌ [[cases/gold_81_will-the-fed-raise-interest-rates-by-25-bps-afte|Will the Fed raise interest rates by 25 bps after its December meeting?]] `rate_decision` 50d
- ✅ [[cases/gold_87_will-the-fed-raise-interest-rates-by-0-bps-after|Will the Fed raise interest rates by 0 bps after its January meeting?]] `rate_decision` 53d
- ✅ [[cases/gold_31_ethereum-etf-begins-trading-by-july-26|Ethereum ETF begins trading by July 26?]] `regulatory_approval` 24d
- ✅ [[cases/gold_32_sec-approves-first-spot-bitcoin-etf-on-jan-10|SEC approves first spot Bitcoin ETF on Jan 10?]] `regulatory_approval` 1d
- ✅ [[cases/gold_34_ethereum-spot-etf-approved-by-june-30|Ethereum spot ETF approved by June 30?]] `regulatory_approval` 177d
- ❌ [[cases/gold_94_will-sec-delay-bitcoin-etf-decision|Will SEC delay Bitcoin ETF decision?]] `regulatory_approval` 0d
- ❌ [[cases/gold_96_will-the-sec-approve-blackrocks-bitcoin-etf-by-d|Will the SEC approve BlackRock's Bitcoin ETF by December 31?]] `regulatory_approval` 170d
- ❌ [[cases/gold_100_will-the-sec-approve-blackrocks-bitcoin-etf-by-a|Will the SEC approve BlackRock's Bitcoin ETF by August 31?]] `regulatory_approval` 62d

## Geopolitics (19 cases)

- ✅ [[cases/gold_01_israel-x-iran-ceasefire-before-july|Israel x Iran ceasefire before July?]] `ceasefire` 15d
- ✅ [[cases/gold_02_russia-x-ukraine-ceasefire-by-may-31-2026|Russia x Ukraine ceasefire by May 31, 2026?]] `ceasefire` 60d
- ✅ [[cases/gold_03_will-israel-first-announce-ceasefire-on-october|Will Israel first announce ceasefire on October 8?]] `ceasefire` 5d
- ❌ [[cases/gold_04_israel-x-hamas-ceasefire-by-july-15|Israel x Hamas ceasefire by July 15?]] `ceasefire` 18d
- ✅ [[cases/gold_09_russia-x-ukraine-ceasefire-before-2027|Russia x Ukraine ceasefire by end of 2026?]] `ceasefire` 524d
- ❌ [[cases/gold_10_russia-x-ukraine-ceasefire-in-2024|Russia x Ukraine Ceasefire in 2024?]] `ceasefire` 105d
- ❌ [[cases/gold_18_will-israel-first-announce-ceasefire-on-october|Will Israel first announce ceasefire on October 9?]] `ceasefire` 1d
- ✅ [[cases/gold_23_israel-announces-ceasefire-by-tomorrow|Israel announces ceasefire by January 17?]] `ceasefire` 0d
- ✅ [[cases/gold_30_israel-announces-ceasefire-by-sunday|Israel announces ceasefire by Sunday?]] `ceasefire` 2d
- ❌ [[cases/gold_46_israel-x-hamas-ceasefire-before-march|Israel x Hamas ceasefire before March?]] `ceasefire` 31d
- ✅ [[cases/gold_56_israel-and-hamas-ceasefire-in-2023|Israel and Hamas ceasefire in 2023?]] `ceasefire` 82d
- ❌ [[cases/gold_60_will-dpp-win-a-majority-in-the-2024-taiwanese-ge|Will DPP (民進黨) win a majority in the 2024 Taiwanese General Election?]] `election` 38d
- ❌ [[cases/gold_38_will-china-gdp-growth-in-q1-2026-be-between-5pt5|Will China GDP growth in Q1 2026 be between 5.5% and 6.0%?]] `macro_release` 84d
- ✅ [[cases/gold_40_will-china-gdp-growth-in-q1-2026-be-between-5pt0|Will China GDP growth in Q1 2026 be between 5.0% and 5.5%?]] `macro_release` 84d
- ❌ [[cases/gold_42_will-china-gdp-growth-in-q4-2025-be-less-than-2p|Will China GDP growth in Q4 2025 be less than 2.5%?]] `macro_release` 52d
- ❌ [[cases/gold_47_will-china-gdp-growth-in-q4-2025-be-over-5pt0|Will China GDP growth in Q4 2025 be over 5.0%?]] `macro_release` 52d
- ❌ [[cases/gold_50_will-israel-attack-iran-by-february-15|Will Israel attack Iran by February 15?]] `military_strike` 20d
- ❌ [[cases/gold_63_trump-blames-biden-for-iran-israel-escalation-by|Trump blames Biden for Iran-Israel escalation by Friday?]] `other` 4d
- ❌ [[cases/gold_86_will-ukraine-join-nato-before-july|Will Ukraine join NATO before July?]] `other` 191d

## Politics (49 cases)

- ❌ [[cases/gold_25_will-another-man-be-the-2024-democratic-vp-nomin|Will another man be the 2024 Democratic VP nominee?]] `appointment` 28d
- ❌ [[cases/gold_26_will-another-woman-be-the-2024-democratic-vp-nom|Will another woman be the 2024 Democratic VP nominee?]] `appointment` 28d
- ❌ [[cases/gold_16_trump-sentenced-to-between-12-and-23-months-pris|Trump sentenced to between 12 and 23 months prison time?]] `court_ruling` 40d
- ❌ [[cases/gold_17_trump-sentenced-to-between-24-and-35-months-pris|Trump sentenced to between 24 and 35 months prison time?]] `court_ruling` 40d
- ❌ [[cases/gold_97_will-trump-testify-in-hush-money-trial|Will Trump testify in hush money trial?]] `court_ruling` 191d
- ❌ [[cases/gold_05_will-fit-u-hold-the-most-seats-in-the-chamber-of|Will FIT-U hold the most seats in the Chamber of Deputies following the 2025 Arg]] `election` 262d
- ❌ [[cases/gold_06_will-hnp-hold-the-most-seats-in-the-chamber-of-d|Will HNP hold the most seats in the Chamber of Deputies following the 2025 Argen]] `election` 262d
- ❌ [[cases/gold_07_will-hnp-win-the-most-seats-in-the-chamber-of-de|Will HNP win the most seats in the Chamber of Deputies following the 2025 Argent]] `election` 262d
- ✅ [[cases/gold_08_will-lla-win-the-most-seats-in-the-chamber-of-de|Will LLA win the most seats in the Chamber of Deputies following the 2025 Argent]] `election` 262d
- ❌ [[cases/gold_13_will-trump-drop-out-of-presidential-race|Will Trump drop out of presidential race?]] `election` 145d
- ✅ [[cases/gold_14_will-biden-drop-out-of-presidential-race|Biden drops out of presidential race?]] `election` 409d
- ✅ [[cases/gold_15_trump-election-interference-trial-doesnt-start-b|Trump election interference trial doesn't start before November?]] `election` 257d
- ✅ [[cases/gold_21_will-edmundo-gonzalez-win-the-2024-venezuela-pre|Will Edmundo González win the 2024 Venezuela presidential election?]] `election` 57d
- ❌ [[cases/gold_22_will-nicolas-maduro-win-the-2024-venezuela-presi|Will Nicolas Maduro Win the 2024 Venezuela presidential election?]] `election` 57d
- ✅ [[cases/gold_28_taiwan-presidential-election-will-lai-ching-te-w|Taiwan Presidential Election: Will Lai Ching-te win?]] `election` 78d
- ❌ [[cases/gold_29_taiwan-presidential-election-will-ko-wen-je-win|Taiwan Presidential Election: Will Ko Wen-je win?]] `election` 45d
- ❌ [[cases/gold_41_other-party-wins-the-most-seats-in-next-uk-elect|Other party wins the most seats in next UK election?]] `election` 272d
- ❌ [[cases/gold_49_will-reform-win-7-seats-in-uk-election|Will Reform win 7+ seats in UK Election?]] `election` -14d
- ❌ [[cases/gold_53_will-rnuxd-win-the-most-seats-in-the-french-elec|Will RN/UXD win the most seats in the French election?]] `election` 12d
- ❌ [[cases/gold_54_reform-wins-the-most-seats-in-next-uk-election|Reform wins the most seats in next UK election?]] `election` 272d
- ✅ [[cases/gold_57_will-nfp-win-the-most-seats-in-the-french-electi|Will NFP win the most seats in the French election?]] `election` 12d
- ✅ [[cases/gold_65_will-trumps-ny-sentencing-be-delayed-past-electi|Trump's NY sentencing delayed past election?]] `election` 60d
- ✅ [[cases/gold_73_conservatives-win-the-second-most-seats-in-next|Conservatives win the second most seats in next UK election?]] `election` 23d
- ✅ [[cases/gold_76_will-nigel-farage-win-election-to-uk-parliament|Will Nigel Farage win election to UK parliament?]] `election` 30d
- ❌ [[cases/gold_77_reform-wins-the-second-most-seats-in-next-uk-ele|Reform wins the second most seats in next UK election?]] `election` 23d
- ❌ [[cases/gold_84_will-trump-election-interference-trial-start-in|Will Trump election interference trial start in September or October?]] `election` 256d
- ❌ [[cases/gold_98_supreme-court-unanimous-vote-in-trump-immunity-c|Supreme Court unanimous vote in Trump immunity case?]] `election` 191d
- ✅ [[cases/gold_11_us-government-shutdown-before-2025|Will there be a US Government shutdown?]] `government_action` 118d
- ❌ [[cases/gold_51_debt-ceiling-abolished-before-trump-inauguration|Debt ceiling abolished before Trump inauguration?]] `government_action` 30d
- ❌ [[cases/gold_52_will-there-be-a-us-government-shutdown-by-novemb|Will there be a US government shutdown by November 19?]] `government_action` 45d
- ❌ [[cases/gold_61_will-there-be-a-us-government-shutdown-by-jan-20|Will there be a US government shutdown by Jan 20?]] `government_action` 8d
- ✅ [[cases/gold_80_will-us-unemployment-be-4pt1-or-lower-in-decembe|Will US unemployment be 4.1% or lower in December 2024?]] `macro_release` 29d
- ❌ [[cases/gold_82_will-us-unemployment-be-4pt4-or-higher-in-decemb|Will US unemployment be 4.4% or higher in December 2024?]] `macro_release` 29d
- ✅ [[cases/gold_20_will-biden-drop-out-before-the-democratic-nation|Biden drops out before the Democratic convention?]] `other` 52d
- ❌ [[cases/gold_35_joe-biden-reinstated-as-democratic-nominee-at-dn|Joe Biden reinstated as Dem Nominee at DNC?]] `other` 15d
- ✅ [[cases/gold_37_will-opec-hike-production-by-next-meeting|Will OPEC hike production by next meeting?]] `other` 29d
- ❌ [[cases/gold_43_milei-out-as-president-of-argentina-in-2025|Milei out as President of Argentina in 2025?]] `other` 330d
- ❌ [[cases/gold_45_will-scotus-block-trumps-hush-money-sentencing|Will SCOTUS block Trump's hush money sentencing?]] `other` 10d
- ❌ [[cases/gold_66_trumps-january-10-sentencing-pushed-back|Trump's January 10 sentencing pushed back?]] `other` 6d
- ❌ [[cases/gold_69_milei-out-as-president-of-argentina-before-july|Milei out as President of Argentina before July?]] `other` 146d
- ❌ [[cases/gold_70_will-house-and-senate-pass-funding-bill-by-midni|Will House and Senate pass funding bill by midnight?]] `other` 0d
- ✅ [[cases/gold_72_will-keir-starmer-be-next-uk-prime-minister|Will Keir Starmer be next UK prime minister?]] `other` 42d
- ❌ [[cases/gold_83_will-suella-braverman-be-next-uk-prime-minister|Will Suella Braverman be next UK prime minister?]] `other` 42d
- ✅ [[cases/gold_85_democrats-nominate-prez-candidate-by-aug-7|Democrats nominate Prez candidate by August 7?]] `other` 18d
- ❌ [[cases/gold_88_will-brics-add-a-new-member-by-december-31|Will BRICS add a new member by December 31?]] `other` 156d
- ✅ [[cases/gold_89_trumps-november-26-sentencing-pushed-back|Trump's November 26 sentencing pushed back?]] `other` 19d
- ❌ [[cases/gold_90_will-a-country-leave-brics-before-the-end-of-202|Will a country leave BRICS before 2026?]] `other` 174d
- ❌ [[cases/gold_91_democrats-nominate-prez-candidate-august-19-22|Democrats nominate Prez candidate August 19-22?]] `other` 18d
- ✅ [[cases/gold_99_will-saudi-arabia-accept-invitation-to-join-bric|Will Saudi Arabia accept invitation to join BRICS?]] `other` 73d

## Technology (9 cases)

- ❌ [[cases/gold_24_will-supreme-court-delay-the-tiktok-ban|Will Supreme Court delay the Tiktok ban?]] `court_ruling` 9d
- ❌ [[cases/gold_33_will-the-next-model-released-by-openai-debut-at|Will the next model released by OpenAI debut at a score of at least 1480?]] `other` 70d
- ❌ [[cases/gold_36_will-the-next-model-released-by-openai-debut-at|Will the next model released by OpenAI debut at a score of at least 1500?]] `other` 70d
- ❌ [[cases/gold_58_will-the-central-bank-of-colombia-announce-a-dec|Will the Central Bank of Colombia announce a decrease at the March meeting?]] `other` 97d
- ❌ [[cases/gold_62_will-the-central-bank-of-colombia-announce-a-dec|Will the Central Bank of Colombia announce a decrease at the April meeting?]] `other` 88d
- ❌ [[cases/gold_64_will-the-central-bank-of-colombia-announce-an-in|Will the Central Bank of Colombia announce an increase at the April meeting?]] `other` 88d
- ✅ [[cases/gold_67_will-the-central-bank-of-colombia-announce-an-in|Will the Central Bank of Colombia announce an increase at the March meeting?]] `other` 97d
- ✅ [[cases/gold_12_tiktok-banned-in-the-us-before-may-2025|TikTok banned in the US before May 2025?]] `regulatory_approval` 224d
- ❌ [[cases/gold_68_tiktok-banned-in-the-us-by-june-30|TikTok banned in the US by June 30?]] `regulatory_approval` 184d
