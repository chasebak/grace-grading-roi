# GRACE findings (generated)

Written by `python -m grace.report` from the committed synthetic data. Do not edit:
`python -m grace.report --check` fails when this file is stale.

## Headline

**Oracle value captured at a $3,000 budget (regular tier, synthetic truth): GRACE 0.857, against 0.336 for a common multiplier heuristic (V10/raw, no gates) and 0.120 for random liquid cards.** 1.000 is the best batch an exact-knapsack oracle can buy knowing the true rates and prices ([reports/numbers.json:1323](reports/numbers.json)).

## KPIs

| KPI | formula | why it matters | compared against | what the software does to it | measured on | value now |
|---|---|---|---|---|---|---|
| **Oracle value captured** (headline; $3,000 budget, regular tier) | True-expected NPV of the rule's batch / that of the best feasible batch | How much of the knowable money the ranking finds | A common multiplier heuristic (V10/raw, no gates); random liquid cards; oracle = exact knapsack | GRACE should sit far above the heuristic and below 1.0 | synthetic truth | GRACE 0.857, heuristic 0.336, random 0.120 (reports/numbers.json:1323) |
| Regret and NPV per budget dollar | oracle $ - rule $; rule $ / budget B (not / dollars spent) | Dividing by B means underspending is never rewarded | The heuristic and random; gated NPV per dollar (a = 1); every rule is in the Experiment 4 table | Lower regret than the heuristic and random | synthetic truth | regret GRACE $3,163, heuristic $14,657, random $19,441, gated NPV per dollar $2,779; $ per budget $ GRACE 6.31, heuristic 2.47, random 0.88, gated NPV per dollar 6.43, oracle 7.36; gated NPV per dollar (a = 1) beats GRACE (a = 0.5) here and the IRR rule ties it; a stays 0.5 per DECISIONS.md (reports/numbers.json:1351) |
| Pick optimism | Mean predicted p10 - mean true p10 of the picks; predicted batch $ - true batch $ (shown as a share of true) | Prices the pop-bias caveat on the cards you actually buy | Heuristic picks | Reported, never tuned away | synthetic truth | p10 GRACE +0.043 vs heuristic +0.056; batch $ GRACE +3.8% vs heuristic +77.6% (reports/numbers.json:1330) |
| Corruption leakage | Planted-error cards in the batch / batch size, and their share of capital | Gate recall turned into its business result | Heuristic; the liquid universe's planted base rate | Below the base rate | synthetic truth | GRACE 1/20 picks, 4.0% of capital; heuristic 21/28, 78.5% of capital; liquid base rate 67/451 (reports/numbers.json:1344) |
| Gem-rate log loss gain vs m = 40 | Paired per-fold difference in binomial log loss, estimator - m = 40 (negative = better; plus Brier and ECE), 5-fold CV over cards | A proper scoring rule grades the probabilities themselves | Fixed prior, era mean, raw rate, empirical Bayes and its two cross arms | Shows whether pooling beats no pooling | out-of-sample on the model's own generator | next pop: empirical Bayes -0.0023 ± 0.0011 (5/5), better than m = 40, as are m = 40 toward the era mean and moment m toward the fixed prior; unscreened: empirical Bayes +0.0067 ± 0.0013 (0/5), worse than m = 40, and no estimator beats m = 40 there (fixed prior, the lowest mean log loss, is a null) (reports/numbers.json:1278) |

> **Why this metric.** Of the money a perfectly informed buyer could make with this budget, how much does the ranking find?
> *Kind:* Simulated: closed-form expected NPV at the generator's planted gem rates and true prices (synthetic truth), divided by the exact-knapsack oracle's batch.
> *Could lie if:* It would lie if the oracle were not the best batch (guard: the oracle-bound tests in tests/test_backtest.py); it scores the generator against itself, so the rule order is the claim, not the dollars.

**Guarantees (tests).**

- The oracle is an exact integer-dollar knapsack bracketed between a feasible batch and an upper bound, and no rule's true NPV ever exceeds the upper bound (`tests/test_backtest.py`).
- No rule overspends its budget, and every per-dollar figure divides by the budget (`tests/test_backtest.py`).
- `generate()` reproduces `synth/*.csv` byte for byte, and capturing true prices adds no random draws (`tests/test_golden_data.py`).
- Every number in this README's generated blocks equals `reports/numbers.json`, which equals a fresh run (`tests/test_readme_drift.py`, `tests/test_report.py`).

## Experiment 4: decision backtest

Oracle value captured (true NPV of the batch / the oracle's):

| rule | $1,000 | $3,000 | $10,000 |
|---|---:|---:|---:|
| GRACE (NPV / sqrt(capital), gated) | 0.873 | 0.857 | 0.880 |
| Gated NPV | 0.367 | 0.691 | 0.809 |
| Gated NPV per dollar | 0.959 | 0.874 | 0.888 |
| IRR rule | 0.959 | 0.874 | 0.888 |
| A common multiplier heuristic (V10/raw, no gates) | 0.229 | 0.336 | 0.647 |
| Random liquid cards | 0.090 | 0.120 | 0.205 |
| Oracle (exact knapsack) | 1.000 | 1.000 | 1.000 |

> **Why this metric.** Of the money a perfectly informed buyer could make with this budget, how much does the ranking find?
> *Kind:* Simulated: closed-form expected NPV at the generator's planted gem rates and true prices (synthetic truth), divided by the exact-knapsack oracle's batch.
> *Could lie if:* It would lie if the oracle were not the best batch (guard: the oracle-bound tests in tests/test_backtest.py); it scores the generator against itself, so the rule order is the claim, not the dollars.

Every rule at the $3,000 budget:

| rule | regret | true NPV per budget $ | planted picks | planted capital share | pick optimism (p10) | pick optimism ($) |
|---|---:|---:|---:|---:|---:|---:|
| GRACE (NPV / sqrt(capital), gated) | $3,163 | 6.31 | 1/20 | 4.0% | +0.043 | $723 (+3.8%) |
| Gated NPV | $6,817 | 5.09 | 0/12 | 0.0% | +0.025 | $72 (+0.5%) |
| Gated NPV per dollar | $2,779 | 6.43 | 1/24 | 4.0% | +0.048 | $715 (+3.7%) |
| IRR rule | $2,779 | 6.43 | 1/24 | 4.0% | +0.048 | $715 (+3.7%) |
| A common multiplier heuristic (V10/raw, no gates) | $14,657 | 2.47 | 21/28 | 78.5% | +0.056 | $5,761 (+77.6%) |
| Random liquid cards | $19,441 | 0.88 | 3.4/22.7 (mean) | 13.8% | +0.062 | $988 (+117.1%) |
| Oracle (exact knapsack) | $0 | 7.36 | 1/22 | 3.9% | +0.035 | $549 (+2.5%) |

> **Why this metric.** How many dollars does each rule leave on the table at this budget?
> *Kind:* Simulated: oracle true NPV minus the rule's true NPV, and the rule's true NPV divided by the budget B (synthetic truth).
> *Could lie if:* Dividing by dollars spent would reward a rule that leaves cash idle, and an overspent budget would inflate it (guards: test_denominator_is_the_budget, test_no_rule_overspends_the_budget).

> **Why this metric.** On the cards a rule actually buys, how much rosier is the engine than the truth?
> *Kind:* Simulated: mean predicted p10 minus mean planted p10 of the picks, and predicted batch NPV minus true batch NPV (synthetic truth).
> *Could lie if:* It would hide the pop-bias caveat if the truth reused the selection-boosted or predicted rates; every universe row's true NPV is recomputed from the planted latent rate and true prices, and a control shows the predicted rate gives other values (guard: test_universe_true_npv_is_recomputed_from_the_truth). It is reported, never tuned away.

> **Why this metric.** How much of the budget goes to cards whose feed prices are junk?
> *Kind:* Counted: planted-error cards in the batch over batch size, and their share of the capital spent, against the liquid universe's planted base rate.
> *Could lie if:* It would flatter the gates if junk cards were scored at their junk prices; they are scored at the base card's true prices (guard: test_junk_plants_true_prices_are_the_base_card).

Oracle bracket (feasible batch vs knapsack upper bound): $1,000: feasible $9,533, upper bound $9,569 (width 0.37%); $3,000: exact at $22,081; $10,000: feasible $43,222, upper bound $43,367 (width 0.34%).

- **Universe:** 451 liquid cards, 67 of them planted errors (base rate 0.149); 169 pass the gates, 13 of those planted.
- **Order holds across the sweep:** GRACE beats the heuristic at 28 of 28 budgets ($250-$20,000), the heuristic beats random at 28 of 28, and GRACE's leakage sits below the base rate at 28 of 28.
- **a = 1 beats the shipped a = 0.5 under a cash budget:** gated NPV per dollar captures 0.874 against GRACE's 0.857 at $3,000, and has more true NPV at 24 of 28 budgets. a stays 0.5 (DECISIONS.md); any switch gets its own commit.
- **The IRR rule picks the same batch as gated NPV per dollar** at 28 of 28 budgets: at one tier the cash cycle T is the same for every card, so annualized ROI is a monotone function of NPV / capital.
- **A greedy oracle is not an upper bound:** filling by absolute true NPV, some rule beats it by up to $4,097 on the sweep, which is why the oracle is an exact knapsack.

> **Why this metric.** Does the rule order hold at every budget, or only at the one quoted?
> *Kind:* Simulated: true NPV per budget dollar at each solved budget of the sweep, each budget filled on its own (greedy fills are not nested).
> *Could lie if:* A sweep cell that differed from the standalone run would mean two different backtests (guard: test_sweep_cell_equals_standalone_run).

## Experiment 1: the gates as a classifier

```
pathology      planted caught  recall
wrong_card          25     23    0.92
fake_grail          15     14    0.93
thin_outlier        15     12    0.80
bad_raw             30     21    0.70
flag precision (flagged cards that were planted): 0.69
drop precision (dropped cards that were planted): 0.64
clean cards flagged: 31/500
genuine grails kept: 15/15
borderline exempt grails kept: 16/16
false-positive cost: 31 clean cards dropped, 24 with positive shadow NPV, $9,584 NPV forgone
```

> **Why this metric.** Do the gates catch junk feed rows without killing real grails?
> *Kind:* Counted: gate decisions against the generator's planted labels (synthetic truth), as recall per pathology and flag precision.
> *Could lie if:* Plants that never approach a threshold would make recall and precision 1.00 by construction; plants straddle the gates on both sides (guard: tests/test_experiments.py, tests/test_gates.py).

## Experiment 2: is the fixed m = 40 reasonable?

```
beta-binomial method-of-moments pseudo-count (vs fixed m=40):
  mid      m_hat = 26.6
  modern   m_hat = 28.8
  vintage  m_hat = 17.4
held-out cards: 165
mean (predicted - realized) gem rate: +0.059
  low-value cards:  +0.076
  high-value cards: +0.042
prior sensitivity (Spearman of GRACE ranking vs default priors):
  era priors -0.10: rho = 0.995
  era priors +0.10: rho = 0.994
```

> **Why this metric.** Are the pooled gem rates honest for a card nobody pre-screened?
> *Kind:* Simulated: pooled p10 against a fresh simulated submission at each held-out card's latent rate (out-of-sample on the model's own generator).
> *Could lie if:* One 30% holdout with one seed can be lucky; the planted selection effect must show as predictions above realized (guard: test_calibration_shows_planted_optimism).

## Experiment 3: are the gem-rate probabilities any good?

600 cards (46 with empty ladders), 5 folds over cards (120/120/120/120/120 test cards); outcome = one fresh 20-card submission per card; log loss clipped to [1e-06, 1 - 1e-06]; ECE over 10 equal-width bins. Each score is mean ± sd over the folds; "vs m = 40" is the paired per-fold difference (estimator - m = 40, negative = better) with the folds in which the estimator beat m = 40; a gap under one fold sd is a null.

Target: next pop submissions (observed, selection-boosted rate).

| estimator | log loss | vs m = 40 | verdict | Brier | ECE | clipped |
|---|---:|---:|---|---:|---:|---:|
| Fixed era prior | 0.6355 ± 0.0094 | +0.0157 ± 0.0034 (0/5 folds better) | worse | 0.2228 ± 0.0046 | 0.0518 ± 0.0075 | 0 |
| Era mean (training folds) | 0.6307 ± 0.0074 | +0.0108 ± 0.0012 (0/5 folds better) | worse | 0.2205 ± 0.0036 | 0.0238 ± 0.0075 | 0 |
| Raw rate n10 / n | 0.6436 ± 0.0214 | +0.0238 ± 0.0241 (0/5 folds better) | null | 0.2172 ± 0.0025 | 0.0271 ± 0.0059 | 7 |
| Shipped pooling, m = 40 | 0.6199 ± 0.0067 | (baseline) | - | 0.2159 ± 0.0032 | 0.0333 ± 0.0084 | 0 |
| Cross arm: m = 40 toward the era mean | 0.6183 ± 0.0058 | -0.0015 ± 0.0010 (5/5 folds better) | better | 0.2151 ± 0.0028 | 0.0277 ± 0.0050 | 0 |
| Cross arm: m fit by moments, toward the fixed prior | 0.6188 ± 0.0068 | -0.0010 ± 0.0006 (5/5 folds better) | better | 0.2155 ± 0.0032 | 0.0309 ± 0.0083 | 0 |
| Empirical Bayes, m fit by moments | 0.6175 ± 0.0060 | -0.0023 ± 0.0011 (5/5 folds better) | better | 0.2148 ± 0.0028 | 0.0237 ± 0.0061 | 0 |
| True rate (reference, unknowable) | 0.6107 ± 0.0043 | -0.0092 ± 0.0034 (5/5 folds better) | reference | 0.2119 ± 0.0020 | 0.0217 ± 0.0049 | 0 |

Target: unscreened raw cards (latent rate).

| estimator | log loss | vs m = 40 | verdict | Brier | ECE | clipped |
|---|---:|---:|---|---:|---:|---:|
| Fixed era prior | 0.5885 ± 0.0093 | -0.0001 ± 0.0029 (2/5 folds better) | null | 0.2011 ± 0.0043 | 0.0409 ± 0.0120 | 0 |
| Era mean (training folds) | 0.5989 ± 0.0079 | +0.0103 ± 0.0046 (0/5 folds better) | worse | 0.2059 ± 0.0036 | 0.0783 ± 0.0147 | 0 |
| Raw rate n10 / n | 0.6388 ± 0.0223 | +0.0502 ± 0.0271 (0/5 folds better) | worse | 0.2106 ± 0.0047 | 0.0922 ± 0.0121 | 7 |
| Shipped pooling, m = 40 | 0.5886 ± 0.0084 | (baseline) | - | 0.2016 ± 0.0039 | 0.0637 ± 0.0079 | 0 |
| Cross arm: m = 40 toward the era mean | 0.5946 ± 0.0076 | +0.0060 ± 0.0016 (0/5 folds better) | worse | 0.2043 ± 0.0035 | 0.0821 ± 0.0105 | 0 |
| Cross arm: m fit by moments, toward the fixed prior | 0.5902 ± 0.0081 | +0.0016 ± 0.0009 (0/5 folds better) | worse | 0.2023 ± 0.0038 | 0.0676 ± 0.0076 | 0 |
| Empirical Bayes, m fit by moments | 0.5953 ± 0.0075 | +0.0067 ± 0.0013 (0/5 folds better) | worse | 0.2047 ± 0.0034 | 0.0830 ± 0.0097 | 0 |
| True rate (reference, unknowable) | 0.5706 ± 0.0123 | -0.0180 ± 0.0043 (5/5 folds better) | reference | 0.1941 ± 0.0052 | 0.0126 ± 0.0031 | 0 |

> **Why this metric.** Are the gem-rate probabilities themselves any good, and does pooling beat no pooling?
> *Kind:* Simulated: 5-fold cross-validation over cards, scored by log loss, Brier and ECE against a fresh simulated submission per card (out-of-sample on the model's own generator), paired fold by fold against the shipped m = 40.
> *Could lie if:* Era means or m_hat fit on all cards would leak the test fold (guard: test_changing_the_test_fold_leaves_the_fit_unchanged); a gap under one fold sd is printed as a null, and the log-loss clip count is shown so a 0-or-1 prediction cannot hide.

- **Fold fits:** training-fold m_hat runs mid 25.6-29.7, modern 24.8-32.0, vintage 15.8-19.0, so the fixed m = 40 shrinks harder than every fold's fit. The fixed era priors (mid 0.30, modern 0.45, vintage 0.18) sit below every fold's era mean (mid 0.321-0.332, modern 0.498-0.509, vintage 0.195-0.210) in 3 of 3 eras.
- **Next pop:** the best real estimator by log loss is empirical Bayes (0.6175 ± 0.0060). Against m = 40 (folds better): fixed prior worse (0/5), era mean worse (0/5), raw rate null (0/5), m = 40 toward the era mean better (5/5), moment m toward the fixed prior better (5/5), empirical Bayes better (5/5). The unknowable true rate sits 0.0092 below m = 40, the room any estimator has left.
- **Unscreened:** the best real estimator by log loss is fixed prior (0.5885 ± 0.0093). Against m = 40 (folds better): fixed prior null (2/5), era mean worse (0/5), raw rate worse (0/5), m = 40 toward the era mean worse (0/5), moment m toward the fixed prior worse (0/5), empirical Bayes worse (0/5). The unknowable true rate sits 0.0180 below m = 40, the room any estimator has left.
- **Clipping:** the log-loss clip moved raw rate predictions (7 on next pop, 7 on unscreened): ladders with no gems or all gems give a prediction of exactly 0 or 1. It moved none for fixed prior, era mean, m = 40, m = 40 toward the era mean, moment m toward the fixed prior, empirical Bayes.
- **m stays 40:** empirical Bayes against m = 40 is better on next pop and worse on unscreened. It changes two things at once, the shrink target (fixed prior to era mean) and the shrink strength (40 to m_hat); the cross arms split them: on next pop, the target alone -0.0015 ± 0.0010 (5/5), better, the strength alone -0.0010 ± 0.0006 (5/5), better; on unscreened, the target alone +0.0060 ± 0.0016 (0/5), worse, the strength alone +0.0016 ± 0.0009 (0/5), worse. m stays 40 (DECISIONS.md); any switch gets its own commit.

## Demo

```
cards scored:    600
kept:            515
dropped:         85
repaired raws:   16
rankable:        169
on frontier:     11
```

## Assumptions

| knob | value | source | what changes if wrong |
|---|---|---|---|
| Regular tier fee and turnaround | $79.99, 1.2 months | published default (PSA public fee schedule; rough public turnaround) | Every capital figure moves with the fee; the effect on rule order is to measure. |
| Time to sell after grading | 1.5 months | illustrative | Each extra month discounts proceeds by about the monthly rate. |
| Annual discount rate | 0.12 | illustrative | Higher rates shrink every NPV and favor faster tiers. |
| Selling fee and ship/pack | 0.13 of the sale, $5.00 per sale | illustrative | Scales every net sale; cards near the NPV floor flip first. |
| Pooling pseudo-count m | 40 | design choice (method of moments measures 17.4-28.8 on this data) | Shrinks gem rates harder or softer toward the era prior. |
| Era priors | modern 0.45 / mid 0.30 / vintage 0.18 | illustrative (sensitivity-analyzed, Experiment 2) | A +/-0.10 shift barely reorders the ranking (Spearman >= 0.994). |
| GRACE capital exponent a | 0.5 | design choice | a = 1 is NPV per dollar; Experiment 4 prices the difference. |
| Rankability gates | NPV >= $25.00, >= 3 graded-10 sales | design choice | Looser gates admit more cards and more planted errors. |
| Selection boost (generator) | 0.18 * exp(-V10 / 400) | illustrative | Sets how optimistic pop-derived gem rates are; 0 is the null control. |
| Nine-share of non-gem copies (generator and truth) | 0.6 | design choice | The V9 leg's weight in true NPV; the truth reuses the generator's value. |
| True price of a junk-feed plant | the base card it replaced | design choice | Junk picks would score differently, and leakage counts could shift too: fills are charged at true capital, so a junk card's true price changes which cards fit. |
| Backtest universe | liquid cards only (>= 3 graded-10 sales) | design choice | Illiquid cards would need a time-to-sell model the repo lacks. |
| Resale noise and time to sell | not modelled | design choice | Real batches spread around the expected values shown. |
| Budget fill | greedy down each rule's ranking, charged at the price paid | design choice (greedy is not optimal; the oracle measures the gap) | Charging predicted prices overspends real budgets (DECISIONS.md). |
| Experiment 3 outcome and folds | one 20-card submission per card, 5 folds over cards (seeds 3003, 3011) | design choice | A larger submission lowers outcome noise; fewer folds widen the fold sd. |
| Log-loss clip | [1e-06, 1 - 1e-06] | design choice | Sets what a 0-or-1 prediction costs; only the clipped raw-rate cards move. |
| Random rule | mean of 200 seeded shuffles (seed 4004) | design choice | A single shuffle would make the random row noisy. |
