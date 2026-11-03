# grace-grading-roi

Should you grade this card? NPV-per-dollar ranking with partial-pooled gem rates, data-quality gates, and a Pareto frontier — with the gates tested as a classifier.

[![CI](https://github.com/chasebak/grace-grading-roi/actions/workflows/ci.yml/badge.svg)](https://github.com/chasebak/grace-grading-roi/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Grade-or-not Pareto frontier on synthetic data](figures/frontier.png)

```bash
git clone https://github.com/chasebak/grace-grading-roi && cd grace-grading-roi
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-ci.txt
python -m pytest -q
python -m grace.demo
```

Supported platforms: macOS arm64 and Linux. The demo reads the committed
synthetic data in `synth/`, writes a frontier plot and a top-N table to the
untracked `out/`, and prints its own elapsed time (about a second).

Provenance: Public re-implementation of the grading-ROI engine from my private platform.

**Domain primer.** PSA grades trading cards on a 1-10 scale; a 10 ("gem mint")
often sells for a large multiple of a 9 or of the raw (ungraded) card. A pop
(population) report counts how many copies have been graded at each level —
the "pop ladder" — and this engine turns ladders into gem-rate estimates.

*(GRACE here is a grading capital-efficiency score — not the GRACE of
ML explainability, and not the cardiology GRACE risk score.)*

## What it computes

The headline is an **NPV/capital-efficiency ranking with an optional
mean-risk penalty**. For one card at one grading service tier:

```
Q   = p10*(V10*(1-f) - s) + p9*(V9*(1-f) - s) + p_le8*(Vle8*(1-f) - s)   expected net sale
R   = qty*(raw_cost + grading_fee) + other_fixed                          capital committed
T   = grading_months + sale_months
U   = 1 / (1 + r)^(T/12)                                                  discount factor
NPV = Q*qty*U - R
```

*(f = selling fee, s = ship/pack, r = annual discount rate, V10/V9/Vle8 =
expected sale price by grade.)*

Defaults (all configuration in `grace/config.py`): sale time 1.5 months,
annual discount rate 0.12, sell fee 13%, ship/pack $5. Tiers come from
PSA's [public published fee schedule](https://www.psacard.com/pricing)
(e.g. Regular $79.99 / ~1.2 mo, Express $149 / ~0.7 mo, Super Express
$299 / ~0.35 mo, Walk-Through $599 / ~0.25 mo). Tier choice is argmax NPV —
a small decision tree under uncertainty. Annualized ROI
`((Q*qty*U)/R)^(12/T) - 1` is computed but display-capped: a hover-style
metric, never the ranking. Marginal NPV versus selling raw today is
`npv + (raw - max(raw*(1-f) - s, 0))`.

**Grade legs.** V9 falls back to `0.5*V10` when unobserved (labeled
synthetic); V&le;8 defaults to the raw price — a sub-9 resells at roughly raw
while you still paid the grading fee.

**Gem rates: partial pooling.** Observed pop ladders shrink toward an era
base rate with a fixed pseudo-count `m = 40` (this is partial pooling with a
fixed pseudo-count, not "empirical Bayes" — m is not estimated from the data):

```
p10    = (n10 + m*mu_era) / (n_ladder + m)
share9 = (n9 + m/2) / (n9 + n_le8 + m)      splits the remaining mass
```

A card with no pop at all gets `p10 = mu_era`, rest 50/50, flagged `prior`.
The era defaults are illustrative era base rates, sensitivity-analyzed
(modern 0.45 / mid 0.30 / vintage 0.18). Public gem-rate statistics (e.g.
GemRate's published reports; PSA's public population pages) show era-level
differences of this order.

**The GRACE score.** `GRACE = (NPV - lambda*sigma) / R^a`, defaults
`lambda = 0`, `a = 0.5`: the default ranking is capital-efficiency
NPV/&radic;capital; lambda is an optional mean-risk penalty (in the sense of
Ogryczak & Ruszczyński [1]). `sigma` is the probability-weighted stdev
of per-grade net payoffs, computed on **discounted** payoffs so sigma and NPV
share one scale.

**Pop-bias caveat.** pop-derived gem rates are conditional on the submission
decision: submitters pre-screen condition, crack-and-resubmit 9s, and cross
over from other graders, so observed gem rates are an optimistic estimate of
P(10) for an unscreened raw card; the engine treats them as an upper bound,
partially offset by the raw-price gates.

## The gate ladder

Market feeds mismatch cards and carry junk prices, so plausibility gates run
before any scoring (`grace/gates.py`, in ladder order):

| # | Rule | Action |
|---|------|--------|
| 1 | V10 > 3.0&times; the third-party graded-10 sold median (median needs n&ge;3) | DROP `wrong_card_match` |
| 2 | V10/V9 > 60 (above the genuine ceiling, measured on production (private data)) | DROP `ratio_above_ceiling` |
| 3 | No V9, &le;1 lifetime feed sale, V10 > $200 | DROP `thin_outlier` |
| 4 | V10 &ge; $1,000 with zero graded-sold corroboration and &le;2 feed sales | DROP `unsupported_high_value` |
| 5 | Ungraded sold median (n&ge;3, discarded if &ge; V10) is primary truth: raw < median/3 repairs raw to the median (`raw_confidence=repaired`). No signal: structural floors — raw < 0.02&middot;V9 (or < 0.005&middot;V10 when no V9) repairs from a plausible near-mint price if available, else DROP | REPAIR / DROP `raw_below_floor_no_signal` |
| 6 | Post-repair raw < $1 | DROP `raw_below_one_dollar` |
| 7 | Effective multiplier > 100&times; with NO graded-sold corroboration (corroborated grails (genuinely very-high-multiplier cards) exempt) | DROP `uncorroborated_multiplier` |

After scoring, **rankability gates** hide rather than drop (the way a UI
filter would): NPV &ge; $25, liquidity &ge; 3 graded-10 sales within the
last 180 days (the 180-day window is a property of the input column
`grade10_sales_180d`, not an engine knob), and era-dependent raw floors
($1 modern / $2 mid / $2 vintage). The **Pareto frontier** is the
non-dominated staircase over (raw price, NPV) among gated cards: the cards
where no card as cheap or cheaper has a higher NPV.

Production scale, for context — production (private data): an adversarial
audit found raw-price errors up to 69,100&times;; plausibility gates cut the
scored universe 5,810 &rarr; 3,835.

## Experiment 1 — the gates as a classifier

"How do you know the gates didn't kill real grails?" The synthetic generator
(`grace/synth.py`, seeded) plants pathologies at known rows — wrong-card
matches, fake grails, thin outliers, corrupted raws — plus **genuine grails**
(80-300&times; multipliers WITH graded-sold corroboration) that the ladder
must keep. Plants deliberately straddle the gate thresholds (e.g. wrong-card
inflation drawn 2.2-9&times; against the 3&times; gate; ~20% of corrupted
raws divided by only 1.5-2.8&times;, too mild for the repair rules to see),
so recall measures sensitivity to borderline corruption rather than
rewarding easy cases. The generator also plants a **borderline-legit clean
subpopulation** (68 of the 500 clean cards) that precision is measured
against: hot-market cards whose graded-10 sold median lags the feed V10,
sparse-history new releases with no graded-sold record yet, honest V10/V9
ratios drawn into the 40-75 tail against the 60 ceiling, and corroborated
80-120&times; grails that exercise the multiplier exemption from the legit
side. Without cards like these, precision is 1.00 by construction — clean
cards that never approach a gate cannot be false-flagged, and the number
means nothing. `python -m grace.experiments.gates_eval` prints:

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

![Gate ladder precision and recall](figures/gates_classifier.png)

Every recall miss is a sub-threshold plant (wrong-card matches inflated
only 2.21&times; and 2.82&times; against the 3&times; gate; thin outliers
at $171-$197 against the $200 gate; a $931 fake-grail plant against the
$1,000 gate; corrupted raws divided by only 1.6-2.6&times;). Precision is
a real trade, not a free lunch: the ladder flags 31 of the 500 clean cards
— per the script's drop-reason breakdown, 12 hot-market cards whose
lagging sold median makes them look like wrong-card matches, 14 sparse new
releases the thin-outlier gate cannot tell from junk, and 5 honest
ratio-tail cards above the 60&times; ceiling. Those false positives cost a
buyer real money: 24 of the 31 carried positive shadow NPV, $9,584 of it
forgone on this draw — the price paid for 0.70-0.93 recall on planted
corruption at 0.69 flag precision. The corroboration exemption earns its
keep on both sides: all 15 genuine grails and all 16 borderline
80-120&times; grails survive.

## Experiment 2 — is the fixed m=40 reasonable?

`python -m grace.experiments.m_estimation` estimates the pooling pseudo-count
from the synthetic ladders' overdispersion with a beta-binomial
method-of-moments estimator, checks calibration on held-out cards, and
stress-tests the illustrative priors:

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

The fixed `m = 40` shrinks somewhat harder than the method-of-moments
estimates (17-29 on this data) — conservative toward the era mean. The
calibration plot shows predictions sitting above realized gem rates
(realized = a fresh simulated submission at each card's latent true rate),
most strongly for low-value cards: that is the planted submission-selection
effect, and it is exactly the optimism the pop-bias caveat above warns
about. Shifting every era prior by &plusmn;0.10 barely reorders the ranking
(Spearman &rho; &ge; 0.994), which is what "illustrative, sensitivity-analyzed"
means here.

![Calibration under the planted selection effect](figures/calibration.png)

## Demo output

`python -m grace.demo` on the committed synthetic set:

```
cards scored:    600
kept:            515
dropped:         85
repaired raws:   16
rankable:        169
on frontier:     11
```

## Design decisions / what didn't work

- **A flat multiplier cap didn't work.** An earlier production rule capped
  V10/raw at a flat 150&times;; on production (private data) it dropped ~265
  genuine high-multiplier cards and missed bad-raw cards. The current ladder
  replaces it with corroboration-aware rules: repair the raw when a sold
  median contradicts it, and only cap multipliers that nothing corroborates.
- **One discount scale.** The production engine computed sigma on
  undiscounted payoffs next to a discounted NPV; the public version computes
  both on discounted payoffs so they share one scale.
- **Risk penalty off by default.** Production ran risk-penalized variants;
  the public default is the plain capital-efficiency ranking (`lambda = 0`).
- **Drops before repairs.** Identity failures (wrong-card, thin, unsupported)
  drop before the repair step so a mismatched card is dropped for what it is
  rather than having its raw "repaired" toward an unrelated card's median;
  the post-repair rules (sub-$1 raw, multiplier cap) then run on the
  repaired raw.
- **Plants straddle the thresholds — on both sides.** Planting only
  clear-cut pathologies makes every recall 1.00 and tests nothing; the
  generator draws plants across the gate boundaries so the classifier
  numbers can actually move. An earlier draft of this repo had the same
  problem with precision: with no clean cards anywhere near a gate,
  precision was 1.00 by construction, so the generator now plants a
  borderline-legit clean subpopulation the ladder can actually lose.
- **The cheap slow tier keeps winning.** On this synthetic set argmax-NPV
  picks the cheapest tier for every rankable card: at a 12% annual discount
  rate, saving half a month of turnaround is worth ~0.5% of the discounted
  proceeds, which never
  beats a $69+ fee gap at these card values. Faster tiers only pay for
  far more expensive cards than this generator produces.

## What this does NOT claim

- **Not portfolio optimization.** GRACE is a single-asset mean-risk
  criterion; there are no covariances and no allocation step (hence the
  Ogryczak & Ruszczyński framing rather than Markowitz).
- **Not investment advice.** It is a decision-support ranking on noisy data.
- **The era priors are illustrative** rounded defaults, sensitivity-analyzed
  above; they carry no numeric source attribution.
- **Gem rates are submission-conditioned** (see the pop-bias caveat): the
  engine's P(10) is an upper bound for an unscreened raw card.
- Prices come only from a licensed third-party market-data feed in
  production; this repo ships synthetic data only.

More math background (derivations, the method-of-moments estimator, decision-
tree framing) lives in [MODEL.md](MODEL.md).

[1] W. Ogryczak and A. Ruszczyński, "From stochastic dominance to
mean-risk models: Semideviations as risk measures," *European Journal of
Operational Research* 116 (1999) 33-50.

---

Part of a set of public repos extracted from my card-business tooling — see my GitHub profile for the map.
