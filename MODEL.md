# Model notes

Background and derivations that would crowd the README. Everything here is
implemented in `grace/` and exercised by the tests.

## 1. The NPV engine as a decision tree

Grading one card at tier t is a lottery: pay `R = qty*(raw_cost + fee_t) +
other_fixed` now, receive one of three net payoffs at time `T_t =
grading_months_t + sale_months`:

| outcome | probability | payoff at T |
|---------|-------------|-------------|
| grade 10 | p10 | V10*(1-f) - s |
| grade 9 | p9 | V9*(1-f) - s |
| grade &le;8 | p_le8 | Vle8*(1-f) - s |

Discounting each branch by `U_t = (1+r)^(-T_t/12)` and netting the capital
gives `NPV_t = Q*qty*U_t - R_t`. Choosing the tier is then a one-stage
decision tree under uncertainty — pick `argmax_t NPV_t` (Raiffa, *Decision
Analysis*, 1968). Faster tiers buy a larger `U_t` at a larger `R_t`; at a 12%
annual rate the discounting gain is small unless the card value is large
relative to the fee gap.

Two auxiliary quantities:

- `annualized_roi = ((Q*qty*U)/R)^(12/T) - 1`. On short horizons tiny
  absolute edges annualize to absurd rates, so it is display-capped and never
  used for ranking.
- `npv_marginal = npv + (raw - raw_net_now)` with
  `raw_net_now = max(raw*(1-f) - s, 0)`: the margin of grading over selling
  the raw card today, since selling now has NPV `raw_net_now - raw` against
  the same cost basis.

## 2. Partial pooling, and what m means

Model each card's gem outcome as Binomial(n_ladder, p) with p drawn from an
era-level Beta(alpha, beta). The posterior mean of p after observing n10
gems is

```
E[p | ladder] = (n10 + alpha) / (n_ladder + alpha + beta)
```

Writing `m = alpha + beta` and `mu_era = alpha/m` gives exactly the engine's
`p10 = (n10 + m*mu_era) / (n_ladder + m)`: m is the prior's worth in
pseudo-observations. The engine FIXES m = 40 rather than estimating it,
which is why the docs say "partial pooling with a fixed pseudo-count" and
not "empirical Bayes".

The 9-vs-&le;8 split applies the same idea one level down with a flat 50/50
prior worth m/2 pseudo-observations on each side:
`share9 = (n9 + m/2) / (n9 + n_le8 + m)`.

## 3. Method-of-moments check on m

If p_i ~ Beta with mean mu and pseudo-count M, the observed gem rates
`phat_i = n10_i / n_i` satisfy

```
Var(phat_i) = mu*(1-mu) * (rho + (1-rho)/n_i),   rho = 1/(M+1)
```

With s2 the across-card sample variance of phat and `minv = mean(1/n_i)`:

```
rho_hat = (s2/(mu*(1-mu)) - minv) / (1 - minv),   M_hat = 1/rho_hat - 1
```

`grace/experiments/m_estimation.py` applies this per era (ladders with
n &ge; 5). On the committed synthetic data M_hat lands at 17-29 against the
fixed m = 40, i.e. the shipped engine shrinks somewhat more aggressively
toward the era mean than the data demands — a conservative choice under
selection-inflated ladders.

## 4. The mean-risk penalty

`GRACE = (NPV - lambda*sigma) / R^a` with sigma the probability-weighted
standard deviation of the discounted per-grade net payoffs. For lambda in
[0, 1] this is a classical mean-risk functional in the sense of Ogryczak &
Ruszczyński ("From stochastic dominance to mean-risk models," EJOR 116,
1999): mean minus a risk multiple, applied to a SINGLE lottery. There is no
covariance structure and no allocation step, so none of the portfolio-theory
machinery (efficient frontiers over weights, diversification) applies — the
Pareto frontier in this repo is over (raw price, NPV) across candidate
cards, which is a different object. The shipped default lambda = 0 reduces
GRACE to NPV/R^0.5: NPV per square-root dollar of capital, a concave
capital-efficiency ranking that favors cheaper cards at equal NPV.

## 5. Why the gates come before the probabilities matter

A wrong-card match or a corrupted raw price poisons every downstream number:
Q through V10, R through raw_cost, and the multiplier diagnostics through
both. Repairing the raw from the ungraded sold median (when n &ge; 3 and the
median is credibly below V10) fixes R and the V&le;8 leg in one move;
structural floors (2% of V9, 0.5% of V10) catch feeds whose raw field is
garbage with no sold signal to repair from. The uncorroborated-multiplier
cap at 100x runs on the post-repair raw, and graded-sold corroboration
exempts genuine grails — the committed experiment keeps 15/15 planted
genuine grails at 80-300x multipliers and 16/16 borderline-legit grails
at 80-120x.

## 6. Synthetic generator notes

- Latent era gem-rate means (0.40/0.25/0.14) sit deliberately BELOW the
  engine's illustrative priors (0.45/0.30/0.18): the priors describe
  observed, selection-boosted ladders, the latent rates describe an
  unscreened card.
- The selection boost is `0.18 * exp(-V10/400)`: about +0.14 at a $100 card,
  about +0.01 at a $1,100 card. The calibration experiment recovers this as
  a +0.076 predicted-minus-realized gap on low-value held-out cards versus
  +0.042 on high-value ones.
- Realized gem rates in the calibration experiment are fresh Binomial(20,
  latent_p10) draws per held-out card — a simulated future submission, not a
  re-read of the training ladder.

## 7. Assumptions

Every knob the numbers depend on, with its source (measured / published
default / illustrative / design choice) and what changes if it is wrong.
The values are read from the code by `python -m grace.report`, never typed.

<!-- grace.report:begin assumptions -->
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
<!-- grace.report:end assumptions -->

## 8. The decision backtest: truth model and oracle bracket

`grace/experiments/backtest.py` scores a batch of cards on synthetic truth.
For one card graded at the regular tier (qty 1):

```
p10      = latent (planted) gem rate of the card
p9       = (1 - p10) * s9,   p<=8 = (1 - p10) * (1 - s9)     s9 = the generator's nine-share
Q*       = p10*(V10*(1-f) - s) + p9*(V9*(1-f) - s) + p<=8*(raw*(1-f) - s)   at TRUE prices
R*       = raw + fee_regular + other_fixed                                  at the TRUE raw
true NPV = Q* * U - R*
```

The same `npv.py` functions compute it, so the truth differs from the
engine's prediction only where the world does: the planted gem rate instead
of the pooled, selection-boosted one, and the true prices instead of a junk
feed row. With p10 = 0 and V9 = raw, true NPV collapses to the &le;8 net
sale discounted, minus capital; `tests/test_backtest.py` checks that hand
case.

**Oracle bracket.** Choosing the batch with the most true NPV under budget
B is a 0/1 knapsack with real-valued weights R*. Solving it by dynamic
programming over whole dollars twice brackets the true optimum OPT:

```
ceil weights   ->  a batch that is feasible at the real prices  ->  value <= OPT
floor weights  ->  a relaxation every feasible batch fits        ->  OPT <= value
```

When the floor-weight optimum also fits at the real prices it is OPT
itself and the bracket closes. A greedy fill by true NPV per dollar, or by
absolute true NPV, is only a feasible batch; it is not an upper bound, and
on this data some rule beats the absolute-NPV fill (see DECISIONS.md).

**Why divide by B.** NPV per budget dollar is `true NPV / B`, never
`true NPV / dollars spent`: a rule that leaves cash idle must not look more
efficient for it.
