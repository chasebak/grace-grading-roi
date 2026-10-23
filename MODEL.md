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
