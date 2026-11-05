# Decisions

One dated entry per upgrade, written as "decided X because measurement Y;
alternative Z lost by N". Nulls and reversals stay in. Every measured
number below is printed by `python -m grace.report` into
`reports/numbers.json` (the page entries: by `python -m grace.viz build`
into `docs/data/`); `tests/test_readme_drift.py` fails if this file
quotes a number those commands no longer print.

## 2026-11-04 — Synthetic generator knobs and the golden-data pin

Decided to expose `selection_boost_max` and `multiplier_scale` as
`generate()` parameters whose defaults leave the random stream untouched,
because the null control (boost off) and stress runs need them and nothing
pinned the committed CSVs before. Measurement: `generate()` at the default
knobs writes `synth/cards.csv` and `synth/truth.csv` byte for byte, and
changing either knob changes the output (both checked in
`tests/test_golden_data.py`). Alternative: editing the module constants
from a script (the old prototype route); it lost because it changes the
generator for every later caller in the same process.

## 2026-11-04 — True prices are captured in memory, not written to truth.csv

Decided to capture each card's true prices inside
`synth.generate_world()` as plain copies of values already drawn, because
the decision backtest needs them and the committed CSVs must not move.
Measurement: `generate_world()` returns cards and truth equal to
`generate()`'s, so capturing adds no random draws
(`tests/test_golden_data.py`). Alternative: new truth.csv columns; it lost
because it would change committed data that the README numbers are pinned
to.

## 2026-11-04 — The oracle must be an exact knapsack

Decided the backtest's oracle is an exact integer-dollar 0/1 knapsack,
solved with capital rounded up (a feasible batch) and rounded down (an
upper bound), because only an upper bound makes "oracle value captured"
mean anything. Measurement: the bracket is exact at $22,081 at the $3,000
budget, and the widest bracket among the three budgets is 0.37%.
Alternative: a greedy fill by absolute true NPV, the scratch probe's
oracle; it lost because some rule beats a greedy fill by absolute true NPV
by $4,097 on the sweep, so it is not an upper bound.

## 2026-11-04 — Divide by the budget, not by dollars spent

Decided NPV per dollar is true NPV / budget B, because dividing by dollars
spent rewards a rule that leaves cash idle. Measurement (an honest null on
this data): switching to dollars spent changed the rule order at 0 of 28
budgets, but its largest inflation is 2.74 per dollar (GRACE at $250),
where few cards fit and part of the budget sits idle. The principle
stands; the measurement says it does not change the ranking here.

## 2026-11-04 — Charge the budget at the price actually paid

Decided every rule's greedy fill is charged each card's true capital (the
price paid at the till), because the oracle bound holds only when every
rule's batch is feasible at the same prices. Measurement: filling by the
engine's predicted capital instead puts 20 of 140 (rule, budget) cells over
budget at the real prices. Alternative: predicted capital; it lost by those
20 overspent cells.

## 2026-11-04 — The baseline is "a common multiplier heuristic"

Decided to call the V10/raw rule with no gates "a common multiplier
heuristic", because ranking by the graded-to-raw price multiple is a
common rule of thumb for picking cards to grade, and the name says what
the rule does without implying it is anyone's tuned method. It is the fair
baseline: the same feed prices, no gem-rate model and no gates.
Measurement: GRACE 0.857 against 0.336 for a common multiplier heuristic
at $3,000. Alternative: a bare formula label ("V10/raw"); it lost because
a reader would not know it stands for a rule people actually use (a
principle, not a measurement).

## 2026-11-04 — An IRR rule row

Decided to include an "IRR rule" row (rank the gated cards by uncapped
annualized ROI), because annualized ROI is a common ranking key for
capital and a reader will ask how it compares. Measurement: it picks the
same batch at 28 of 28 budgets as gated NPV per dollar, by construction:
at one tier the cash cycle is the same for every card, so annualized ROI
is a monotone function of NPV / capital
(`test_irr_rule_ties_npv_per_dollar_at_one_tier`). Alternative: leave the
row out; it lost because the tie is itself the finding, and a missing row
would leave the question open (a principle, not a measurement).

## 2026-11-04 — a stays 0.5 even though a = 1 wins under a cash budget

Decided to keep the shipped capital exponent a = 0.5, because the cash
budget is one use of the ranking and a switch deserves its own commit with
its own review. Measurement: gated NPV per dollar (a = 1) captures 0.874
against GRACE's 0.857 at $3,000 and has more true NPV at 24 of 28 budgets.
Alternative: switch to a = 1 now; deferred, not rejected. This is
reported, not tuned away.

## 2026-11-04 — The backtest universe is liquid cards only

Decided every rule picks from the cards with at least the rankability
liquidity (3 graded-10 sales in the 180-day column), because time to sell
is not modelled and an illiquid card's "true" value would be fiction. The
universe still carries the junk the gates must catch: 451 liquid cards
hold 67 planted errors. Alternative: all 600 cards; it lost because the
truth model has no time-to-sell term to price illiquidity.

## 2026-11-04 — Grade at the regular tier only

Decided Experiment 4 grades every card at the regular tier, because the
engine's own argmax already picks it on this data (README, "The cheap slow
tier keeps winning"), so the forced tier scores the engine's own choice
and keeps the heuristic and random rules on the same footing. Measurement:
169 of 169 gated cards already argmax to the regular tier, and forcing it
changes 0 gated NPV or GRACE values and leaves 169 gated cards
(`tests/test_backtest.py::test_forcing_the_regular_tier_changes_no_gated_prediction`).
Alternative: let each card's argmax pick its tier inside the backtest; it
ties on this data by that measurement, and it would make the truth model
depend on a per-card tier, so the simpler fixed tier stays. A grid that
forces each tier in turn is not built yet (to measure).

## 2026-11-04 — The random rule is a mean over seeded shuffles

Decided the random-liquid-cards row is the mean over 200 seeded shuffles
(seed 4004), drawn once and reused at every budget, because a single
shuffle is noisy and a fixed set makes the sweep's cells and the
standalone runs identical (`tests/test_backtest.py`). Measurement: one
shuffle's true NPV per budget dollar at $3,000 runs 0.02-2.14 over the 200
shuffles (sd 0.43) around their mean of 0.88. Alternative: one shuffle; it
lost by that spread.

## 2026-11-04 — m stays 40 although empirical Bayes fits the next pop better

Decided to keep the shipped pooling pair, m = 40 toward the fixed era
prior, because Experiment 3 splits by target and the target a grader faces
favours the shipped pair. Measurement: with the training-fold m_hat runs
15.8-32.0, empirical Bayes beats m = 40 on the next pop submissions by
-0.0023 ± 0.0011 (5/5 folds better) log loss per trial, but on unscreened
raw cards it is +0.0067 ± 0.0013 (0/5 folds better). Empirical Bayes
changes two things at once, the shrink target (fixed prior to the
training-fold era mean, which sits above every prior) and the shrink
strength (40 to m_hat), so two cross arms change one each. On unscreened
raw cards the target alone, m = 40 toward the era mean, is +0.0060 ±
0.0016 (0/5 folds better); the strength alone, moment m toward the fixed
prior, is +0.0016 ± 0.0009 (0/5 folds better). Both halves lose there, and
the shrink target carries most of the loss; on the next pop the target
alone is -0.0015 and the strength alone -0.0010. An unscreened raw card is
what a grader holds. Alternative: switch to the moment estimate now (both
halves); it lost by the +0.0067 on unscreened cards, and the strength-only
switch lost by the +0.0016. Deferred, not rejected: any switch gets its
own commit.

## 2026-11-04 — Experiment 3 folds over cards and fits on the training folds only

Decided the cross-validation splits by card (one row per card, a seeded
permutation taken mod 5) and fits every era mean and m_hat on the training
folds only, because a fit that saw the test fold's ladders would leak them
into the score. Measurement: changing every test-fold ladder and outcome
leaves the fitted era means and m_hat unchanged, while changing one
training card moves them (`tests/test_gem_cv.py`). Alternative: the
existing single 30% holdout of Experiment 2; it lost because one split
with one seed gives no fold-to-fold spread to read a gap against.

## 2026-11-04 — Log loss clipped at 1e-6, and the clip count printed

Decided to clip every probability to [1e-6, 1 - 1e-6] before the log and to
print how many predictions the clip moved per estimator, because the raw
rate predicts exactly 0 or 1 for ladders with no gems or all gems and an
unclipped log loss would be infinite. Measurement: the clip moves 7
raw-rate predictions on the next pop target and 7 on the unscreened one,
and none for any other estimator. Alternative: drop those cards from the
raw rate's score; it lost because the raw rate would then be scored on
easier cards than every other estimator (a principle, not a measurement).

## 2026-11-04 — A gap under one fold sd is a null

Decided a paired log-loss gap counts as better or worse than m = 40 only
when its mean exceeds the sample sd of the per-fold differences, with the
folds-better count printed beside it, because five folds cannot carry a
formal test. Measurement: the rule is conservative; the raw rate is
+0.0238 ± 0.0241 on the next pop submissions, better than m = 40 in 0 of 5
folds, and is printed as a null because clipped cards make its folds
spread. Alternative: a sign test over the folds; it lost because with five
folds it cannot reach conventional significance even when every fold
agrees.

## 2026-11-04 — The page's budget slider snaps to solved budgets only

Decided the "Hand GRACE a Budget" slider stops only at the Experiment 4
sweep's solved budgets, and every batch the page lists is a solved run
copied from the backtest, because greedy fills are not nested.
Measurement: GRACE's batch is not inside the next budget's batch at 5 of 27
consecutive budget pairs (the oracle's at 8) (`docs/data/summary.json`,
`nesting`). Alternative: a continuous slider that interpolates the lines
and takes the batch from the nearest smaller budget; it lost because it
would list cards that no run bought at that budget
(`test_page_batches_equal_the_backtest_picks`).

## 2026-11-04 — The page lists the batches of GRACE, the heuristic and the oracle

Decided the page's batch table offers GRACE, its fair baseline (the
heuristic) and the ceiling (the oracle), because those three carry the
page's story, and the random rule's tiles are the mean over its seeded
shuffles, so listing one shuffle's batch would contradict them. The three
gated variants stay in the every-rule table. Measurement: the shipped
budget grid is 243 KB; with every rule's batch it would be 345 KB. Both fit
the 800 KB page-data budget, so size did not decide it (a principle, not a
measurement). Alternative: every rule's batch; it lost on the random
rule's mismatch.

## 2026-11-04 — No p5-p95 band on the page yet

Decided the page's lines carry no spread band and say "to measure",
because the backtest runs on one synthetic world: GRACE, the heuristic and
the oracle are deterministic there, and only the random rule has a spread
(over its shuffles), which is not the uncertainty a reader would take a
band to mean. A spread over generator seeds is not computed yet (to
measure). Alternative: shade the random rule's shuffle spread; it
lost because it would read as a confidence band on GRACE (a principle, not
a measurement).

## 2026-11-04 — The page data come from the backtest functions, not a copy

Decided `python -m grace.viz build` calls the Experiment 4 functions
itself rather than copying `reports/backtest_sweep.json`, because every
page value should come from those functions and a copy would be only as
fresh as a second file. `test_page_matches_the_report_sweep`
proves the two agree cell for cell, and `--check` (run by
`tests/test_viz.py`) fails on stale page data. A principle, not a
measurement.

## 2026-11-04 — The report draws the sweep figure and checks it by digest

Decided `python -m grace.report` draws `figures/backtest_sweep.png` from
the sweep cells it writes, and stores a sha256 digest of the plotted
series in the PNG's text chunk, because before this only
`python -m grace.experiments.backtest` drew it and nothing in CI could
tell a stale chart. `--check` (run by `tests/test_report.py`) compares the
stored digest with the fresh sweep's, and a control draws the figure from
a changed sweep and must be reported stale. Alternative: compare the image
bytes; it lost because PNG bytes move with the matplotlib version while
the plotted numbers do not (a principle, not a measurement).
