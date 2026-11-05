"""The "why this metric" text and the assumptions table, kept in one place.

Why one module: the README, MODEL.md, reports/findings.md and the page's
JSON all show this text, and a second hand-typed copy would drift. The
report command (python -m grace.report) renders it into every place; the
drift test fails if a rendered copy differs.

Each WHY entry answers three things in one line each:
  question  the owner's question the number answers
  kind      what kind of number it is (counted / modelled / forecast /
            simulated) and where its truth comes from
  lie       what would make it lie, and which test guards against that

Each ASSUMPTIONS row: the knob, its value (read from the code, never
typed), its source (measured / published default / illustrative / design
choice) and what changes if it is wrong.
"""

from grace import synth
from grace.config import DEFAULT_CONFIG
from grace.experiments import backtest, gem_cv

WHY_THIS_METRIC = {
    "oracle_value_captured": {
        "question": (
            "Of the money a perfectly informed buyer could make with this budget, "
            "how much does the ranking find?"
        ),
        "kind": (
            "Simulated: closed-form expected NPV at the generator's planted gem rates "
            "and true prices (synthetic truth), divided by the exact-knapsack oracle's batch."
        ),
        "lie": (
            "It would lie if the oracle were not the best batch (guard: the oracle-bound "
            "tests in tests/test_backtest.py); it scores the generator against itself, so "
            "the rule order is the claim, not the dollars."
        ),
    },
    "regret_and_npv_per_budget_dollar": {
        "question": "How many dollars does each rule leave on the table at this budget?",
        "kind": (
            "Simulated: oracle true NPV minus the rule's true NPV, and the rule's true NPV "
            "divided by the budget B (synthetic truth)."
        ),
        "lie": (
            "Dividing by dollars spent would reward a rule that leaves cash idle, and an "
            "overspent budget would inflate it (guards: test_denominator_is_the_budget, "
            "test_no_rule_overspends_the_budget)."
        ),
    },
    "pick_optimism": {
        "question": (
            "On the cards a rule actually buys, how much rosier is the engine than the truth?"
        ),
        "kind": (
            "Simulated: mean predicted p10 minus mean planted p10 of the picks, and predicted "
            "batch NPV minus true batch NPV (synthetic truth)."
        ),
        "lie": (
            "It would hide the pop-bias caveat if the truth reused the selection-boosted "
            "or predicted rates; every universe row's true NPV is recomputed from the "
            "planted latent rate and true prices, and a control shows the predicted rate "
            "gives other values (guard: test_universe_true_npv_is_recomputed_from_the_"
            "truth). It is reported, never tuned away."
        ),
    },
    "corruption_leakage": {
        "question": "How much of the budget goes to cards whose feed prices are junk?",
        "kind": (
            "Counted: planted-error cards in the batch over batch size, and their share of "
            "the capital spent, against the liquid universe's planted base rate."
        ),
        "lie": (
            "It would flatter the gates if junk cards were scored at their junk prices; "
            "they are scored at the base card's true prices (guard: "
            "test_junk_plants_true_prices_are_the_base_card)."
        ),
    },
    "backtest_sweep": {
        "question": "Does the rule order hold at every budget, or only at the one quoted?",
        "kind": (
            "Simulated: true NPV per budget dollar at each solved budget of the sweep, "
            "each budget filled on its own (greedy fills are not nested)."
        ),
        "lie": (
            "A sweep cell that differed from the standalone run would mean two different "
            "backtests (guard: test_sweep_cell_equals_standalone_run)."
        ),
    },
    "batch_list": {
        "question": (
            "Which cards does the rule actually buy at this budget, and which of them carry "
            "junk feed prices?"
        ),
        "kind": (
            "Counted and simulated: the rule's own fill at this budget in buy order; predicted "
            "NPV is the engine's, true NPV is closed-form at the planted gem rate and true "
            "prices (synthetic truth); red rows are the generator's planted errors."
        ),
        "lie": (
            "A batch interpolated between solved budgets would list cards no run bought, "
            "because greedy fills are not nested; every list is a solved run (guard: "
            "test_page_batches_equal_the_backtest_picks in tests/test_viz.py)."
        ),
    },
    "gem_cv": {
        "question": (
            "Are the gem-rate probabilities themselves any good, and does pooling beat "
            "no pooling?"
        ),
        "kind": (
            "Simulated: 5-fold cross-validation over cards, scored by log loss, Brier and "
            "ECE against a fresh simulated submission per card (out-of-sample on the "
            "model's own generator), paired fold by fold against the shipped m = 40."
        ),
        "lie": (
            "Era means or m_hat fit on all cards would leak the test fold (guard: "
            "test_changing_the_test_fold_leaves_the_fit_unchanged); a gap under one fold "
            "sd is printed as a null, and the log-loss clip count is shown so a 0-or-1 "
            "prediction cannot hide."
        ),
    },
    "gates_classifier": {
        "question": "Do the gates catch junk feed rows without killing real grails?",
        "kind": (
            "Counted: gate decisions against the generator's planted labels (synthetic "
            "truth), as recall per pathology and flag precision."
        ),
        "lie": (
            "Plants that never approach a threshold would make recall and precision 1.00 by "
            "construction; plants straddle the gates on both sides (guard: "
            "tests/test_experiments.py, tests/test_gates.py)."
        ),
    },
    "calibration": {
        "question": "Are the pooled gem rates honest for a card nobody pre-screened?",
        "kind": (
            "Simulated: pooled p10 against a fresh simulated submission at each held-out "
            "card's latent rate (out-of-sample on the model's own generator)."
        ),
        "lie": (
            "One 30% holdout with one seed can be lucky; the planted selection effect must "
            "show as predictions above realized (guard: "
            "test_calibration_shows_planted_optimism)."
        ),
    },
}


def money(value):
    return f"${value:,.2f}"


def assumption_rows(m_hat_low, m_hat_high, spearman_low):
    """The MODEL.md assumptions table, with every value read from code.

    The three measured values come from the m-estimation experiment via
    the report, so no measured number is typed here.
    """
    cfg = DEFAULT_CONFIG
    regular_cfg = backtest.regular_tier_config(cfg)
    regular = regular_cfg.tiers[0]
    era_priors = cfg.era_priors
    prior_text = (
        f"modern {era_priors['modern']:.2f} / mid {era_priors['mid']:.2f} / "
        f"vintage {era_priors['vintage']:.2f}"
    )
    rows = [
        {
            "knob": "Regular tier fee and turnaround",
            "value": f"{money(regular.fee)}, {regular.grading_months:g} months",
            "source": "published default (PSA public fee schedule; rough public turnaround)",
            "if_wrong": (
                "Every capital figure moves with the fee; the effect on rule order is to measure."
            ),
        },
        {
            "knob": "Time to sell after grading",
            "value": f"{cfg.sale_months:g} months",
            "source": "illustrative",
            "if_wrong": "Each extra month discounts proceeds by about the monthly rate.",
        },
        {
            "knob": "Annual discount rate",
            "value": f"{cfg.discount_rate_annual:g}",
            "source": "illustrative",
            "if_wrong": "Higher rates shrink every NPV and favor faster tiers.",
        },
        {
            "knob": "Selling fee and ship/pack",
            "value": f"{cfg.sell_fee:g} of the sale, {money(cfg.ship_pack)} per sale",
            "source": "illustrative",
            "if_wrong": "Scales every net sale; cards near the NPV floor flip first.",
        },
        {
            "knob": "Pooling pseudo-count m",
            "value": f"{cfg.m:g}",
            "source": (
                f"design choice (method of moments measures {m_hat_low:.1f}-{m_hat_high:.1f} "
                "on this data)"
            ),
            "if_wrong": "Shrinks gem rates harder or softer toward the era prior.",
        },
        {
            "knob": "Era priors",
            "value": prior_text,
            "source": "illustrative (sensitivity-analyzed, Experiment 2)",
            "if_wrong": (
                f"A +/-0.10 shift barely reorders the ranking (Spearman >= {spearman_low:.3f})."
            ),
        },
        {
            "knob": "GRACE capital exponent a",
            "value": f"{cfg.a:g}",
            "source": "design choice",
            "if_wrong": "a = 1 is NPV per dollar; Experiment 4 prices the difference.",
        },
        {
            "knob": "Rankability gates",
            "value": f"NPV >= {money(cfg.min_npv)}, >= {cfg.min_liquidity} graded-10 sales",
            "source": "design choice",
            "if_wrong": "Looser gates admit more cards and more planted errors.",
        },
        {
            "knob": "Selection boost (generator)",
            "value": (
                f"{synth.SELECTION_BOOST_MAX:g} * exp(-V10 / {synth.SELECTION_BOOST_SCALE:g})"
            ),
            "source": "illustrative",
            "if_wrong": "Sets how optimistic pop-derived gem rates are; 0 is the null control.",
        },
        {
            "knob": "Nine-share of non-gem copies (generator and truth)",
            "value": f"{synth.NINE_SHARE:g}",
            "source": "design choice",
            "if_wrong": "The V9 leg's weight in true NPV; the truth reuses the generator's value.",
        },
        {
            "knob": "True price of a junk-feed plant",
            "value": "the base card it replaced",
            "source": "design choice",
            "if_wrong": (
                "Junk picks would score differently, and leakage counts could shift too: "
                "fills are charged at true capital, so a junk card's true price changes "
                "which cards fit."
            ),
        },
        {
            "knob": "Backtest universe",
            "value": f"liquid cards only (>= {cfg.min_liquidity} graded-10 sales)",
            "source": "design choice",
            "if_wrong": "Illiquid cards would need a time-to-sell model the repo lacks.",
        },
        {
            "knob": "Resale noise and time to sell",
            "value": "not modelled",
            "source": "design choice",
            "if_wrong": "Real batches spread around the expected values shown.",
        },
        {
            "knob": "Budget fill",
            "value": "greedy down each rule's ranking, charged at the price paid",
            "source": "design choice (greedy is not optimal; the oracle measures the gap)",
            "if_wrong": "Charging predicted prices overspends real budgets (DECISIONS.md).",
        },
        {
            "knob": "Experiment 3 outcome and folds",
            "value": (
                f"one {gem_cv.SUBMISSION_SIZE}-card submission per card, "
                f"{gem_cv.K_FOLDS} folds over cards (seeds {gem_cv.FOLD_SEED}, "
                f"{gem_cv.OUTCOME_SEED})"
            ),
            "source": "design choice",
            "if_wrong": (
                "A larger submission lowers outcome noise; fewer folds widen the fold sd."
            ),
        },
        {
            "knob": "Log-loss clip",
            "value": f"[{gem_cv.CLIP:g}, 1 - {gem_cv.CLIP:g}]",
            "source": "design choice",
            "if_wrong": (
                "Sets what a 0-or-1 prediction costs; only the clipped raw-rate cards move."
            ),
        },
        {
            "knob": "Random rule",
            "value": (
                f"mean of {backtest.N_RANDOM_DRAWS} seeded shuffles "
                f"(seed {backtest.RANDOM_SEED})"
            ),
            "source": "design choice",
            "if_wrong": "A single shuffle would make the random row noisy.",
        },
    ]
    return rows
