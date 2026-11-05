"""One command regenerates every number the README, MODEL.md and
DECISIONS.md quote: python -m grace.report.

Writes, from the committed synthetic world:
  reports/numbers.json         every quoted number, plus the why-box text
                               and the assumptions table
  reports/backtest_sweep.json  Experiment 4 at every sweep budget, with the
                               picked card_ids (the page's slider data)
  reports/findings.md          the rendered blocks, ready to read
  figures/backtest_sweep.png   the README's sweep chart, drawn from the
                               sweep cells; its PNG text chunk carries a
                               digest of the plotted series
JSON is written with sorted keys and floats rounded to 10 significant
digits; non-finite floats become null.

python -m grace.report --check recomputes everything in memory and exits 1
if any committed file differs (CI runs it through tests/test_report.py).
The figure is not compared byte for byte (image bytes move with the
matplotlib version); --check compares the series digest in its PNG text
chunk with the digest of the fresh sweep cells, so a figure drawn from
other numbers is reported stale.

Docs blocks. README.md and MODEL.md carry marked blocks
    <!-- grace.report:begin NAME --> ... <!-- grace.report:end NAME -->
whose text must equal render_blocks()[NAME] exactly, and every number in
prose outside the blocks that comes from a result is listed in
prose_claims(); tests/test_readme_drift.py enforces both. Failure mode:
a number typed into prose outside a block and missing from prose_claims()
is not pinned; the drift test's coverage floor and its mutation control
keep the mechanism honest, but they cannot find a number nobody listed.

Scripts compute, AI only copies: nothing in the docs is typed by hand
that this module can print.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from grace import demo, pipeline, synth, why
from grace.config import DEFAULT_CONFIG
from grace.experiments import backtest, gates_eval, gem_cv, m_estimation

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = "reports"
NUMBERS_PATH = "reports/numbers.json"
SWEEP_PATH = "reports/backtest_sweep.json"
FINDINGS_PATH = "reports/findings.md"
FIGURE_PATH = "figures/backtest_sweep.png"
SIGNIFICANT_DIGITS = 10
HEADLINE_RULES = ("grace", "heuristic", "random")
BLOCK_BEGIN = "<!-- grace.report:begin {name} -->"
BLOCK_END = "<!-- grace.report:end {name} -->"
# Which doc file each block lives in.
BLOCK_FILES = {
    "headline": "README.md",
    "kpi_table": "README.md",
    "exp1_output": "README.md",
    "exp1_why": "README.md",
    "exp2_output": "README.md",
    "exp2_why": "README.md",
    "exp3_tables": "README.md",
    "exp3_findings": "README.md",
    "exp3_why": "README.md",
    "exp4_tables": "README.md",
    "exp4_findings": "README.md",
    "exp4_sweep_why": "README.md",
    "demo_output": "README.md",
    "assumptions": "MODEL.md",
}


def round_floats(value):
    """Floats to 10 significant digits, non-finite to None, keys to str."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        text = f"{value:.{SIGNIFICANT_DIGITS}g}"
        return float(text)
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        rounded = {}
        items = list(value.items())
        for key, inner in items:
            rounded[str(key)] = round_floats(inner)
        return rounded
    if isinstance(value, (list, tuple)):
        rounded_list = []
        for inner in value:
            rounded_list.append(round_floats(inner))
        return rounded_list
    if hasattr(value, "item"):  # a numpy scalar
        return round_floats(value.item())
    return value


def to_json_text(obj):
    """Sorted keys, one-space indent, trailing newline, no NaN."""
    text = json.dumps(obj, indent=1, sort_keys=True, allow_nan=False)
    return text + "\n"


def strip_picks(by_budget):
    """Backtest results without the pick lists (numbers.json keeps them out)."""
    stripped = {}
    budgets = list(by_budget.keys())
    for budget in budgets:
        results = by_budget[budget]
        stripped_rules = {}
        rules = list(results.keys())
        for rule in rules:
            entry = dict(results[rule])
            entry.pop("picks", None)
            stripped_rules[rule] = entry
        stripped[budget] = stripped_rules
    return stripped


def backtest_findings(sweep):
    """Sweep-wide facts the README states (all counted, none typed)."""
    budgets = sorted(sweep.keys())
    irr_same = 0
    npv_per_dollar_beats_grace = 0
    grace_beats_heuristic = 0
    heuristic_beats_random = 0
    for budget in budgets:
        results = sweep[budget]
        if results["irr_rule"]["picks"] == results["gated_npv_per_dollar"]["picks"]:
            irr_same += 1
        if results["gated_npv_per_dollar"]["true_npv"] > results["grace"]["true_npv"]:
            npv_per_dollar_beats_grace += 1
        if results["grace"]["true_npv"] > results["heuristic"]["true_npv"]:
            grace_beats_heuristic += 1
        if results["heuristic"]["true_npv"] > results["random"]["true_npv"]:
            heuristic_beats_random += 1
    facts = {
        "n_budgets": len(budgets),
        "min_budget": budgets[0],
        "max_budget": budgets[-1],
        "irr_rule_same_picks_as_npv_per_dollar": irr_same,
        "npv_per_dollar_beats_grace": npv_per_dollar_beats_grace,
        "grace_beats_heuristic": grace_beats_heuristic,
        "heuristic_beats_random": heuristic_beats_random,
    }
    return facts


def sweep_findings(sweep, base_rate):
    """backtest_findings plus the count of budgets where GRACE's planted-pick
    share sits below the liquid universe's base rate (the README and the
    page both quote these, so one function computes them)."""
    facts = backtest_findings(sweep)
    leak_below = 0
    budgets = sorted(sweep.keys())
    for budget in budgets:
        if sweep[budget]["grace"]["leakage_share"] < base_rate:
            leak_below += 1
    facts["grace_leakage_below_base_rate"] = leak_below
    return facts


def kpi_values(backtest_numbers, universe_summary):
    """Flat, uniquely named KPI keys so the README can cite file:line."""
    headline = backtest_numbers[str(backtest.HEADLINE_BUDGET)]
    kpis = {}
    for rule in backtest.ALL_RULES:
        entry = headline[rule]
        suffix = f"{rule}_{backtest.HEADLINE_BUDGET}"
        kpis[f"kpi_oracle_value_captured_{suffix}"] = entry["oracle_value_captured"]
        kpis[f"kpi_regret_{suffix}"] = entry["regret"]
        kpis[f"kpi_npv_per_budget_dollar_{suffix}"] = entry["npv_per_budget_dollar"]
        kpis[f"kpi_pick_optimism_p10_{suffix}"] = entry["pick_optimism_p10"]
        kpis[f"kpi_pick_optimism_share_{suffix}"] = entry["pick_optimism_share"]
        kpis[f"kpi_planted_picks_{suffix}"] = entry["planted_picks"]
        kpis[f"kpi_n_picks_{suffix}"] = entry["n_picks"]
        kpis[f"kpi_leakage_share_{suffix}"] = entry["leakage_share"]
        kpis[f"kpi_leakage_capital_share_{suffix}"] = entry["leakage_capital_share"]
    kpis["kpi_liquid_universe_cards"] = universe_summary["n_liquid"]
    kpis["kpi_liquid_universe_planted"] = universe_summary["n_planted"]
    kpis["kpi_liquid_universe_planted_base_rate"] = universe_summary["planted_base_rate"]
    return kpis


def gem_cv_kpis(gem_summary):
    """kpi_gem_cv_log_loss_diff_<estimator>_<target>: paired mean vs m = 40."""
    kpis = {}
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        estimators = gem_summary["targets"][target]["estimators"]
        for name in gem_cv.ESTIMATORS:
            if name == gem_cv.BASELINE:
                continue
            paired = estimators[name]["log_loss_diff_vs_m40"]
            kpis[f"kpi_gem_cv_log_loss_diff_{name}_{target}"] = paired["mean"]
    return kpis


def build_outputs_data():
    """Compute everything. Returns (numbers, sweep), both already rounded."""
    cards_df, truth_df = synth.load()

    scored = pipeline.run(cards_df)
    demo_counts = demo.summary_counts(scored)

    gate_metrics = gates_eval.evaluate(cards_df, truth_df)
    gate_metrics.pop("reason_by_label")
    gate_cost = gates_eval.false_positive_cost(cards_df, truth_df)
    gate_flags_by_kind = gates_eval.clean_flags_by_kind(cards_df, truth_df)
    gate_misses = gates_eval.miss_details(cards_df, truth_df)

    m_summary = m_estimation.compute_summary(cards_df, truth_df)
    gem_summary = gem_cv.compute_summary(cards_df, truth_df)

    universe = backtest.load_world()
    universe_summary = backtest.universe_summary(universe)
    main_results = backtest.run_backtest(universe, backtest.BUDGETS)
    sweep_results = backtest.run_backtest(universe, backtest.SWEEP_BUDGETS)
    diagnostics = backtest.diagnostics(universe, backtest.SWEEP_BUDGETS)
    random_spread = backtest.random_shuffle_spread(universe, backtest.HEADLINE_BUDGET)
    tier_check = backtest.tier_check(cards_df)
    sweep_facts = sweep_findings(sweep_results, universe_summary["planted_base_rate"])

    backtest_numbers = round_floats(strip_picks(main_results))
    rounded_summary = round_floats(universe_summary)
    m_hat_values = list(m_summary["m_hat_by_era"].values())
    spearman_values = list(m_summary["prior_shift_spearman"].values())
    assumptions = why.assumption_rows(
        min(m_hat_values), max(m_hat_values), min(spearman_values)
    )

    numbers = {
        "generated_by": "python -m grace.report (committed synthetic data, seed 7)",
        "demo": demo_counts,
        "gates": {
            "metrics": gate_metrics,
            "false_positive_cost": gate_cost,
            "clean_flags_by_kind": gate_flags_by_kind,
            "misses": gate_misses,
        },
        "m_estimation": m_summary,
        "gem_cv": gem_summary,
        "backtest": {
            "settings": {
                "tier": backtest.REGULAR_TIER_NAME,
                "budgets": list(backtest.BUDGETS),
                "headline_budget": backtest.HEADLINE_BUDGET,
                "sweep_budgets": list(backtest.SWEEP_BUDGETS),
                "random_seed": backtest.RANDOM_SEED,
                "random_draws": backtest.N_RANDOM_DRAWS,
                "min_liquidity": DEFAULT_CONFIG.min_liquidity,
                "rule_labels": backtest.RULE_LABELS,
                "rule_short_labels": backtest.RULE_SHORT_LABELS,
            },
            "universe": universe_summary,
            "by_budget": backtest_numbers,
            "diagnostics": diagnostics,
            "random_spread": random_spread,
            "tier_check": tier_check,
            "sweep_facts": sweep_facts,
        },
        "kpis": {
            **kpi_values(backtest_numbers, rounded_summary),
            **gem_cv_kpis(gem_summary),
        },
        "why": why.WHY_THIS_METRIC,
        "assumptions": assumptions,
    }
    sweep = {
        "generated_by": "python -m grace.report (Experiment 4 sweep, regular tier)",
        "budgets": list(backtest.SWEEP_BUDGETS),
        "rule_labels": backtest.RULE_LABELS,
        "picks_note": (
            "picks are card_ids; random shows shuffle 0 of "
            f"{backtest.N_RANDOM_DRAWS}, its metrics are the mean over all of them"
        ),
        "cells": sweep_results,
    }
    return round_floats(numbers), round_floats(sweep)


def json_line(json_text, key):
    """1-based line of the one line holding '"key":' (raises unless unique)."""
    needle = f'"{key}":'
    lines = json_text.splitlines()
    found = []
    numbered = list(enumerate(lines, start=1))
    for line_number, line in numbered:
        if needle in line:
            found.append(line_number)
    if len(found) != 1:
        raise KeyError(f"{key} appears {len(found)} times in the JSON")
    return found[0]


def ratio(value):
    return f"{value:.3f}"


def dollars(value):
    if value < 0:
        return f"-${-value:,.0f}"
    return f"${value:,.0f}"


def signed(value, digits=3):
    return f"{value:+.{digits}f}"


def percent(value, digits=1):
    return f"{value * 100:+.{digits}f}%"


def why_box(key):
    """Three-line "why this metric" blockquote from the shared dict."""
    entry = why.WHY_THIS_METRIC[key]
    lines = [
        f"> **Why this metric.** {entry['question']}",
        f"> *Kind:* {entry['kind']}",
        f"> *Could lie if:* {entry['lie']}",
    ]
    return "\n".join(lines)


def picks_text(entry, rule):
    """'1/20' for a ranked rule; '3.1/27.4 (mean)' for the random rule.

    The label is decided by the rule, not by whether the numbers happen to
    be whole: a shuffle mean that lands on a whole number is still a mean.
    """
    planted = entry["planted_picks"]
    picks = entry["n_picks"]
    if rule == "random":
        return f"{planted:.1f}/{picks:.1f} (mean)"
    return f"{int(planted)}/{int(picks)}"


def render_headline(numbers, numbers_text):
    kpis = numbers["kpis"]
    budget = backtest.HEADLINE_BUDGET
    grace_value = kpis[f"kpi_oracle_value_captured_grace_{budget}"]
    heuristic_value = kpis[f"kpi_oracle_value_captured_heuristic_{budget}"]
    random_value = kpis[f"kpi_oracle_value_captured_random_{budget}"]
    line = json_line(numbers_text, f"kpi_oracle_value_captured_grace_{budget}")
    text = (
        f"**Oracle value captured at a ${budget:,} budget (regular tier, synthetic truth): "
        f"GRACE {ratio(grace_value)}, against {ratio(heuristic_value)} for a common "
        f"multiplier heuristic (V10/raw, no gates) and {ratio(random_value)} for random "
        f"liquid cards.** 1.000 is the best batch an exact-knapsack oracle can buy "
        f"knowing the true rates and prices ([{NUMBERS_PATH}:{line}]({NUMBERS_PATH}))."
    )
    return text


def render_kpi_table(numbers, numbers_text):
    kpis = numbers["kpis"]
    budget = backtest.HEADLINE_BUDGET

    def value(name, rule):
        return kpis[f"kpi_{name}_{rule}_{budget}"]

    def cite(name, rule):
        line = json_line(numbers_text, f"kpi_{name}_{rule}_{budget}")
        return f"{NUMBERS_PATH}:{line}"

    captured = (
        f"GRACE {ratio(value('oracle_value_captured', 'grace'))}, heuristic "
        f"{ratio(value('oracle_value_captured', 'heuristic'))}, random "
        f"{ratio(value('oracle_value_captured', 'random'))} "
        f"({cite('oracle_value_captured', 'grace')})"
    )
    regret = (
        f"regret GRACE {dollars(value('regret', 'grace'))}, heuristic "
        f"{dollars(value('regret', 'heuristic'))}, random {dollars(value('regret', 'random'))}, "
        f"gated NPV per dollar {dollars(value('regret', 'gated_npv_per_dollar'))}; "
        f"$ per budget $ GRACE {value('npv_per_budget_dollar', 'grace'):.2f}, heuristic "
        f"{value('npv_per_budget_dollar', 'heuristic'):.2f}, random "
        f"{value('npv_per_budget_dollar', 'random'):.2f}, gated NPV per dollar "
        f"{value('npv_per_budget_dollar', 'gated_npv_per_dollar'):.2f}, oracle "
        f"{value('npv_per_budget_dollar', 'oracle'):.2f}; {a_one_text(numbers)} "
        f"({cite('regret', 'grace')})"
    )
    optimism = (
        f"p10 GRACE {signed(value('pick_optimism_p10', 'grace'))} vs heuristic "
        f"{signed(value('pick_optimism_p10', 'heuristic'))}; batch $ GRACE "
        f"{percent(value('pick_optimism_share', 'grace'))} vs heuristic "
        f"{percent(value('pick_optimism_share', 'heuristic'))} "
        f"({cite('pick_optimism_p10', 'grace')})"
    )
    headline_rules = numbers["backtest"]["by_budget"][str(budget)]
    gem_cell = gem_cv_kpi_cell(numbers, numbers_text)
    leakage = (
        f"GRACE {picks_text(headline_rules['grace'], 'grace')} picks, "
        f"{value('leakage_capital_share', 'grace') * 100:.1f}% of capital; heuristic "
        f"{picks_text(headline_rules['heuristic'], 'heuristic')}, "
        f"{value('leakage_capital_share', 'heuristic') * 100:.1f}% of capital; liquid "
        f"base rate {kpis['kpi_liquid_universe_planted']}/{kpis['kpi_liquid_universe_cards']} "
        f"({cite('planted_picks', 'grace')})"
    )
    header = (
        "| KPI | formula | why it matters | compared against | what the software does to it "
        "| measured on | value now |"
    )
    lines = [
        header,
        "|---|---|---|---|---|---|---|",
        (
            f"| **Oracle value captured** (headline; ${budget:,} budget, regular tier) "
            "| True-expected NPV of the rule's batch / that of the best feasible batch "
            "| How much of the knowable money the ranking finds "
            "| A common multiplier heuristic (V10/raw, no gates); random liquid cards; "
            "oracle = exact knapsack "
            "| GRACE should sit far above the heuristic and below 1.0 "
            f"| synthetic truth | {captured} |"
        ),
        (
            "| Regret and NPV per budget dollar "
            "| oracle $ - rule $; rule $ / budget B (not / dollars spent) "
            "| Dividing by B means underspending is never rewarded "
            "| The heuristic and random; gated NPV per dollar (a = 1); every rule is in "
            "the Experiment 4 table "
            "| Lower regret than the heuristic and random "
            f"| synthetic truth | {regret} |"
        ),
        (
            "| Pick optimism "
            "| Mean predicted p10 - mean true p10 of the picks; predicted batch $ - true "
            "batch $ (shown as a share of true) "
            "| Prices the pop-bias caveat on the cards you actually buy "
            "| Heuristic picks "
            "| Reported, never tuned away "
            f"| synthetic truth | {optimism} |"
        ),
        (
            "| Corruption leakage "
            "| Planted-error cards in the batch / batch size, and their share of capital "
            "| Gate recall turned into its business result "
            "| Heuristic; the liquid universe's planted base rate "
            "| Below the base rate "
            f"| synthetic truth | {leakage} |"
        ),
        (
            "| Gem-rate log loss gain vs m = 40 "
            "| Paired per-fold difference in binomial log loss, estimator - m = 40 "
            "(negative = better; plus Brier and ECE), 5-fold CV over cards "
            "| A proper scoring rule grades the probabilities themselves "
            "| Fixed prior, era mean, raw rate, empirical Bayes and its two cross arms "
            "| Shows whether pooling beats no pooling "
            f"| {GEM_CV_MEASURED_ON} | {gem_cell} |"
        ),
    ]
    guarantees = [
        "",
        why_box("oracle_value_captured"),
        "",
        "**Guarantees (tests).**",
        "",
        "- The oracle is an exact integer-dollar knapsack bracketed between a feasible batch "
        "and an upper bound, and no rule's true NPV ever exceeds the upper bound "
        "(`tests/test_backtest.py`).",
        "- No rule overspends its budget, and every per-dollar figure divides by the budget "
        "(`tests/test_backtest.py`).",
        "- `generate()` reproduces `synth/*.csv` byte for byte, and capturing true prices adds "
        "no random draws (`tests/test_golden_data.py`).",
        "- Every number in this README's generated blocks equals `reports/numbers.json`, which "
        "equals a fresh run (`tests/test_readme_drift.py`, `tests/test_report.py`).",
    ]
    return "\n".join(lines + guarantees)


def a_one_text(numbers):
    """Whether gated NPV per dollar (a = 1) beats GRACE (a = 0.5) at the headline budget."""
    headline = numbers["backtest"]["by_budget"][str(backtest.HEADLINE_BUDGET)]
    grace_regret = headline["grace"]["regret"]
    a_one_regret = headline["gated_npv_per_dollar"]["regret"]
    if a_one_regret < grace_regret:
        verdict_text = "gated NPV per dollar (a = 1) beats GRACE (a = 0.5) here"
    else:
        verdict_text = "GRACE (a = 0.5) beats gated NPV per dollar (a = 1) here"
    if headline["irr_rule"]["regret"] == a_one_regret:
        verdict_text += " and the IRR rule ties it"
    return f"{verdict_text}; a stays 0.5 per DECISIONS.md"


VERDICT_PHRASE = {
    "better": "better than m = 40",
    "worse": "worse than m = 40",
    "null": "a null against m = 40",
}
GEM_CV_MEASURED_ON = "out-of-sample on the model's own generator"


def short_paired_text(entry, digits=4):
    """'-0.0023 ± 0.0011 (5/5)': the paired gap with the folds-better count."""
    return (
        f"{entry['mean']:+.{digits}f} ± {entry['sd']:.{digits}f} "
        f"({entry['folds_better']}/{gem_cv.K_FOLDS})"
    )


def gem_cv_target_text(target_section, short):
    """One target's part of the KPI cell: empirical Bayes, then who else beats m = 40.

    Rule: empirical Bayes' paired gap and verdict first; then the other
    estimators whose verdict is 'better'; with none, "no estimator beats
    m = 40 there", naming the lowest-mean estimator when it is only a null.
    """
    estimators = target_section["estimators"]
    verdicts = target_section["log_loss_verdict_vs_m40"]
    paired = estimators["eb_mom"]["log_loss_diff_vs_m40"]
    text = (
        f"empirical Bayes {short_paired_text(paired)}, {VERDICT_PHRASE[verdicts['eb_mom']]}"
    )
    others_better = []
    any_better = verdicts["eb_mom"] == "better"
    for name in gem_cv.ESTIMATORS:
        if name in (gem_cv.BASELINE, "eb_mom"):
            continue
        if verdicts[name] == "better":
            others_better.append(short[name])
            any_better = True
    if others_better:
        text += f", as are {' and '.join(others_better)}"
    if not any_better:
        text += ", and no estimator beats m = 40 there"
        best = target_section["best_by_log_loss"]
        if best != gem_cv.BASELINE and verdicts[best] == "null":
            text += f" ({short[best]}, the lowest mean log loss, is a null)"
    return text


def gem_cv_kpi_cell(numbers, numbers_text):
    """The KPI table's Experiment 3 cell: EB vs m = 40 on both targets."""
    section = numbers["gem_cv"]
    short = section["settings"]["estimator_short"]
    parts = []
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        target_section = section["targets"][target]
        parts.append(f"{TARGET_SHORT[target]}: {gem_cv_target_text(target_section, short)}")
    line = json_line(numbers_text, "kpi_gem_cv_log_loss_diff_eb_mom_next_pop")
    joined = "; ".join(parts)
    return f"{joined} ({NUMBERS_PATH}:{line})"


TARGET_SHORT = {"next_pop": "next pop", "unscreened": "unscreened"}


def plus_minus(entry, digits=4):
    return f"{entry['mean']:.{digits}f} ± {entry['sd']:.{digits}f}"


def paired_text(entry, digits=4):
    """'+0.0012 ± 0.0008 (3/5 folds better)'."""
    return (
        f"{entry['mean']:+.{digits}f} ± {entry['sd']:.{digits}f} "
        f"({entry['folds_better']}/{gem_cv.K_FOLDS} folds better)"
    )


def render_exp3_tables(numbers):
    section = numbers["gem_cv"]
    settings = section["settings"]
    labels = settings["estimator_labels"]
    sizes = []
    for size in section["fold_sizes"]:
        sizes.append(str(size))
    size_text = "/".join(sizes)
    lines = [
        (
            f"{section['n_cards']} cards ({section['empty_ladders']} with empty ladders), "
            f"{settings['k_folds']} folds over cards ({size_text} test cards); outcome = one "
            f"fresh {settings['submission_size']}-card submission per card; log loss clipped "
            f"to [{settings['clip']:g}, 1 - {settings['clip']:g}]; ECE over "
            f"{settings['ece_bins']} equal-width bins. Each score is mean ± sd over the folds; "
            "\"vs m = 40\" is the paired per-fold difference (estimator - m = 40, negative = "
            "better) with the folds in which the estimator beat m = 40; a gap under one fold "
            "sd is a null."
        ),
    ]
    targets = list(gem_cv.TARGETS.keys())
    names = [*gem_cv.ESTIMATORS, gem_cv.REFERENCE]
    for target in targets:
        target_section = section["targets"][target]
        estimators = target_section["estimators"]
        verdicts = target_section["log_loss_verdict_vs_m40"]
        lines.append("")
        lines.append(f"Target: {settings['target_labels'][target]}.")
        lines.append("")
        lines.append(
            "| estimator | log loss | vs m = 40 | verdict | Brier | ECE | clipped |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for name in names:
            entry = estimators[name]
            if name == gem_cv.BASELINE:
                versus = "(baseline)"
                word = "-"
            else:
                versus = paired_text(entry["log_loss_diff_vs_m40"])
                word = verdicts.get(name, "reference")
            lines.append(
                f"| {labels[name]} | {plus_minus(entry['log_loss'])} | {versus} | {word} "
                f"| {plus_minus(entry['brier'])} | {plus_minus(entry['ece'])} "
                f"| {entry['clipped']} |"
            )
    return "\n".join(lines)


def render_exp3_findings(numbers, cfg=DEFAULT_CONFIG):
    section = numbers["gem_cv"]
    short = section["settings"]["estimator_short"]
    ranges = section["training_fit_ranges"]
    eras = sorted(ranges.keys())
    m_hat_parts = []
    mu_hat_parts = []
    prior_parts = []
    largest_m_hat = 0.0
    priors_below = 0
    for era in eras:
        entry = ranges[era]
        prior = cfg.era_priors[era]
        m_hat_parts.append(f"{era} {entry['m_hat_min']:.1f}-{entry['m_hat_max']:.1f}")
        mu_hat_parts.append(f"{era} {entry['mu_hat_min']:.3f}-{entry['mu_hat_max']:.3f}")
        prior_parts.append(f"{era} {prior:.2f}")
        largest_m_hat = max(largest_m_hat, entry["m_hat_max"])
        if prior < entry["mu_hat_min"]:
            priors_below += 1
    if largest_m_hat < gem_cv.SHIPPED_M:
        shrink_text = "the fixed m = 40 shrinks harder than every fold's fit"
    else:
        shrink_text = "some fold's fit shrinks at least as hard as the fixed m = 40"
    lines = [
        (
            f"- **Fold fits:** training-fold m_hat runs {', '.join(m_hat_parts)}, so "
            f"{shrink_text}. The fixed era priors ({', '.join(prior_parts)}) sit below "
            f"every fold's era mean ({', '.join(mu_hat_parts)}) in {priors_below} of "
            f"{len(eras)} eras."
        ),
    ]
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        target_section = section["targets"][target]
        estimators = target_section["estimators"]
        verdicts = target_section["log_loss_verdict_vs_m40"]
        best = target_section["best_by_log_loss"]
        verdict_parts = []
        for name in gem_cv.ESTIMATORS:
            if name == gem_cv.BASELINE:
                continue
            paired = estimators[name]["log_loss_diff_vs_m40"]
            verdict_parts.append(
                f"{short[name]} {verdicts[name]} ({paired['folds_better']}/{gem_cv.K_FOLDS})"
            )
        reference = estimators[gem_cv.REFERENCE]["log_loss_diff_vs_m40"]
        lines.append(
            f"- **{TARGET_SHORT[target].capitalize()}:** the best real estimator by log loss "
            f"is {short[best]} ({plus_minus(estimators[best]['log_loss'])}). Against "
            f"m = 40 (folds better): {', '.join(verdict_parts)}. The unknowable true rate "
            f"sits {-reference['mean']:.4f} below m = 40, the room any estimator has left."
        )
    clip_parts = []
    unclipped = []
    for name in gem_cv.ESTIMATORS:
        counts = []
        for target in targets:
            counts.append(section["targets"][target]["estimators"][name]["clipped"])
        if max(counts) == 0:
            unclipped.append(short[name])
            continue
        count_parts = []
        paired_counts = list(zip(targets, counts))
        for target, count in paired_counts:
            count_parts.append(f"{count} on {TARGET_SHORT[target]}")
        clip_parts.append(f"{short[name]} predictions ({', '.join(count_parts)})")
    lines.append(
        f"- **Clipping:** the log-loss clip moved {'; '.join(clip_parts)}: ladders with no "
        f"gems or all gems give a prediction of exactly 0 or 1. It moved none for "
        f"{', '.join(unclipped)}."
    )
    eb_words = []
    split_parts = []
    for target in targets:
        target_section = section["targets"][target]
        word = target_section["log_loss_verdict_vs_m40"]["eb_mom"]
        eb_words.append(f"{word} on {TARGET_SHORT[target]}")
        split_parts.append(f"on {TARGET_SHORT[target]}, {cross_arm_text(target_section)}")
    lines.append(
        f"- **m stays 40:** empirical Bayes against m = 40 is {' and '.join(eb_words)}. "
        "It changes two things at once, the shrink target (fixed prior to era mean) and "
        f"the shrink strength (40 to m_hat); the cross arms split them: {'; '.join(split_parts)}. "
        "m stays 40 (DECISIONS.md); any switch gets its own commit."
    )
    return "\n".join(lines)


def cross_arm_text(target_section):
    """'the target alone +0.0060 ± 0.0016 (0/5), worse; the strength alone ...'."""
    estimators = target_section["estimators"]
    verdicts = target_section["log_loss_verdict_vs_m40"]
    target_alone = estimators["m40_era_mean"]["log_loss_diff_vs_m40"]
    strength_alone = estimators["eb_prior"]["log_loss_diff_vs_m40"]
    return (
        f"the target alone {short_paired_text(target_alone)}, {verdicts['m40_era_mean']}, "
        f"the strength alone {short_paired_text(strength_alone)}, {verdicts['eb_prior']}"
    )


def render_code_block(lines):
    return "\n".join(["```", *lines, "```"])


def render_exp1_output(numbers):
    gates = numbers["gates"]
    lines = gates_eval.summary_lines(gates["metrics"], gates["false_positive_cost"])
    return render_code_block(lines)


def render_exp2_output(numbers):
    summary = dict(numbers["m_estimation"])
    shifts = {}
    raw_shifts = summary["prior_shift_spearman"]
    shift_keys = list(raw_shifts.keys())
    for key in shift_keys:
        shifts[float(key)] = raw_shifts[key]
    summary["prior_shift_spearman"] = shifts
    lines = m_estimation.summary_lines(summary)
    return render_code_block(lines)


def render_demo_output(numbers):
    lines = demo.summary_lines(numbers["demo"])
    return render_code_block(lines)


def render_exp4_tables(numbers):
    by_budget = numbers["backtest"]["by_budget"]
    budgets = numbers["backtest"]["settings"]["budgets"]
    headline_budget = backtest.HEADLINE_BUDGET
    budget_headers = []
    for budget in budgets:
        budget_headers.append(f"${budget:,}")
    header_cells = " | ".join(budget_headers)
    separator_cells = "|".join(["---:"] * len(budgets))
    lines = [
        "Oracle value captured (true NPV of the batch / the oracle's):",
        "",
        f"| rule | {header_cells} |",
        f"|---|{separator_cells}|",
    ]
    for rule in backtest.ALL_RULES:
        cells = []
        for budget in budgets:
            cells.append(ratio(by_budget[str(budget)][rule]["oracle_value_captured"]))
        row_cells = " | ".join(cells)
        lines.append(f"| {backtest.RULE_LABELS[rule]} | {row_cells} |")
    lines.append("")
    lines.append(why_box("oracle_value_captured"))
    lines.append("")
    lines.append(f"Every rule at the ${headline_budget:,} budget:")
    lines.append("")
    lines.append(
        "| rule | regret | true NPV per budget $ | planted picks | planted capital share "
        "| pick optimism (p10) | pick optimism ($) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    headline = by_budget[str(headline_budget)]
    for rule in backtest.ALL_RULES:
        entry = headline[rule]
        lines.append(
            f"| {backtest.RULE_LABELS[rule]} | {dollars(entry['regret'])} "
            f"| {entry['npv_per_budget_dollar']:.2f} | {picks_text(entry, rule)} "
            f"| {entry['leakage_capital_share'] * 100:.1f}% "
            f"| {signed(entry['pick_optimism_p10'])} "
            f"| {dollars(entry['pick_optimism_dollars'])} "
            f"({percent(entry['pick_optimism_share'])}) |"
        )
    lines.append("")
    lines.append(why_box("regret_and_npv_per_budget_dollar"))
    lines.append("")
    lines.append(why_box("pick_optimism"))
    lines.append("")
    lines.append(why_box("corruption_leakage"))
    lines.append("")
    bracket_parts = []
    for budget in budgets:
        oracle = by_budget[str(budget)]["oracle"]
        width = oracle["bracket_width"]
        if width == 0:
            bracket_parts.append(f"${budget:,}: exact at {dollars(oracle['true_npv'])}")
        else:
            share = width / oracle["upper_bound"] * 100
            bracket_parts.append(
                f"${budget:,}: feasible {dollars(oracle['true_npv'])}, upper bound "
                f"{dollars(oracle['upper_bound'])} (width {share:.2f}%)"
            )
    joined = "; ".join(bracket_parts)
    lines.append(f"Oracle bracket (feasible batch vs knapsack upper bound): {joined}.")
    return "\n".join(lines)


def render_exp4_findings(numbers):
    section = numbers["backtest"]
    facts = section["sweep_facts"]
    universe = section["universe"]
    diagnostics = section["diagnostics"]
    headline = section["by_budget"][str(backtest.HEADLINE_BUDGET)]
    n_budgets = facts["n_budgets"]
    span = f"${facts['min_budget']:,}-${facts['max_budget']:,}"
    lines = [
        (
            f"- **Universe:** {universe['n_liquid']} liquid cards, {universe['n_planted']} of "
            f"them planted errors (base rate {universe['planted_base_rate']:.3f}); "
            f"{universe['n_gated']} pass the gates, {universe['n_gated_planted']} of those "
            "planted."
        ),
        (
            f"- **Order holds across the sweep:** GRACE beats the heuristic at "
            f"{facts['grace_beats_heuristic']} of {n_budgets} budgets ({span}), the heuristic "
            f"beats random at {facts['heuristic_beats_random']} of {n_budgets}, and GRACE's "
            f"leakage sits below the base rate at {facts['grace_leakage_below_base_rate']} of "
            f"{n_budgets}."
        ),
        (
            f"- **a = 1 beats the shipped a = 0.5 under a cash budget:** gated NPV per dollar "
            f"captures {ratio(headline['gated_npv_per_dollar']['oracle_value_captured'])} "
            f"against GRACE's {ratio(headline['grace']['oracle_value_captured'])} at "
            f"${backtest.HEADLINE_BUDGET:,}, and has more true NPV at "
            f"{facts['npv_per_dollar_beats_grace']} of {n_budgets} budgets. a stays 0.5 "
            "(DECISIONS.md); any switch gets its own commit."
        ),
        (
            f"- **The IRR rule picks the same batch as gated NPV per dollar** at "
            f"{facts['irr_rule_same_picks_as_npv_per_dollar']} of {n_budgets} budgets: at one "
            "tier the cash cycle T is the same for every card, so annualized ROI is a "
            "monotone function of NPV / capital."
        ),
        (
            "- **A greedy oracle is not an upper bound:** filling by absolute true NPV, some "
            f"rule beats it by up to {dollars(-diagnostics['min_regret_vs_greedy_true_npv'])} "
            "on the sweep, which is why the oracle is an exact knapsack."
        ),
    ]
    return "\n".join(lines)


def render_assumptions(numbers):
    lines = [
        "| knob | value | source | what changes if wrong |",
        "|---|---|---|---|",
    ]
    for row in numbers["assumptions"]:
        lines.append(f"| {row['knob']} | {row['value']} | {row['source']} | {row['if_wrong']} |")
    return "\n".join(lines)


def render_blocks(numbers, numbers_text):
    """{block name: exact text} for every generated docs block."""
    blocks = {
        "headline": render_headline(numbers, numbers_text),
        "kpi_table": render_kpi_table(numbers, numbers_text),
        "exp1_output": render_exp1_output(numbers),
        "exp1_why": why_box("gates_classifier"),
        "exp2_output": render_exp2_output(numbers),
        "exp2_why": why_box("calibration"),
        "exp3_tables": render_exp3_tables(numbers),
        "exp3_findings": render_exp3_findings(numbers),
        "exp3_why": why_box("gem_cv"),
        "exp4_tables": render_exp4_tables(numbers),
        "exp4_findings": render_exp4_findings(numbers),
        "exp4_sweep_why": why_box("backtest_sweep"),
        "demo_output": render_demo_output(numbers),
        "assumptions": render_assumptions(numbers),
    }
    return blocks


def range_text(values, fmt):
    """'a-b' for a sorted list, or 'a' when every value rounds the same."""
    low = fmt.format(values[0])
    high = fmt.format(values[-1])
    if low == high:
        return low
    return f"{low}-{high}"


def prose_claims(numbers):
    """(doc file, exact substring) pairs for result numbers in prose.

    Each substring is rendered here from the generated numbers and must
    appear verbatim in its file (whitespace runs, including line breaks,
    count as one space).
    """
    gates = numbers["gates"]
    metrics = gates["metrics"]
    cost = gates["false_positive_cost"]
    kinds = gates["clean_flags_by_kind"]
    misses = gates["misses"]
    m_summary = numbers["m_estimation"]
    m_hats = sorted(m_summary["m_hat_by_era"].values())
    spearman = sorted(m_summary["prior_shift_spearman"].values())
    recalls = []
    pathologies = list(metrics["per_pathology"].keys())
    for pathology in pathologies:
        recalls.append(metrics["per_pathology"][pathology]["recall"])
    recalls.sort()
    recall_span = f"{recalls[0]:.2f}-{recalls[-1]:.2f} recall"
    wrong = misses["wrong_card"]
    thin = misses["thin_outlier"]
    fake = misses["fake_grail"]
    bad = misses["bad_raw"]
    clean_flagged = metrics["clean_flagged"]
    clean_total = metrics["clean_total"]
    readme_claims = [
        f"only {wrong[0]:.2f}&times; and {wrong[1]:.2f}&times; against the 3&times; gate",
        f"thin outliers at {range_text(thin, '${:.0f}')} against the $200 gate",
        f"a {range_text(fake, '${:.0f}')} fake-grail plant",
        f"corrupted raws divided by only {range_text(bad, '{:.1f}')}&times;",
        f"the ladder flags {clean_flagged} of the {clean_total} clean cards",
        f"{kinds['hot_market']} hot-market cards",
        f"{kinds['sparse_release']} sparse new releases",
        f"{kinds['ratio_tail']} honest ratio-tail cards",
        (
            f"{cost['positive_npv_drops']} of the {cost['clean_dropped']} carried positive "
            f"shadow NPV, ${cost['forgone_npv_total']:,.0f} of it forgone"
        ),
        f"the price paid for {recall_span} on planted corruption",
        f"at {metrics['flag_precision']:.2f} flag precision",
        (
            f"all {metrics['grail_survived']} genuine grails and all "
            f"{metrics['exempt_grail_survived']} borderline"
        ),
        f"estimates ({m_hats[0]:.0f}-{m_hats[-1]:.0f} on this data)",
        f"(Spearman &rho; &ge; {spearman[0]:.3f})",
    ]
    model_claims = [
        f"M_hat lands at {m_hats[0]:.0f}-{m_hats[-1]:.0f}",
        f"as a {m_summary['low_value_gap']:+.3f} predicted-minus-realized gap",
        f"versus {m_summary['high_value_gap']:+.3f} on high-value ones",
        (
            f"keeps {metrics['grail_survived']}/{metrics['grail_total']} planted genuine grails"
        ),
        (
            f"{metrics['exempt_grail_survived']}/{metrics['exempt_grail_total']} "
            "borderline-legit grails"
        ),
    ]
    claims = []
    for text in readme_claims:
        claims.append(("README.md", text))
    for text in model_claims:
        claims.append(("MODEL.md", text))
    decision_claims = decisions_claims(numbers)
    for text in decision_claims:
        claims.append(("DECISIONS.md", text))
    return claims


def decisions_claims(numbers):
    """The measured numbers DECISIONS.md quotes, rendered from the results."""
    section = numbers["backtest"]
    by_budget = section["by_budget"]
    diagnostics = section["diagnostics"]
    facts = section["sweep_facts"]
    headline_budget = str(backtest.HEADLINE_BUDGET)
    headline = by_budget[headline_budget]
    oracle = headline["oracle"]
    grace_entry = headline["grace"]
    heuristic_entry = headline["heuristic"]
    short_labels = section["settings"]["rule_short_labels"]
    widths = []
    budgets = section["settings"]["budgets"]
    for budget in budgets:
        entry = by_budget[str(budget)]["oracle"]
        widths.append(entry["bracket_width"] / entry["upper_bound"] * 100)
    widths.sort()
    budget_text = f"${backtest.HEADLINE_BUDGET:,}"
    if oracle["bracket_width"] == 0:
        headline_bracket = f"exact at {dollars(oracle['true_npv'])} at the {budget_text} budget"
    else:
        headline_bracket = (
            f"feasible {dollars(oracle['true_npv'])} under an upper bound of "
            f"{dollars(oracle['upper_bound'])} at the {budget_text} budget"
        )
    claims = [
        headline_bracket,
        f"widest bracket among the three budgets is {widths[-1]:.2f}%",
        (
            "some rule beats a greedy fill by absolute true NPV by "
            f"{dollars(-diagnostics['min_regret_vs_greedy_true_npv'])}"
        ),
        (
            f"{diagnostics['predicted_cost_overspend_cells']} of "
            f"{diagnostics['predicted_cost_checked_cells']} (rule, budget) cells"
        ),
        (
            f"changed the rule order at {diagnostics['per_spent_rank_flips']} of "
            f"{diagnostics['n_budgets']} budgets"
        ),
        (
            f"largest inflation is {diagnostics['per_spent_largest_gap']:.2f} per dollar "
            f"({short_labels[diagnostics['per_spent_largest_gap_rule']]} at "
            f"${diagnostics['per_spent_largest_gap_budget']:,})"
        ),
        (
            f"{ratio(headline['gated_npv_per_dollar']['oracle_value_captured'])} against "
            f"GRACE's {ratio(grace_entry['oracle_value_captured'])}"
        ),
        f"more true NPV at {facts['npv_per_dollar_beats_grace']} of {facts['n_budgets']} budgets",
        (
            f"GRACE {ratio(grace_entry['oracle_value_captured'])} against "
            f"{ratio(heuristic_entry['oracle_value_captured'])} for a common multiplier heuristic"
        ),
        f"{universe_line(section['universe'])}",
        (
            f"same batch at {facts['irr_rule_same_picks_as_npv_per_dollar']} of "
            f"{facts['n_budgets']} budgets"
        ),
    ]
    settings = section["settings"]
    spread = section["random_spread"]
    tiers = section["tier_check"]
    setting_claims = [
        (
            f"mean over {settings['random_draws']} seeded shuffles "
            f"(seed {settings['random_seed']})"
        ),
        f"({settings['min_liquidity']} graded-10 sales in the 180-day column)",
        (
            f"one shuffle's true NPV per budget dollar at ${spread['budget']:,} runs "
            f"{spread['min']:.2f}-{spread['max']:.2f} over the {spread['n_draws']} shuffles "
            f"(sd {spread['sd']:.2f}) around their mean of {spread['mean']:.2f}"
        ),
        (
            f"{tiers['n_gated_regular_argmax']} of {tiers['n_gated_default']} gated cards "
            "already argmax to the regular tier"
        ),
        (
            f"forcing it changes {tiers['n_prediction_differences']} gated NPV or GRACE "
            f"values and leaves {tiers['n_gated_regular']} gated cards"
        ),
    ]
    for text in setting_claims:
        claims.append(text)
    gem_claims = gem_cv_decision_claims(numbers["gem_cv"])
    for text in gem_claims:
        claims.append(text)
    return claims


def gem_cv_decision_claims(section):
    """The Experiment 3 numbers DECISIONS.md quotes, rendered from the results."""
    targets = section["targets"]
    next_pop = targets["next_pop"]["estimators"]
    unscreened = targets["unscreened"]["estimators"]
    eb_next = next_pop["eb_mom"]["log_loss_diff_vs_m40"]
    eb_unscreened = unscreened["eb_mom"]["log_loss_diff_vs_m40"]
    raw_next = next_pop["raw_rate"]["log_loss_diff_vs_m40"]
    target_alone = unscreened["m40_era_mean"]["log_loss_diff_vs_m40"]
    strength_alone = unscreened["eb_prior"]["log_loss_diff_vs_m40"]
    next_target_alone = next_pop["m40_era_mean"]["log_loss_diff_vs_m40"]
    next_strength_alone = next_pop["eb_prior"]["log_loss_diff_vs_m40"]
    m_hat_values = []
    ranges = section["training_fit_ranges"]
    eras = sorted(ranges.keys())
    for era in eras:
        m_hat_values.append(ranges[era]["m_hat_min"])
        m_hat_values.append(ranges[era]["m_hat_max"])
    m_hat_values.sort()
    claims = [
        f"training-fold m_hat runs {m_hat_values[0]:.1f}-{m_hat_values[-1]:.1f}",
        f"next pop submissions by {paired_text(eb_next)}",
        f"unscreened raw cards it is {paired_text(eb_unscreened)}",
        f"m = 40 toward the era mean, is {paired_text(target_alone)}",
        f"moment m toward the fixed prior, is {paired_text(strength_alone)}",
        (
            f"on the next pop the target alone is {next_target_alone['mean']:+.4f} and the "
            f"strength alone {next_strength_alone['mean']:+.4f}"
        ),
        (
            f"the raw rate is {raw_next['mean']:+.4f} ± {raw_next['sd']:.4f} on the next pop "
            f"submissions, better than m = 40 in {raw_next['folds_better']} of "
            f"{gem_cv.K_FOLDS} folds"
        ),
        (
            f"the clip moves {next_pop['raw_rate']['clipped']} raw-rate predictions on the "
            f"next pop target and {unscreened['raw_rate']['clipped']} on the unscreened one"
        ),
    ]
    return claims


def universe_line(universe):
    """'N liquid cards hold M planted errors' (DECISIONS.md quotes it)."""
    return (
        f"{universe['n_liquid']} liquid cards hold {universe['n_planted']} planted errors"
    )


def render_findings(numbers, numbers_text):
    """reports/findings.md: every block, in reading order."""
    blocks = render_blocks(numbers, numbers_text)
    sections = [
        "# GRACE findings (generated)",
        "",
        "Written by `python -m grace.report` from the committed synthetic data. Do not edit:",
        "`python -m grace.report --check` fails when this file is stale.",
        "",
        "## Headline",
        "",
        blocks["headline"],
        "",
        "## KPIs",
        "",
        blocks["kpi_table"],
        "",
        "## Experiment 4: decision backtest",
        "",
        blocks["exp4_tables"],
        "",
        blocks["exp4_findings"],
        "",
        blocks["exp4_sweep_why"],
        "",
        "## Experiment 1: the gates as a classifier",
        "",
        blocks["exp1_output"],
        "",
        blocks["exp1_why"],
        "",
        "## Experiment 2: is the fixed m = 40 reasonable?",
        "",
        blocks["exp2_output"],
        "",
        blocks["exp2_why"],
        "",
        "## Experiment 3: are the gem-rate probabilities any good?",
        "",
        blocks["exp3_tables"],
        "",
        blocks["exp3_why"],
        "",
        blocks["exp3_findings"],
        "",
        "## Demo",
        "",
        blocks["demo_output"],
        "",
        "## Assumptions",
        "",
        blocks["assumptions"],
        "",
    ]
    return "\n".join(sections)


def doc_block_bounds(text, name):
    """(start, end) character offsets of a block's content, or None.

    The content runs from just after the begin marker's line to just
    before the end marker; a missing or duplicated marker gives None.
    """
    begin = BLOCK_BEGIN.format(name=name)
    end = BLOCK_END.format(name=name)
    if text.count(begin) != 1 or text.count(end) != 1:
        return None
    start = text.index(begin) + len(begin) + 1
    stop = text.index(end)
    if stop < start:
        return None
    return start, stop


def fill_doc_blocks(text, blocks, doc_name):
    """The doc text with every block assigned to doc_name replaced."""
    names = sorted(BLOCK_FILES.keys())
    filled = text
    for name in names:
        if BLOCK_FILES[name] != doc_name:
            continue
        bounds = doc_block_bounds(filled, name)
        if bounds is None:
            raise ValueError(f"{doc_name}: block {name} markers missing or duplicated")
        start, stop = bounds
        filled = filled[:start] + blocks[name] + "\n" + filled[stop:]
    return filled


def doc_names():
    """The doc files that carry generated blocks."""
    names = sorted(set(BLOCK_FILES.values()))
    return names


def build_outputs(root=REPO_ROOT):
    """{repo-relative path: file text} for every generated file.

    The docs (README.md, MODEL.md) are included with their generated
    blocks refilled; their hand-written text is kept as it is in root.
    """
    numbers, sweep = build_outputs_data()
    numbers_text = to_json_text(numbers)
    sweep_text = to_json_text(sweep)
    findings_text = render_findings(numbers, numbers_text)
    outputs = {
        NUMBERS_PATH: numbers_text,
        SWEEP_PATH: sweep_text,
        FINDINGS_PATH: findings_text,
    }
    blocks = render_blocks(numbers, numbers_text)
    docs = doc_names()
    for doc_name in docs:
        doc_path = Path(root) / doc_name
        doc_text = doc_path.read_text(encoding="utf-8")
        outputs[doc_name] = fill_doc_blocks(doc_text, blocks, doc_name)
    return outputs


def sweep_cells(outputs):
    """The sweep cells from the rendered backtest_sweep.json text."""
    sweep = json.loads(outputs[SWEEP_PATH])
    return sweep["cells"]


def write_figure(root, outputs):
    """Draw figures/backtest_sweep.png from the sweep cells, digest included."""
    cells = sweep_cells(outputs)
    figure_path = Path(root) / FIGURE_PATH
    os.makedirs(figure_path.parent, exist_ok=True)
    backtest.make_figure(cells, str(figure_path), backtest.figure_metadata(cells))
    return FIGURE_PATH


def write(root=REPO_ROOT):
    outputs = build_outputs(root)
    os.makedirs(Path(root) / REPORT_DIR, exist_ok=True)
    paths = sorted(outputs.keys())
    for path in paths:
        target = Path(root) / path
        target.write_text(outputs[path], encoding="utf-8")
    paths.append(write_figure(root, outputs))
    return paths


def figure_problems(root, outputs):
    """A missing figure, or one whose series digest is not the fresh sweep's."""
    figure_path = Path(root) / FIGURE_PATH
    if not figure_path.is_file():
        return [f"{FIGURE_PATH} is missing; run python -m grace.report"]
    if not backtest.figure_is_current(figure_path, sweep_cells(outputs)):
        return [f"{FIGURE_PATH} is stale; run python -m grace.report"]
    return []


def check(root=REPO_ROOT, outputs=None):
    """Problems (empty when every committed output equals a fresh run)."""
    if outputs is None:
        outputs = build_outputs(root)
    problems = []
    paths = sorted(outputs.keys())
    for path in paths:
        target = Path(root) / path
        if not target.is_file():
            problems.append(f"{path} is missing; run python -m grace.report")
            continue
        committed = target.read_text(encoding="utf-8")
        if committed != outputs[path]:
            problems.append(f"{path} is stale; run python -m grace.report")
    problems += figure_problems(root, outputs)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.root)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print("reports are up to date")
        return 0
    paths = write(args.root)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
