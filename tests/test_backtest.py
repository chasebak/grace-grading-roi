"""Experiment 4 logic checks: oracle bound, budget rules, sweep = standalone,
hand case, truth model. Each check is written so it can fail, and the
controls show that it does when the property is broken.

Hand-computed constants were worked out with a calculator, not by calling
the code under test.
"""

import dataclasses
import math

import pandas as pd
import pytest

from grace import npv as npv_mod
from grace import pipeline, synth
from grace.config import DEFAULT_CONFIG
from grace.experiments import backtest

# U at the regular tier: 1 / 1.12^((1.2 + 1.5)/12) = 1 / 1.12^0.225 = 0.9748243
REGULAR_U = 0.9748243
REGULAR_FEE = 79.99
BOUND_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def universe():
    return backtest.load_world()


@pytest.fixture(scope="module")
def main_results(universe):
    return backtest.run_backtest(universe, backtest.BUDGETS)


@pytest.fixture(scope="module")
def sweep_results(universe):
    return backtest.run_backtest(universe, backtest.SWEEP_BUDGETS)


def bound_violations(by_budget, bound_key="upper_bound"):
    """(budget, rule) cells whose true NPV exceeds the oracle's bound."""
    violations = []
    budgets = sorted(by_budget.keys())
    for budget in budgets:
        results = by_budget[budget]
        bound = results["oracle"][bound_key]
        for rule in backtest.ALL_RULES:
            if results[rule]["true_npv"] > bound + BOUND_TOLERANCE:
                violations.append((budget, rule))
    return violations


# --- hand-computed fixtures -------------------------------------------------


def test_hand_knapsack_beats_greedy_by_ratio():
    # weights 5, 4, 3; values 10, 7, 5; capacity 7.
    # Best subset: {4, 3} -> value 12. Greedy by value/weight (2.0, 1.75,
    # 1.67) takes the 5 first, then nothing fits -> value 10.
    values = [10.0, 7.0, 5.0]
    weights = [5, 4, 3]
    best, keep = backtest.knapsack_table(values, weights, 7)
    assert best[7] == pytest.approx(12.0)
    assert backtest.knapsack_picks(keep, weights, 7) == [1, 2]
    greedy = backtest.greedy_fill([0, 1, 2], weights, 7)
    assert greedy == [0]


def test_knapsack_never_takes_a_non_positive_value():
    values = [-5.0, 0.0, 3.0]
    weights = [1, 1, 1]
    best, keep = backtest.knapsack_table(values, weights, 3)
    assert best[3] == pytest.approx(3.0)
    assert backtest.knapsack_picks(keep, weights, 3) == [2]


def tiny_universe(capitals, values):
    rows = []
    indices = range(len(capitals))
    for index in indices:
        rows.append(
            {
                "card_id": f"card_{index:04d}",
                "true_capital": capitals[index],
                "true_npv": values[index],
            }
        )
    return pd.DataFrame(rows)


def test_bracket_hand_case_with_fractional_capital():
    # capitals 3.5 and 3.6 (sum 7.1 > 7): floor weights 3 + 3 = 6 fit, so
    # the upper bound is 5 + 6 = 11; ceil weights 4 + 4 = 8 do not, so the
    # feasible batch is the 6 alone. The floor batch costs 7.1 > 7 and must
    # not be taken. Bracket width 5.
    universe = tiny_universe([3.5, 3.6], [5.0, 6.0])
    tables = backtest.solve_oracle_tables(universe, 7)
    picks, feasible, upper = backtest.oracle_bracket(universe, tables, 7)
    assert picks == [1]
    assert feasible == pytest.approx(6.0)
    assert upper == pytest.approx(11.0)


def test_bracket_closes_when_the_floor_batch_fits():
    # capitals 3.2 and 3.3 (sum 6.5 <= 7): the floor optimum fits at the
    # real prices, so it IS the best batch and the bracket is 0.
    universe = tiny_universe([3.2, 3.3], [5.0, 6.0])
    tables = backtest.solve_oracle_tables(universe, 7)
    picks, feasible, upper = backtest.oracle_bracket(universe, tables, 7)
    assert picks == [0, 1]
    assert feasible == pytest.approx(11.0)
    assert upper == pytest.approx(11.0)


def test_hand_case_zero_gem_rate_is_the_le8_net_sale_minus_capital():
    # p10 = 0 and V9 = raw = 20, so every leg nets 20*0.87 - 5 = 12.40.
    # true NPV = 12.40 * 0.9748243 - (20 + 79.99) = 12.0878 - 99.99 = -87.9022
    true_npv, true_capital = backtest.true_expected_npv(0.0, 500.0, 20.0, 20.0)
    assert true_capital == pytest.approx(20.0 + REGULAR_FEE)
    assert true_npv == pytest.approx(-87.9022, abs=1e-3)


def test_truth_uses_latent_rate_and_nine_share():
    # p10 = 0.3, V10 = 100, V9 = 40, raw = 10, nine-share 0.6:
    # Q = 0.3*(87 - 5) + 0.42*(34.8 - 5) + 0.28*(8.7 - 5)
    #   = 24.6 + 12.516 + 1.036 = 38.152
    # true NPV = 38.152 * 0.9748243 - 89.99 = 37.1915 - 89.99 = -52.7985
    assert synth.NINE_SHARE == 0.6  # the generator's split (n9 ~ Binomial(rest, 0.6))
    true_npv, _true_capital = backtest.true_expected_npv(0.3, 100.0, 40.0, 10.0)
    assert true_npv == pytest.approx(-52.7985, abs=1e-3)


def test_truth_falls_back_to_half_v10_without_a_v9():
    # V9 missing -> 0.5*V10 = 50 (the engine's fallback):
    # Q = 0.3*82 + 0.42*(43.5 - 5) + 0.28*3.7 = 24.6 + 16.17 + 1.036 = 41.806
    true_npv, _true_capital = backtest.true_expected_npv(0.3, 100.0, math.nan, 10.0)
    assert true_npv == pytest.approx(41.806 * REGULAR_U - 89.99, abs=1e-3)


# --- oracle bound ---------------------------------------------------------


def test_every_rule_sits_under_the_knapsack_upper_bound(main_results, sweep_results):
    assert bound_violations(main_results) == []
    assert bound_violations(sweep_results) == []


def test_oracle_feasible_batch_is_feasible_and_under_its_bound(sweep_results):
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        oracle = sweep_results[budget]["oracle"]
        assert oracle["spent"] <= budget
        assert oracle["true_npv"] <= oracle["upper_bound"] + BOUND_TOLERANCE
        assert oracle["bracket_width"] >= -BOUND_TOLERANCE
        assert oracle["oracle_value_captured"] == pytest.approx(1.0)


def test_headline_bracket_is_tight(main_results):
    """The bracket is printed; it must stay a small share of the bound."""
    budgets = sorted(main_results.keys())
    for budget in budgets:
        oracle = main_results[budget]["oracle"]
        assert oracle["bracket_width"] / oracle["upper_bound"] < 0.01


def test_control_a_greedy_oracle_fails_the_bound_check(universe, sweep_results):
    """A greedy fill by absolute true NPV is not an upper bound: swapping it
    in as the bound must produce violations, or the bound test is vacuous."""
    costs = universe["true_capital"].tolist()
    values = universe["true_npv"].tolist()
    npv_order = universe.sort_values(
        ["true_npv", "card_id"], ascending=[False, True], kind="mergesort"
    ).index.tolist()
    swapped = {}
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        picks = backtest.greedy_fill(npv_order, costs, budget)
        greedy_value = 0.0
        for position in picks:
            greedy_value += values[position]
        cell = dict(sweep_results[budget])
        oracle = dict(cell["oracle"])
        oracle["greedy_bound"] = greedy_value
        cell["oracle"] = oracle
        swapped[budget] = cell
    assert bound_violations(swapped, bound_key="greedy_bound") != []


# --- budget rules ---------------------------------------------------------


def test_no_rule_overspends_the_budget(sweep_results):
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        for rule in backtest.ALL_RULES:
            assert sweep_results[budget][rule]["spent"] <= budget + 1e-9, (budget, rule)


def test_every_random_shuffle_stays_within_budget(universe):
    costs = universe["true_capital"].tolist()
    orders = backtest.random_orders(len(universe), n_draws=20)
    for order in orders:
        picks = backtest.greedy_fill(order, costs, 3000)
        spent = 0.0
        for position in picks:
            spent += costs[position]
        assert spent <= 3000


def test_control_charging_predicted_capital_overspends(universe):
    """Filling by the engine's predicted capital overspends at real prices;
    the price-paid rule exists because this check can fail."""
    found = backtest.diagnostics(universe, backtest.BUDGETS)
    assert found["predicted_cost_overspend_cells"] > 0


def test_denominator_is_the_budget(main_results):
    budgets = sorted(main_results.keys())
    for budget in budgets:
        for rule in backtest.ALL_RULES:
            entry = main_results[budget][rule]
            assert entry["npv_per_budget_dollar"] * budget == pytest.approx(entry["true_npv"])


def test_dividing_by_budget_never_rewards_idle_cash():
    """An underspending batch scores the same per budget dollar whatever it
    left idle, and below its per-dollar-spent figure."""
    universe = tiny_universe([100.0, 100.0], [50.0, 50.0])
    universe["npv"] = universe["true_npv"]
    universe["predicted_p10"] = 0.5
    universe["latent_p10"] = 0.5
    universe["should_flag"] = False
    metrics = backtest.batch_metrics(universe, [0], 400.0, 100.0)
    assert metrics["npv_per_budget_dollar"] == pytest.approx(50.0 / 400.0)
    assert metrics["npv_per_dollar_spent"] == pytest.approx(0.5)
    assert metrics["npv_per_budget_dollar"] < metrics["npv_per_dollar_spent"]


def test_random_rule_is_seeded(universe):
    first = backtest.random_orders(len(universe), seed=backtest.RANDOM_SEED, n_draws=5)
    again = backtest.random_orders(len(universe), seed=backtest.RANDOM_SEED, n_draws=5)
    other = backtest.random_orders(len(universe), seed=backtest.RANDOM_SEED + 1, n_draws=5)
    assert first == again
    assert first != other


def test_sweep_cell_equals_standalone_run(universe, sweep_results):
    standalone = backtest.run_backtest(universe, (backtest.HEADLINE_BUDGET,))
    assert standalone[backtest.HEADLINE_BUDGET] == sweep_results[backtest.HEADLINE_BUDGET]


def test_sweep_includes_the_three_budgets_and_25_to_40_cells():
    for budget in backtest.BUDGETS:
        assert budget in backtest.SWEEP_BUDGETS
    assert 25 <= len(backtest.SWEEP_BUDGETS) <= 40
    assert list(backtest.SWEEP_BUDGETS) == sorted(set(backtest.SWEEP_BUDGETS))


# --- universe and rules ---------------------------------------------------


def test_universe_is_liquid_only_and_gated_rules_use_gated_cards(universe):
    cards_df, _truth_df, _true_prices_df = synth.generate_world()
    liquid = cards_df[cards_df["grade10_sales_180d"] >= DEFAULT_CONFIG.min_liquidity]
    assert len(universe) == len(liquid)
    gated_ids = set(universe.loc[universe["gated"], "card_id"])
    for rule in backtest.GATED_RULES:
        order = backtest.rule_order(universe, rule)
        ordered_ids = set(universe.loc[order, "card_id"])
        assert ordered_ids == gated_ids, rule
    heuristic_order = backtest.rule_order(universe, "heuristic")
    assert len(heuristic_order) == len(universe)


def test_committed_csv_world_gives_the_same_gate_decisions(universe):
    """The backtest runs on the in-memory world; the CSVs are rounded to 6
    significant digits. Both must make the same gate decisions."""
    cards_df, _truth_df = synth.load()
    cfg = backtest.regular_tier_config()
    scored = pipeline.run(cards_df, cfg)
    scored = scored.set_index("card_id")
    gated_from_csv = []
    card_ids = universe["card_id"].tolist()
    for card_id in card_ids:
        row = scored.loc[card_id]
        gated_from_csv.append(row["status"] == "kept" and bool(row["rankable"]))
    assert gated_from_csv == universe["gated"].tolist()


def test_regular_tier_config_keeps_every_other_knob():
    cfg = backtest.regular_tier_config()
    assert len(cfg.tiers) == 1
    assert cfg.tiers[0].name == backtest.REGULAR_TIER_NAME
    assert cfg.tiers[0].fee == REGULAR_FEE
    assert cfg.m == DEFAULT_CONFIG.m
    assert cfg.a == DEFAULT_CONFIG.a
    assert cfg.discount_rate_annual == DEFAULT_CONFIG.discount_rate_annual


def test_irr_rule_ties_npv_per_dollar_at_one_tier(sweep_results):
    """Annualized ROI is monotone in NPV/capital when T is the same for every
    card, so the two rules must pick identical batches."""
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        irr_picks = sweep_results[budget]["irr_rule"]["picks"]
        ratio_picks = sweep_results[budget]["gated_npv_per_dollar"]["picks"]
        assert irr_picks == ratio_picks, budget


def test_irr_is_the_engine_annualized_roi(universe):
    cfg = backtest.regular_tier_config()
    total_months = cfg.tiers[0].grading_months + cfg.sale_months
    row = universe.iloc[0]
    present_value = row["npv"] + row["capital"]
    expected = (present_value / row["capital"]) ** (12.0 / total_months) - 1.0
    if present_value <= 0:
        expected = -1.0
    assert row["irr"] == pytest.approx(expected)
    direct = npv_mod.annualized_roi(present_value, 1, 1.0, row["capital"], total_months)
    assert row["irr"] == pytest.approx(direct)


# --- findings that must hold (each can fail) -----------------------------


def test_rule_order_at_the_headline_budget(main_results):
    headline = main_results[backtest.HEADLINE_BUDGET]
    captured = {}
    for rule in backtest.ALL_RULES:
        captured[rule] = headline[rule]["oracle_value_captured"]
    assert captured["oracle"] >= captured["grace"]
    assert captured["grace"] > captured["heuristic"] > captured["random"]
    assert captured["grace"] < 1.0


def test_gates_keep_leakage_below_the_base_rate(universe, main_results):
    base_rate = backtest.universe_summary(universe)["planted_base_rate"]
    budgets = sorted(main_results.keys())
    for budget in budgets:
        assert main_results[budget]["grace"]["leakage_share"] < base_rate
        assert main_results[budget]["heuristic"]["leakage_share"] > base_rate


def test_summary_lines_print_every_rule(main_results):
    lines = backtest.summary_lines(main_results, backtest.BUDGETS)
    text = "\n".join(lines)
    for rule in backtest.ALL_RULES:
        assert rule in text


def test_backtest_figure_written(tmp_path, sweep_results):
    out_path = str(tmp_path / "sweep.png")
    backtest.make_figure(sweep_results, out_path)
    assert tmp_path.joinpath("sweep.png").stat().st_size > 0


# --- truth wiring and the forced tier ------------------------------------


def truth_by_card():
    """{card_id: (latent p10, true V10, true V9, true raw, feed V10, feed V9, feed raw)}."""
    cards_df, truth_df, true_prices_df = synth.generate_world()
    merged = cards_df.merge(truth_df, on="card_id").merge(true_prices_df, on="card_id")
    records = merged.to_dict("records")
    found = {}
    for row in records:
        found[row["card_id"]] = (
            float(row["latent_p10"]),
            float(row["true_v10"]),
            row["true_v9"],
            float(row["true_raw"]),
            float(row["v10"]),
            row["v9"],
            float(row["raw_price"]),
        )
    return found


def test_universe_true_npv_is_recomputed_from_the_truth(universe, main_results):
    """Every universe row's true NPV equals true_expected_npv at the planted
    LATENT rate and the TRUE prices, read straight from the generator, not
    from the universe's own columns. Controls: the rows checked include junk
    plants and cards whose predicted p10 is far from the latent one; feeding
    the predicted p10 or the feed prices instead gives other values; and
    GRACE's picks look rosier than the truth (pick optimism > 0)."""
    truth = truth_by_card()
    records = universe.to_dict("records")
    junk_rows = 0
    rate_gap_rows = 0
    predicted_rate_moves = 0
    feed_price_moves = 0
    for row in records:
        latent, true_v10, true_v9, true_raw, feed_v10, feed_v9, feed_raw = truth[row["card_id"]]
        expected_npv, expected_capital = backtest.true_expected_npv(
            latent, true_v10, true_v9, true_raw
        )
        assert row["true_npv"] == pytest.approx(expected_npv, rel=1e-12, abs=1e-9)
        assert row["true_capital"] == pytest.approx(expected_capital, rel=1e-12, abs=1e-9)
        assert row["latent_p10"] == latent
        if row["should_flag"]:
            junk_rows += 1
            feed_npv, _feed_capital = backtest.true_expected_npv(
                latent, feed_v10, feed_v9, feed_raw
            )
            if abs(feed_npv - row["true_npv"]) > 1.0:
                feed_price_moves += 1
        if abs(row["predicted_p10"] - latent) > 0.01:
            rate_gap_rows += 1
            predicted_npv, _capital = backtest.true_expected_npv(
                row["predicted_p10"], true_v10, true_v9, true_raw
            )
            if abs(predicted_npv - row["true_npv"]) > 1.0:
                predicted_rate_moves += 1
    assert junk_rows > 0
    assert rate_gap_rows > 0
    assert predicted_rate_moves > 0
    assert feed_price_moves > 0
    headline = main_results[backtest.HEADLINE_BUDGET]
    assert headline["grace"]["pick_optimism_p10"] > 0.0


def test_forcing_the_regular_tier_changes_no_gated_prediction():
    """pipeline.run with every tier and with the regular tier only give the
    same gated set and the same NPV and GRACE on it, and every gated card's
    argmax tier is already regular. Control: a cheap express tier makes the
    argmax leave regular, and tier_check must report it."""
    cards_df, _truth_df = synth.load()
    default_scored = pipeline.run(cards_df).set_index("card_id")
    regular_scored = pipeline.run(cards_df, backtest.regular_tier_config()).set_index("card_id")
    default_mask = (default_scored["status"] == "kept") & default_scored["rankable"]
    regular_mask = (regular_scored["status"] == "kept") & regular_scored["rankable"]
    default_gated = default_scored[default_mask]
    regular_gated = regular_scored[regular_mask]
    assert sorted(default_gated.index) == sorted(regular_gated.index)
    assert len(default_gated) > 0
    gated_ids = default_gated.index.tolist()
    for card_id in gated_ids:
        assert default_gated.loc[card_id, "best_tier"] == backtest.REGULAR_TIER_NAME
        assert default_gated.loc[card_id, "npv"] == pytest.approx(regular_gated.loc[card_id, "npv"])
        assert default_gated.loc[card_id, "grace"] == pytest.approx(
            regular_gated.loc[card_id, "grace"]
        )

    found = backtest.tier_check(cards_df)
    assert found["gated_set_equal"] is True
    assert found["n_prediction_differences"] == 0
    assert found["n_gated_regular_argmax"] == found["n_gated_default"] == len(default_gated)

    cheap_tiers = []
    for tier in DEFAULT_CONFIG.tiers:
        if tier.name == "express":
            tier = dataclasses.replace(tier, fee=1.0)
        cheap_tiers.append(tier)
    cheap_cfg = dataclasses.replace(DEFAULT_CONFIG, tiers=tuple(cheap_tiers))
    control = backtest.tier_check(cards_df, cheap_cfg)
    assert control["n_gated_regular_argmax"] < control["n_gated_default"]
    assert control["n_prediction_differences"] > 0


def test_random_shuffle_spread_matches_the_random_rule(universe, main_results):
    """The spread's mean is the random rule's NPV per budget dollar."""
    spread = backtest.random_shuffle_spread(universe, backtest.HEADLINE_BUDGET)
    random_entry = main_results[backtest.HEADLINE_BUDGET]["random"]
    assert spread["n_draws"] == backtest.N_RANDOM_DRAWS
    assert spread["mean"] == pytest.approx(random_entry["npv_per_budget_dollar"])
    assert spread["min"] <= spread["mean"] <= spread["max"]
    assert spread["sd"] > 0.0


def test_figure_series_are_the_sweep_cells(sweep_results):
    """The plotted series are the sweep's NPV per budget dollar, in budget
    order, whether the sweep is keyed by int or (as in the JSON) by str."""
    budgets, series = backtest.figure_series(sweep_results)
    assert budgets == list(backtest.SWEEP_BUDGETS)
    by_str = {}
    for budget in budgets:
        by_str[str(budget)] = sweep_results[budget]
    budgets_str, series_str = backtest.figure_series(by_str)
    assert budgets_str == budgets
    assert series_str == series
    for rule in backtest.ALL_RULES:
        expected = []
        for budget in budgets:
            expected.append(sweep_results[budget][rule]["npv_per_budget_dollar"])
        assert series[rule] == expected


def test_figure_digest_round_trips_and_tracks_the_series(tmp_path, sweep_results):
    out_path = tmp_path / "sweep.png"
    backtest.make_figure(sweep_results, str(out_path), backtest.figure_metadata(sweep_results))
    assert backtest.figure_is_current(out_path, sweep_results)
    changed = {}
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        cell = dict(sweep_results[budget])
        grace_entry = dict(cell["grace"])
        grace_entry["npv_per_budget_dollar"] += 0.01
        cell["grace"] = grace_entry
        changed[budget] = cell
    assert not backtest.figure_is_current(out_path, changed)
    not_png = tmp_path / "not.png"
    not_png.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        backtest.png_text_chunks(not_png)
