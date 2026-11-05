"""Experiment 4: the decision backtest. Does the ranking find the money?

Every ranking rule gets the same cash budget B. Each rule walks its own
ranked list and buys a card for grading whenever the card still fits in
what is left of B (a greedy fill). The batch it ends up with is scored on
SYNTHETIC TRUTH: the closed-form expected NPV at each card's planted gem
rate and true prices, graded at the regular tier with the engine's own
fees, sale time and discounting.

Truth model, per card (qty 1, regular tier):

    p10      = the card's planted latent gem rate (an unscreened card)
    p9, p<=8 = (1 - p10) * 0.6, (1 - p10) * 0.4   (the generator's 9-share)
    V10, V9, raw = the TRUE prices (synth.generate_world); V9 falls back to
               0.5*V10 and V<=8 is the true raw, exactly as the engine does
    R        = true raw + regular fee + other_fixed   (the cash actually paid)
    true NPV = Q * U - R    (npv.py's formula with the truth inputs)

Only gem-rate errors (the planted selection effect) and price-feed errors
(the planted junk rows) can separate a rule's predicted value from its
true value. Resale noise and time to sell are not modelled.

Rules:
  grace                 gated (drop ladder + rankability), GRACE score desc
  gated_npv             gated, predicted NPV desc
  gated_npv_per_dollar  gated, predicted NPV / predicted capital desc (a = 1)
  irr_rule              gated, uncapped annualized ROI desc
  heuristic             a common multiplier heuristic: V10 / raw from the
                        feed, desc, no gates
  random                random liquid cards: the mean over a fixed set of
                        seeded shuffles
  oracle                the best batch knowable from the truth: an exact
                        integer-dollar 0/1 knapsack on true NPV

The universe is the LIQUID cards only (>= min_liquidity graded-10 sales in
the input's 180-day column), the same for every rule: with time to sell
not modelled, an illiquid card's "true" value would be fiction.

The budget is charged at the price actually paid (the true raw + fee). A
policy cannot see that price while ranking, but it does see it at the
till; charging every rule the same price is what makes the oracle a
real upper bound for all of them.

Oracle bracket. Capital is not a whole number of dollars, so the exact
knapsack is solved twice on integer weights:
  ceil(capital)   every batch it finds is feasible at the real prices,
                  so its true NPV is a LOWER bound on the best batch
  floor(capital)  every real-feasible batch is feasible here too, so its
                  optimum is an UPPER bound on the best batch; when that
                  optimum batch also fits at the real prices it IS the
                  best batch, the oracle takes it and the bracket is 0
The bracket width (upper - feasible) is printed. Headline ratios divide by
the feasible batch's value; the logic-check test asserts every rule's
true NPV sits at or under the upper bound.

Failure modes: the truth model scores the generator against itself, so
the claims are the rule ORDER and the oracle value captured (a ratio);
regret and every other dollar figure are in the generator's dollars. A
greedy or absolute-NPV "oracle" is not an upper bound (the diagnostics
below measure how far each falls short). Wiring the truth to the engine's
predicted p10 instead of the latent rate would hide the pop-bias caveat
(guard: test_universe_true_npv_is_recomputed_from_the_truth).

Usage: python -m grace.experiments.backtest [--figure-dir figures]
"""

import argparse
import base64
import dataclasses
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from grace import npv as npv_mod
from grace import pipeline, synth
from grace.config import DEFAULT_CONFIG

REGULAR_TIER_NAME = "regular"
BUDGETS = (1000, 3000, 10000)
HEADLINE_BUDGET = 3000
# The page's slider snaps to these. Greedy fills are not nested, so every
# budget is solved on its own (no batch is derived from a smaller one).
SWEEP_BUDGETS = (
    250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
    3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 11000,
    12000, 14000, 16000, 18000, 20000,
)  # fmt: skip
RANDOM_SEED = 4004
N_RANDOM_DRAWS = 200

POLICY_RULES = (
    "grace",
    "gated_npv",
    "gated_npv_per_dollar",
    "irr_rule",
    "heuristic",
    "random",
)
ALL_RULES = (*POLICY_RULES, "oracle")
RULE_LABELS = {
    "grace": "GRACE (NPV / sqrt(capital), gated)",
    "gated_npv": "Gated NPV",
    "gated_npv_per_dollar": "Gated NPV per dollar",
    "irr_rule": "IRR rule",
    "heuristic": "A common multiplier heuristic (V10/raw, no gates)",
    "random": "Random liquid cards",
    "oracle": "Oracle (exact knapsack)",
}
# Short names for prose, the page's legend and its batch buttons; the full
# labels above show in the tables and tooltips.
RULE_SHORT_LABELS = {
    "grace": "GRACE",
    "gated_npv": "Gated NPV",
    "gated_npv_per_dollar": "Gated NPV per $",
    "irr_rule": "IRR rule",
    "heuristic": "Heuristic",
    "random": "Random",
    "oracle": "Oracle",
}
# Columns each ranked rule sorts by (descending); gated rules filter first.
RULE_SCORE_COLUMN = {
    "grace": "grace",
    "gated_npv": "npv",
    "gated_npv_per_dollar": "npv_per_dollar",
    "irr_rule": "irr",
    "heuristic": "feed_multiplier",
}
GATED_RULES = ("grace", "gated_npv", "gated_npv_per_dollar", "irr_rule")


def regular_tier_config(cfg=DEFAULT_CONFIG):
    """The engine config with only the regular tier on offer.

    Why: the backtest grades every card at the regular tier, so the
    engine's predictions (NPV, GRACE, rankability) must be made at that
    tier too; dataclasses.replace keeps every other knob.
    """
    regular_tiers = []
    for tier in cfg.tiers:
        if tier.name == REGULAR_TIER_NAME:
            regular_tiers.append(tier)
    if len(regular_tiers) != 1:
        raise ValueError("config must define exactly one regular tier")
    return dataclasses.replace(cfg, tiers=tuple(regular_tiers))


def true_expected_npv(latent_p10, true_v10, true_v9, true_raw, cfg=DEFAULT_CONFIG):
    """Closed-form true expected NPV of grading one card at the regular tier.

    Formula: Q*U - R with p10 = latent_p10, p9 = (1-p10)*NINE_SHARE,
    p<=8 = (1-p10)*(1-NINE_SHARE), legs from npv.resolve_legs on the TRUE
    prices, R = true raw + regular fee + other_fixed. Returns
    (true_npv, true_capital).
    Failure mode: if the generator's 9-share changes, the truth model
    follows it through synth.NINE_SHARE; a hard-coded copy would drift.
    """
    regular_cfg = regular_tier_config(cfg)
    tier = regular_cfg.tiers[0]
    p9 = (1.0 - latent_p10) * synth.NINE_SHARE
    p_le8 = (1.0 - latent_p10) * (1.0 - synth.NINE_SHARE)
    v9_effective, v_le8, _v9_synthetic = npv_mod.resolve_legs(true_v10, true_v9, true_raw)
    q = npv_mod.expected_net_sale(
        latent_p10, p9, p_le8, true_v10, v9_effective, v_le8, cfg.sell_fee, cfg.ship_pack
    )
    true_capital = npv_mod.capital(1, true_raw, tier.fee, cfg.other_fixed)
    total_months = tier.grading_months + cfg.sale_months
    u = npv_mod.discount_factor(total_months, cfg.discount_rate_annual)
    true_npv = npv_mod.npv(q, 1, u, true_capital)
    return true_npv, true_capital


def build_universe(cards_df, truth_df, true_prices_df, cfg=DEFAULT_CONFIG):
    """One row per LIQUID card: engine predictions next to synthetic truth.

    Columns: card_id, label, should_flag, gated (kept and rankable),
    predicted p10/npv/capital/grace/irr, feed_multiplier, latent_p10,
    true_npv, true_capital. Rows are sorted by card_id so positions are
    stable across runs.
    """
    regular_cfg = regular_tier_config(cfg)
    tier = regular_cfg.tiers[0]
    total_months = tier.grading_months + cfg.sale_months
    scored = pipeline.run(cards_df, regular_cfg)

    feed_columns = cards_df[["card_id", "raw_price", "v10", "grade10_sales_180d"]]
    truth_columns = truth_df[["card_id", "label", "should_flag", "latent_p10"]]
    merged = scored.merge(feed_columns, on="card_id", validate="one_to_one")
    merged = merged.merge(truth_columns, on="card_id", validate="one_to_one")
    merged = merged.merge(true_prices_df, on="card_id", validate="one_to_one")

    liquid = merged[merged["grade10_sales_180d"] >= cfg.min_liquidity]
    liquid = liquid.sort_values("card_id").reset_index(drop=True)

    records = liquid.to_dict("records")
    rows = []
    for card in records:
        true_npv, true_capital = true_expected_npv(
            float(card["latent_p10"]),
            float(card["true_v10"]),
            card["true_v9"],
            float(card["true_raw"]),
            cfg,
        )
        predicted_capital = float(card["capital"])
        predicted_npv = float(card["npv"])
        present_value = predicted_npv + predicted_capital
        irr = npv_mod.annualized_roi(present_value, 1, 1.0, predicted_capital, total_months)
        gated = card["status"] == "kept" and bool(card["rankable"])
        row = {
            "card_id": card["card_id"],
            "label": card["label"],
            "should_flag": bool(card["should_flag"]),
            "gated": gated,
            "predicted_p10": float(card["p10"]),
            "npv": predicted_npv,
            "capital": predicted_capital,
            "npv_per_dollar": predicted_npv / predicted_capital,
            "grace": float(card["grace"]),
            "irr": irr,
            "feed_multiplier": float(card["v10"]) / float(card["raw_price"]),
            "latent_p10": float(card["latent_p10"]),
            "true_npv": true_npv,
            "true_capital": true_capital,
        }
        rows.append(row)
    universe = pd.DataFrame(rows)
    return universe


def rule_order(universe, rule):
    """Ranked row positions for a ranked rule (best first).

    Gated rules keep only gated rows. Ties break by card_id so the order
    is deterministic.
    """
    score_column = RULE_SCORE_COLUMN[rule]
    candidates = universe
    if rule in GATED_RULES:
        candidates = universe[universe["gated"]]
    ranked = candidates.sort_values(
        [score_column, "card_id"], ascending=[False, True], kind="mergesort"
    )
    positions = ranked.index.tolist()
    return positions


def greedy_fill(order, costs, budget):
    """Walk the order; buy each card whose cost still fits the budget.

    A card that does not fit is skipped and the walk goes on, so a cheap
    card further down can still use the leftover cash. Returns positions.
    Invariant (tested): the summed cost never exceeds the budget.
    """
    picks = []
    spent = 0.0
    for position in order:
        cost = costs[position]
        if spent + cost <= budget:
            picks.append(position)
            spent += cost
    return picks


def knapsack_table(values, weights, max_budget):
    """Exact 0/1 knapsack by dynamic programming over integer weights.

    best[c] = the highest total value of a subset with summed weight <= c.
    keep[i, c] is True when item i is taken in the optimum for capacity c
    given items 0..i (a strict improvement, so ties keep the earlier
    solution and the result is deterministic). Items with value <= 0 are
    never taken. Returns (best, keep).
    Failure mode: weights must be non-negative integers; a fractional
    weight would be truncated by the array slicing, so callers round
    explicitly (ceil for feasible, floor for the bound).
    """
    n_items = len(values)
    best = np.zeros(max_budget + 1)
    keep = np.zeros((n_items, max_budget + 1), dtype=bool)
    item_indices = range(n_items)
    for item in item_indices:
        weight = int(weights[item])
        value = float(values[item])
        if value <= 0.0 or weight > max_budget:
            continue
        reachable = max_budget + 1 - weight
        candidate = best[:reachable] + value
        current = best[weight:]
        improves = candidate > current
        keep[item, weight:] = improves
        best[weight:] = np.where(improves, candidate, current)
    return best, keep


def knapsack_picks(keep, weights, budget):
    """Recover the optimal subset for one capacity from the keep table."""
    picks = []
    capacity = int(budget)
    n_items = keep.shape[0]
    reverse_indices = range(n_items - 1, -1, -1)
    for item in reverse_indices:
        if keep[item, capacity]:
            picks.append(item)
            capacity -= int(weights[item])
    picks.reverse()
    return picks


@dataclasses.dataclass
class OracleTables:
    """Both knapsack tables, solved once up to the largest budget needed."""

    ceil_weights: list
    ceil_keep: np.ndarray
    floor_weights: list
    floor_keep: np.ndarray
    floor_best: np.ndarray


def solve_oracle_tables(universe, max_budget):
    """Solve the ceil- and floor-weight knapsacks up to max_budget."""
    values = universe["true_npv"].tolist()
    capitals = universe["true_capital"].tolist()
    ceil_weights = []
    floor_weights = []
    for capital in capitals:
        ceil_weights.append(math.ceil(capital))
        floor_weights.append(math.floor(capital))
    _ceil_best, ceil_keep = knapsack_table(values, ceil_weights, max_budget)
    floor_best, floor_keep = knapsack_table(values, floor_weights, max_budget)
    return OracleTables(ceil_weights, ceil_keep, floor_weights, floor_keep, floor_best)


def batch_value_and_cost(universe, picks):
    """(summed true NPV, summed true capital) of a batch."""
    true_values = universe["true_npv"].tolist()
    true_capitals = universe["true_capital"].tolist()
    value = 0.0
    cost = 0.0
    for position in picks:
        value += true_values[position]
        cost += true_capitals[position]
    return value, cost


def oracle_bracket(universe, tables, budget):
    """(feasible picks, feasible true NPV, upper bound) at one budget.

    The feasible batch is the better of the ceil-weight optimum and the
    floor-weight optimum, the latter only when it fits at the real
    prices. Invariant (tested): feasible value <= upper bound, and the
    feasible batch never costs more than the budget.
    """
    ceil_picks = knapsack_picks(tables.ceil_keep, tables.ceil_weights, budget)
    ceil_value, _ceil_cost = batch_value_and_cost(universe, ceil_picks)
    floor_picks = knapsack_picks(tables.floor_keep, tables.floor_weights, budget)
    floor_value, floor_cost = batch_value_and_cost(universe, floor_picks)
    picks = ceil_picks
    feasible_value = ceil_value
    if floor_cost <= budget and floor_value > ceil_value:
        picks = floor_picks
        feasible_value = floor_value
    upper_bound = float(tables.floor_best[int(budget)])
    return picks, feasible_value, upper_bound


def random_orders(n_rows, seed=RANDOM_SEED, n_draws=N_RANDOM_DRAWS):
    """A fixed list of seeded shuffles of the universe positions.

    Drawn once, independent of the budget, so every budget (and the
    standalone and sweep runs) sees the same random lists.
    """
    rng = np.random.default_rng(seed)
    orders = []
    draw_indices = range(n_draws)
    for _draw in draw_indices:
        permutation = rng.permutation(n_rows)
        orders.append(permutation.tolist())
    return orders


def batch_metrics(universe, picks, budget, oracle_value):
    """KPIs of one batch.

    true_npv            sum of true NPV over the picks
    npv_per_budget_dollar = true_npv / B (B, never dollars spent, so an
                        underspending rule is not rewarded)
    regret              oracle_value - true_npv
    oracle_value_captured = true_npv / oracle_value
    pick_optimism_p10   mean predicted p10 - mean latent p10 of the picks
    pick_optimism_dollars = predicted batch NPV - true batch NPV
    pick_optimism_share = that gap / true batch NPV
    leakage_share       planted-error picks / picks
    leakage_capital_share  their true capital / true capital spent
    Failure mode: an empty batch gives NaN means and shares; they stay NaN
    rather than becoming a flattering 0.
    """
    true_npvs = universe["true_npv"].tolist()
    true_capitals = universe["true_capital"].tolist()
    predicted_npvs = universe["npv"].tolist()
    predicted_p10s = universe["predicted_p10"].tolist()
    latent_p10s = universe["latent_p10"].tolist()
    flags = universe["should_flag"].tolist()

    total_true = 0.0
    total_predicted = 0.0
    spent = 0.0
    predicted_p10_sum = 0.0
    latent_p10_sum = 0.0
    planted_count = 0
    planted_capital = 0.0
    for position in picks:
        total_true += true_npvs[position]
        total_predicted += predicted_npvs[position]
        spent += true_capitals[position]
        predicted_p10_sum += predicted_p10s[position]
        latent_p10_sum += latent_p10s[position]
        if flags[position]:
            planted_count += 1
            planted_capital += true_capitals[position]

    n_picks = len(picks)
    nan = float("nan")
    mean_predicted_p10 = predicted_p10_sum / n_picks if n_picks else nan
    mean_latent_p10 = latent_p10_sum / n_picks if n_picks else nan
    metrics = {
        "budget": float(budget),
        "n_picks": n_picks,
        "spent": spent,
        "true_npv": total_true,
        "npv_per_budget_dollar": total_true / budget,
        "npv_per_dollar_spent": total_true / spent if spent > 0 else nan,
        "regret": oracle_value - total_true,
        "oracle_value_captured": total_true / oracle_value if oracle_value > 0 else nan,
        "predicted_npv": total_predicted,
        "pick_optimism_dollars": total_predicted - total_true,
        "pick_optimism_share": (total_predicted - total_true) / total_true
        if total_true > 0
        else nan,
        "mean_predicted_p10": mean_predicted_p10,
        "mean_true_p10": mean_latent_p10,
        "pick_optimism_p10": mean_predicted_p10 - mean_latent_p10,
        "planted_picks": planted_count,
        "leakage_share": planted_count / n_picks if n_picks else nan,
        "leakage_capital_share": planted_capital / spent if spent > 0 else nan,
    }
    return metrics


def mean_metrics(metric_list):
    """Field-wise mean of several batch_metrics dicts (the random rule)."""
    first = metric_list[0]
    keys = list(first.keys())
    averaged = {}
    for key in keys:
        values = []
        for metrics in metric_list:
            values.append(float(metrics[key]))
        averaged[key] = float(np.mean(values))
    return averaged


def run_budget(universe, budget, tables, orders_by_rule, random_order_list):
    """Every rule at one budget. Returns {rule: {metrics..., picks}}.

    Picks are card_ids. For the random rule the metrics are the mean over
    the seeded shuffles and the picks are those of shuffle 0.
    """
    card_ids = universe["card_id"].tolist()
    costs = universe["true_capital"].tolist()
    oracle_picks, oracle_value, upper_bound = oracle_bracket(universe, tables, budget)

    results = {}
    for rule in POLICY_RULES:
        if rule == "random":
            draw_metrics = []
            first_picks = None
            for order in random_order_list:
                picks = greedy_fill(order, costs, budget)
                if first_picks is None:
                    first_picks = picks
                metrics = batch_metrics(universe, picks, budget, oracle_value)
                draw_metrics.append(metrics)
            entry = mean_metrics(draw_metrics)
            shown_picks = first_picks
        else:
            order = orders_by_rule[rule]
            shown_picks = greedy_fill(order, costs, budget)
            entry = batch_metrics(universe, shown_picks, budget, oracle_value)
        pick_ids = []
        for position in shown_picks:
            pick_ids.append(card_ids[position])
        entry["picks"] = pick_ids
        results[rule] = entry

    oracle_entry = batch_metrics(universe, oracle_picks, budget, oracle_value)
    oracle_pick_ids = []
    for position in oracle_picks:
        oracle_pick_ids.append(card_ids[position])
    oracle_entry["picks"] = oracle_pick_ids
    oracle_entry["upper_bound"] = upper_bound
    oracle_entry["bracket_width"] = upper_bound - oracle_value
    results["oracle"] = oracle_entry
    return results


def ranked_orders(universe):
    """{rule: ranked positions} for every ranked (non-random) policy rule."""
    orders = {}
    for rule in POLICY_RULES:
        if rule == "random":
            continue
        orders[rule] = rule_order(universe, rule)
    return orders


def load_world(cfg=DEFAULT_CONFIG):
    """The default synthetic world with true prices (in memory only)."""
    cards_df, truth_df, true_prices_df = synth.generate_world()
    universe = build_universe(cards_df, truth_df, true_prices_df, cfg)
    return universe


def run_backtest(universe, budgets):
    """All rules at each budget; the knapsack tables are solved once."""
    max_budget = int(max(budgets))
    tables = solve_oracle_tables(universe, max_budget)
    orders_by_rule = ranked_orders(universe)
    n_rows = len(universe)
    random_order_list = random_orders(n_rows)
    by_budget = {}
    for budget in budgets:
        results = run_budget(universe, budget, tables, orders_by_rule, random_order_list)
        by_budget[int(budget)] = results
    return by_budget


def rules_ranked_by(results, metric):
    """Policy rules sorted best first by one metric (ties by rule name)."""

    def sort_key(rule):
        return (-results[rule][metric], rule)

    ranked = sorted(POLICY_RULES, key=sort_key)
    return ranked


def diagnostics(universe, budgets):
    """Measurements behind DECISIONS.md: the alternatives that lost.

    - greedy_true_ratio and greedy_true_npv: "oracles" that fill greedily
      by true NPV per dollar and by absolute true NPV. Each is a feasible
      batch, not an upper bound; min_regret_vs_* < 0 means some policy
      beat it, which disqualifies it as an oracle.
    - predicted_cost_overspend: fill each ranked rule charging the
      PREDICTED capital instead of the price paid; count the
      (rule, budget) cells whose true spend then exceeds B.
    - per_spent_rank_flips: budgets where ranking the rules by NPV per
      dollar SPENT gives a different order than NPV per budget dollar,
      plus the largest per-spent minus per-budget gap and where it is.
    """
    costs = universe["true_capital"].tolist()
    predicted_costs = universe["capital"].tolist()
    true_values = universe["true_npv"].tolist()
    true_ratio = universe["true_npv"] / universe["true_capital"]
    universe_ratio = universe.assign(true_ratio=true_ratio)
    ratio_order = universe_ratio.sort_values(
        ["true_ratio", "card_id"], ascending=[False, True], kind="mergesort"
    ).index.tolist()
    npv_order = universe.sort_values(
        ["true_npv", "card_id"], ascending=[False, True], kind="mergesort"
    ).index.tolist()
    backtest = run_backtest(universe, budgets)
    orders_by_rule = ranked_orders(universe)
    ranked_rules = sorted(orders_by_rule.keys())

    min_regret_ratio = float("inf")
    min_regret_npv = float("inf")
    overspend_cells = 0
    checked_cells = 0
    rank_flips = 0
    largest_gap = 0.0
    largest_gap_rule = ""
    largest_gap_budget = 0
    for budget in budgets:
        cell = backtest[int(budget)]
        order_by_budget = rules_ranked_by(cell, "npv_per_budget_dollar")
        order_by_spent = rules_ranked_by(cell, "npv_per_dollar_spent")
        if order_by_budget != order_by_spent:
            rank_flips += 1
        for rule in POLICY_RULES:
            gap = cell[rule]["npv_per_dollar_spent"] - cell[rule]["npv_per_budget_dollar"]
            if gap > largest_gap:
                largest_gap = gap
                largest_gap_rule = rule
                largest_gap_budget = int(budget)
        ratio_picks = greedy_fill(ratio_order, costs, budget)
        npv_picks = greedy_fill(npv_order, costs, budget)
        ratio_value = 0.0
        for position in ratio_picks:
            ratio_value += true_values[position]
        npv_value = 0.0
        for position in npv_picks:
            npv_value += true_values[position]
        results = backtest[int(budget)]
        for rule in POLICY_RULES:
            rule_value = results[rule]["true_npv"]
            min_regret_ratio = min(min_regret_ratio, ratio_value - rule_value)
            min_regret_npv = min(min_regret_npv, npv_value - rule_value)
        for rule in ranked_rules:
            order = orders_by_rule[rule]
            picks = greedy_fill(order, predicted_costs, budget)
            true_spend = 0.0
            for position in picks:
                true_spend += costs[position]
            checked_cells += 1
            if true_spend > budget:
                overspend_cells += 1
    found = {
        "min_regret_vs_greedy_true_ratio": min_regret_ratio,
        "min_regret_vs_greedy_true_npv": min_regret_npv,
        "predicted_cost_overspend_cells": overspend_cells,
        "predicted_cost_checked_cells": checked_cells,
        "per_spent_rank_flips": rank_flips,
        "per_spent_largest_gap": largest_gap,
        "per_spent_largest_gap_rule": largest_gap_rule,
        "per_spent_largest_gap_budget": largest_gap_budget,
        "n_budgets": len(budgets),
    }
    return found


def tier_check(cards_df, cfg=DEFAULT_CONFIG):
    """Does forcing the regular tier change any gated prediction?

    Scores the cards twice, with every tier on offer (the engine's argmax)
    and with only the regular tier, and compares the gated set (kept and
    rankable) and each gated card's NPV and GRACE score. Returns counts:
    n_gated_default, n_gated_regular, n_gated_regular_argmax (gated cards
    whose argmax tier is already regular), gated_set_equal and
    n_prediction_differences (gated cards whose NPV or GRACE differs).
    Failure mode: if fees or turnarounds change so that a faster tier wins
    for some card, n_gated_regular_argmax drops below n_gated_default and
    the regular-tier backtest no longer scores the engine's own choice.
    """
    default_scored = pipeline.run(cards_df, cfg)
    regular_scored = pipeline.run(cards_df, regular_tier_config(cfg))
    default_gated = default_scored[
        (default_scored["status"] == "kept") & default_scored["rankable"]
    ]
    regular_gated = regular_scored[
        (regular_scored["status"] == "kept") & regular_scored["rankable"]
    ]
    default_ids = default_gated["card_id"].tolist()
    regular_ids = regular_gated["card_id"].tolist()
    argmax_regular = int((default_gated["best_tier"] == REGULAR_TIER_NAME).sum())
    regular_by_id = regular_scored.set_index("card_id")
    differences = 0
    default_records = default_gated.to_dict("records")
    for card in default_records:
        regular_row = regular_by_id.loc[card["card_id"]]
        npv_same = math.isclose(card["npv"], regular_row["npv"], rel_tol=1e-12, abs_tol=1e-9)
        grace_same = math.isclose(
            card["grace"], regular_row["grace"], rel_tol=1e-12, abs_tol=1e-12
        )
        if not (npv_same and grace_same):
            differences += 1
    found = {
        "n_gated_default": len(default_ids),
        "n_gated_regular": len(regular_ids),
        "n_gated_regular_argmax": argmax_regular,
        "gated_set_equal": sorted(default_ids) == sorted(regular_ids),
        "n_prediction_differences": differences,
    }
    return found


def random_shuffle_spread(universe, budget):
    """How far ONE seeded shuffle's NPV per budget dollar strays from the mean.

    Fills every shuffle of random_orders() at the budget and returns the
    mean, sample sd, min and max of their true NPV per budget dollar. Why:
    a single shuffle is the alternative to the mean, and its spread is
    what that alternative would add as noise.
    """
    costs = universe["true_capital"].tolist()
    values = universe["true_npv"].tolist()
    orders = random_orders(len(universe))
    per_dollar = []
    for order in orders:
        picks = greedy_fill(order, costs, budget)
        total = 0.0
        for position in picks:
            total += values[position]
        per_dollar.append(total / budget)
    array = np.asarray(per_dollar, dtype=float)
    spread = {
        "budget": int(budget),
        "n_draws": len(per_dollar),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    return spread


def universe_summary(universe):
    """Counts that set the scene: universe size and the planted base rate."""
    n_rows = len(universe)
    planted = int(universe["should_flag"].sum())
    gated = int(universe["gated"].sum())
    gated_planted = int((universe["gated"] & universe["should_flag"]).sum())
    positive = int((universe["true_npv"] > 0).sum())
    summary = {
        "n_liquid": n_rows,
        "n_planted": planted,
        "planted_base_rate": planted / n_rows,
        "n_gated": gated,
        "n_gated_planted": gated_planted,
        "n_true_npv_positive": positive,
        "total_true_capital": float(universe["true_capital"].sum()),
    }
    return summary


def summary_lines(by_budget, budgets):
    """The printed table: rule order and regret first, dollars after."""
    lines = []
    for budget in budgets:
        results = by_budget[int(budget)]
        oracle = results["oracle"]
        lines.append(
            f"budget ${int(budget):,}: oracle ${oracle['true_npv']:,.0f} "
            f"(upper bound ${oracle['upper_bound']:,.0f}, "
            f"bracket ${oracle['bracket_width']:,.2f})"
        )
        lines.append(
            f"  {'rule':<22} {'captured':>8} {'regret':>8} {'$/B':>6} "
            f"{'picks':>5} {'leak':>5} {'opt p10':>7}"
        )
        for rule in ALL_RULES:
            entry = results[rule]
            lines.append(
                f"  {rule:<22} {entry['oracle_value_captured']:>8.3f} "
                f"{entry['regret']:>8.0f} {entry['npv_per_budget_dollar']:>6.2f} "
                f"{entry['n_picks']:>5.0f} {entry['leakage_share']:>5.2f} "
                f"{entry['pick_optimism_p10']:>+7.3f}"
            )
    return lines


def figure_series(sweep):
    """(budgets, {rule: true NPV per budget dollar at each budget}).

    The exact series make_figure plots. Budgets are sorted as numbers, so
    a sweep keyed by int or by str gives the same order (tested against
    the committed reports/backtest_sweep.json cells).
    """
    budget_keys = list(sweep.keys())
    by_number = {}
    for key in budget_keys:
        by_number[int(key)] = key
    budgets = sorted(by_number.keys())
    series = {}
    for rule in ALL_RULES:
        values = []
        for budget in budgets:
            cell = sweep[by_number[budget]]
            values.append(float(cell[rule]["npv_per_budget_dollar"]))
        series[rule] = values
    return budgets, series


FIGURE_DIGEST_KEY = "Description"
FIGURE_DIGEST_PREFIX = "grace.report series sha256 "
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def figure_digest(sweep):
    """sha256 (base64) of the plotted series, each value at 10 significant digits.

    Rounding first makes the digest the same for the in-memory sweep and
    for the rounded copy in reports/backtest_sweep.json.
    """
    budgets, series = figure_series(sweep)
    parts = []
    for budget in budgets:
        parts.append(str(budget))
    for rule in ALL_RULES:
        values = series[rule]
        texts = []
        for value in values:
            texts.append(f"{value:.10g}")
        parts.append(rule + ":" + ",".join(texts))
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return base64.b64encode(digest).decode("ascii")


def figure_metadata(sweep):
    """The PNG text chunk that names the series a figure was drawn from."""
    return {FIGURE_DIGEST_KEY: FIGURE_DIGEST_PREFIX + figure_digest(sweep)}


def png_text_chunks(path):
    """{keyword: text} from a PNG's tEXt chunks (a hand-rolled chunk walk).

    Failure mode: a file that is not a PNG raises ValueError rather than
    returning an empty dict that would read as "no digest".
    """
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")
    chunks = {}
    offset = len(PNG_SIGNATURE)
    data_length = len(data)
    while offset + 8 <= data_length:
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"tEXt":
            keyword, _separator, text = body.partition(b"\x00")
            chunks[keyword.decode("latin-1")] = text.decode("latin-1")
        if kind == b"IEND":
            break
        offset += 12 + length
    return chunks


def figure_is_current(path, sweep):
    """True when the PNG at path carries the digest of this sweep's series."""
    chunks = png_text_chunks(path)
    expected = figure_metadata(sweep)[FIGURE_DIGEST_KEY]
    return chunks.get(FIGURE_DIGEST_KEY) == expected


def make_figure(sweep, out_path, metadata=None):
    """True NPV per budget dollar across the sweep, one line per rule.

    metadata goes into the PNG's text chunks (python -m grace.report
    stores a digest of the plotted series there, so --check can tell a
    stale figure without comparing image bytes).
    """
    import matplotlib.pyplot as plt

    from grace import plots  # noqa: F401  (forces the Agg backend)

    budgets, series = figure_series(sweep)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    styles = {
        "oracle": ("#1f4e79", "-"),
        "grace": ("#2e8b57", "-"),
        "gated_npv": ("#7a9e3d", "--"),
        "gated_npv_per_dollar": ("#5b7fa6", "--"),
        "irr_rule": ("#9467bd", ":"),
        "heuristic": ("#e08b2d", "-"),
        "random": ("#999999", "-"),
    }
    for rule in ALL_RULES:
        values = series[rule]
        color, linestyle = styles[rule]
        ax.plot(budgets, values, color=color, linestyle=linestyle, label=RULE_LABELS[rule])
    ax.set_xscale("log")
    ax.set_xlabel("cash budget B (dollars, log scale)")
    ax.set_ylabel("true NPV per budget dollar")
    ax.set_title("Decision backtest on synthetic truth (regular tier)")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, metadata=metadata)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args(argv)

    universe = load_world()
    summary = universe_summary(universe)
    print(
        f"liquid universe: {summary['n_liquid']} cards, {summary['n_planted']} planted errors "
        f"(base rate {summary['planted_base_rate']:.3f}), {summary['n_gated']} pass the gates"
    )
    by_budget = run_backtest(universe, BUDGETS)
    lines = summary_lines(by_budget, BUDGETS)
    for line in lines:
        print(line)
    sweep = run_backtest(universe, SWEEP_BUDGETS)
    os.makedirs(args.figure_dir, exist_ok=True)
    figure_path = os.path.join(args.figure_dir, "backtest_sweep.png")
    make_figure(sweep, figure_path, figure_metadata(sweep))
    print(f"\nwrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
