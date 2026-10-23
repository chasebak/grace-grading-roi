"""End-to-end scoring pipeline.

For each card: gate ladder -> pooled gem rates -> per-tier NPV (argmax
tier) -> discounted-payoff sigma -> GRACE score -> rankability gates.
Dropped cards still get shadow metrics (computed on the post-ladder
effective raw price) so they can be plotted; they never rank. The Pareto frontier is
computed over the kept-and-rankable set.
"""

import pandas as pd

from grace import frontier as frontier_mod
from grace import gates as gates_mod
from grace import npv as npv_mod
from grace import pooling as pooling_mod
from grace import score as score_mod
from grace.config import DEFAULT_CONFIG


def score_card(card, cfg=DEFAULT_CONFIG):
    """Score one card (a mapping of fields). Returns a flat dict."""
    decision = gates_mod.apply_gates(card)
    raw_effective = decision.raw_effective

    era = card["era"]
    mu_era = cfg.era_priors[era]
    rates = pooling_mod.pooled_rates(
        int(card["n10"]), int(card["n9"]), int(card["n_le8"]), mu_era, cfg.m
    )

    v10 = float(card["v10"])
    v9_effective, v_le8, v9_synthetic = npv_mod.resolve_legs(
        v10, card.get("v9"), raw_effective
    )

    qty = int(card.get("qty") or 1)
    _tier_results, best = npv_mod.evaluate_tiers(
        rates.p10, rates.p9, rates.p_le8, v10, v9_effective, v_le8, raw_effective, qty, cfg
    )

    sigma = score_mod.discounted_payoff_sigma(
        rates.p10,
        rates.p9,
        rates.p_le8,
        v10,
        v9_effective,
        v_le8,
        cfg.sell_fee,
        cfg.ship_pack,
        best.u,
    )
    grace = score_mod.grace_score(best.npv, sigma, best.r_capital, cfg.lam, cfg.a)

    raw_net = npv_mod.raw_net_now(raw_effective, cfg.sell_fee, cfg.ship_pack)
    marginal = npv_mod.npv_marginal(best.npv, raw_effective, raw_net)

    if decision.dropped:
        rank = gates_mod.Rankability(False, ["dropped"])
    else:
        rank = gates_mod.rankability(
            best.npv, int(card.get("grade10_sales_180d") or 0), raw_effective, era, cfg
        )

    row = {
        "card_id": card["card_id"],
        "era": era,
        "status": "dropped" if decision.dropped else "kept",
        "drop_reason": decision.drop_reason or "",
        "raw_effective": raw_effective,
        "raw_confidence": decision.raw_confidence,
        "repair_source": decision.repair_source or "",
        "p10": rates.p10,
        "p9": rates.p9,
        "p_le8": rates.p_le8,
        "rate_source": rates.source,
        "v9_synthetic": v9_synthetic,
        "best_tier": best.tier_name,
        "capital": best.r_capital,
        "npv": best.npv,
        "npv_marginal": marginal,
        "sigma": sigma,
        "grace": grace,
        "roi_display": best.roi_display,
        "rankable": rank.rankable,
        "rank_reasons": ";".join(rank.reasons),
    }
    return row


def run(cards_df, cfg=DEFAULT_CONFIG):
    """Score a card table. Returns a DataFrame with an on_frontier column."""
    records = cards_df.to_dict("records")
    rows = []
    for card in records:
        row = score_card(card, cfg)
        rows.append(row)
    result = pd.DataFrame(rows)

    rankable_mask = (result["status"] == "kept") & result["rankable"]
    on_frontier = [False] * len(result)
    rankable_positions = []
    for position, flag in enumerate(rankable_mask.tolist()):
        if flag:
            rankable_positions.append(position)
    raw_values = []
    npv_values = []
    for position in rankable_positions:
        raw_values.append(float(result["raw_effective"].iloc[position]))
        npv_values.append(float(result["npv"].iloc[position]))
    frontier_flags = frontier_mod.pareto_frontier(raw_values, npv_values)
    for flag_index, position in enumerate(rankable_positions):
        on_frontier[position] = frontier_flags[flag_index]
    result["on_frontier"] = on_frontier
    return result
