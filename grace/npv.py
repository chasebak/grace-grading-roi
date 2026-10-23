"""Per-tier NPV engine.

For one card graded at one service tier:

    Q   = p10*(V10*(1-f) - s) + p9*(V9*(1-f) - s) + p_le8*(Vle8*(1-f) - s)
    R   = qty*(raw_cost + grading_fee) + other_fixed
    T   = grading_months + sale_months
    U   = 1 / (1 + r)^(T/12)
    NPV = Q*qty*U - R

Q is the expected net sale proceeds per card, R the capital committed up
front, U the discount factor over the full cash cycle. Tier choice is
argmax NPV across tiers (a small decision tree under uncertainty).
Annualized ROI is a display metric only, never the ranking.
"""

import math
from dataclasses import dataclass


def net_sale(price, sell_fee, ship_pack):
    """Net proceeds from selling one card at `price`."""
    return price * (1.0 - sell_fee) - ship_pack


def expected_net_sale(p10, p9, p_le8, v10, v9, v_le8, sell_fee, ship_pack):
    """Q: probability-weighted net sale proceeds across the grade legs."""
    leg_10 = p10 * net_sale(v10, sell_fee, ship_pack)
    leg_9 = p9 * net_sale(v9, sell_fee, ship_pack)
    leg_le8 = p_le8 * net_sale(v_le8, sell_fee, ship_pack)
    return leg_10 + leg_9 + leg_le8


def capital(qty, raw_cost, grading_fee, other_fixed=0.0):
    """R: cash committed up front."""
    return qty * (raw_cost + grading_fee) + other_fixed


def discount_factor(total_months, discount_rate_annual):
    """U = 1 / (1 + r)^(T/12)."""
    years = total_months / 12.0
    return 1.0 / (1.0 + discount_rate_annual) ** years


def npv(q, qty, u, r_capital):
    """NPV = Q*qty*U - R."""
    return q * qty * u - r_capital


def annualized_roi(q, qty, u, r_capital, total_months):
    """((Q*qty*U) / R)^(12/T) - 1.

    Display metric only (hover-style), never the ranking. Returns -1.0
    when the discounted proceeds are non-positive (total loss).
    """
    if r_capital <= 0:
        return float("nan")
    present_value = q * qty * u
    if present_value <= 0:
        return -1.0
    return (present_value / r_capital) ** (12.0 / total_months) - 1.0


def raw_net_now(raw_price, sell_fee, ship_pack):
    """Net proceeds from selling the card raw today (floored at zero)."""
    return max(raw_price * (1.0 - sell_fee) - ship_pack, 0.0)


def npv_marginal(npv_value, raw_price, raw_net):
    """Marginal NPV of grading versus selling raw now.

    Selling raw now has NPV (raw_net - raw_price) against the same cost
    basis, so the margin is npv - (raw_net - raw_price).
    """
    return npv_value + (raw_price - raw_net)


def _is_missing(value):
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def resolve_legs(v10, v9, raw_price):
    """Fill in the V9 and V<=8 legs.

    V9 falls back to 0.5*V10 when unobserved (labeled synthetic). V<=8
    defaults to raw: a sub-9 resells at about the raw price while you
    still paid the grading fee.
    """
    v9_synthetic = False
    v9_effective = v9
    v9_missing = _is_missing(v9)
    if v9_missing or v9 <= 0:
        v9_effective = 0.5 * v10
        v9_synthetic = True
    v_le8 = raw_price
    return v9_effective, v_le8, v9_synthetic


@dataclass(frozen=True)
class TierResult:
    """NPV evaluation of one card at one tier."""

    tier_name: str
    q: float
    r_capital: float
    u: float
    total_months: float
    npv: float
    roi_display: float


def evaluate_tiers(p10, p9, p_le8, v10, v9, v_le8, raw_cost, qty, cfg):
    """Evaluate every tier; return (all results, argmax-NPV result)."""
    results = []
    for tier in cfg.tiers:
        q = expected_net_sale(p10, p9, p_le8, v10, v9, v_le8, cfg.sell_fee, cfg.ship_pack)
        r_capital = capital(qty, raw_cost, tier.fee, cfg.other_fixed)
        total_months = tier.grading_months + cfg.sale_months
        u = discount_factor(total_months, cfg.discount_rate_annual)
        value = npv(q, qty, u, r_capital)
        roi = annualized_roi(q, qty, u, r_capital, total_months)
        roi_display = min(roi, cfg.roi_display_cap)
        result = TierResult(tier.name, q, r_capital, u, total_months, value, roi_display)
        results.append(result)
    best = results[0]
    for result in results[1:]:
        if result.npv > best.npv:
            best = result
    return results, best
