"""The GRACE score: (NPV - lambda*sigma) / R^a.

With the shipped defaults lambda=0 and a=0.5 the default ranking is
capital-efficiency NPV/sqrt(capital); lambda is an optional mean-risk
penalty (single-asset mean-risk in the sense of Ogryczak & Ruszczynski;
no covariances, so this is not portfolio optimization).

sigma is the probability-weighted standard deviation of the per-grade
NET payoffs, computed on DISCOUNTED payoffs so that sigma and NPV live
on one scale (see README Design decisions).
"""

import math

from grace import npv as npv_mod


def discounted_payoff_sigma(p10, p9, p_le8, v10, v9, v_le8, sell_fee, ship_pack, u):
    """Probability-weighted stdev of discounted per-grade net payoffs."""
    probs = (p10, p9, p_le8)
    values = (v10, v9, v_le8)
    payoffs = []
    for value in values:
        net = npv_mod.net_sale(value, sell_fee, ship_pack)
        payoffs.append(net * u)
    mean = 0.0
    for prob, payoff in zip(probs, payoffs):
        mean += prob * payoff
    variance = 0.0
    for prob, payoff in zip(probs, payoffs):
        deviation = payoff - mean
        variance += prob * deviation * deviation
    return math.sqrt(variance)


def grace_score(npv_value, sigma, r_capital, lam=0.0, a=0.5):
    """(NPV - lambda*sigma) / R^a. NaN when capital is non-positive."""
    if r_capital <= 0:
        return float("nan")
    return (npv_value - lam * sigma) / (r_capital**a)
