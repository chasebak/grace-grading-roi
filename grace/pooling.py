"""Partial-pooled gem rates.

This is partial pooling toward an era base rate with a FIXED pseudo-count
m, not "empirical Bayes" unqualified: m is not estimated from the data
(the m-estimation experiment checks how the fixed choice compares to a
beta-binomial method-of-moments estimate).

    p10    = (n10 + m*mu_era) / (n_ladder + m)
    share9 = (n9 + m/2) / (n9 + n_le8 + m)
    p9     = (1 - p10) * share9
    p_le8  = (1 - p10) * (1 - share9)

With no pop ladder at all, p10 = mu_era and the remaining mass splits
50/50, flagged "prior".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PooledRates:
    p10: float
    p9: float
    p_le8: float
    source: str  # "pooled" or "prior"


def pooled_rates(n10, n9, n_le8, mu_era, m=40.0):
    """Shrink an observed pop ladder toward the era base rate."""
    n_ladder = n10 + n9 + n_le8
    if n_ladder == 0:
        rest = 1.0 - mu_era
        return PooledRates(mu_era, rest / 2.0, rest / 2.0, "prior")
    p10 = (n10 + m * mu_era) / (n_ladder + m)
    share9 = (n9 + m / 2.0) / (n9 + n_le8 + m)
    rest = 1.0 - p10
    p9 = rest * share9
    p_le8 = rest * (1.0 - share9)
    return PooledRates(p10, p9, p_le8, "pooled")
