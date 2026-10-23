"""Pareto frontier over (raw price, NPV).

A card is on the frontier when no card as cheap or cheaper has a higher
NPV — the non-dominated staircase. Among cards tied on raw price only
the highest NPV survives (and of exact (price, NPV) duplicates, one).
"""


def pareto_frontier(raw_prices, npvs):
    """Return a list of booleans marking frontier membership.

    Both inputs are equal-length sequences aligned by index.
    """
    count = len(raw_prices)
    if len(npvs) != count:
        raise ValueError("raw_prices and npvs must be the same length")
    order = sorted(range(count), key=lambda idx: (raw_prices[idx], -npvs[idx]))
    on_frontier = [False] * count
    best_npv = float("-inf")
    for idx in order:
        if npvs[idx] > best_npv:
            on_frontier[idx] = True
            best_npv = npvs[idx]
    return on_frontier
