"""Data-quality gate ladder: DROP and REPAIR rules, then rankability gates.

The ladder runs before scoring, in this order:

  1. DROP wrong_card_match        V10 > 3x the third-party graded-10 sold
                                  median (median needs n>=3)
  2. DROP ratio_above_ceiling     V10/V9 > 60 (above the genuine ceiling
                                  measured on production (private data);
                                  only with an observed V9)
  3. DROP thin_outlier            no V9, <=1 lifetime feed sale, V10 > $200
  4. DROP unsupported_high_value  V10 >= $1,000 with zero graded-sold
                                  corroboration and <=2 feed sales
  5. REPAIR raw upward            ungraded sold median (n>=3, discarded if
                                  >= V10) is primary truth: raw < median/3
                                  repairs raw to the median. With no signal,
                                  structural floors: raw < 0.02*V9 (or
                                  < 0.005*V10 when no V9) repairs from a
                                  plausible near-mint price if available,
                                  else DROP.
  6. DROP raw_below_one_dollar    post-repair raw < $1
  7. DROP uncorroborated_multiplier
                                  effective multiplier > 100x with NO
                                  graded-sold corroboration (corroborated
                                  grails are exempt)

Rankability gates run after scoring and hide rather than drop (the way a
UI filter would): a card that fails is marked unrankable, not removed.
Gates: NPV >= $25, liquidity >= 3 graded-10 sales (the input column's
180-day window), and era-dependent raw floors.
"""

import math
from dataclasses import dataclass

WRONG_CARD_MEDIAN_MULT = 3.0
RATIO_CEILING = 60.0
THIN_OUTLIER_MIN_V10 = 200.0
THIN_OUTLIER_MAX_SALES = 1
UNSUPPORTED_MIN_V10 = 1000.0
UNSUPPORTED_MAX_FEED_SALES = 2
REPAIR_MEDIAN_DIVISOR = 3.0
FLOOR_FRACTION_OF_V9 = 0.02
FLOOR_FRACTION_OF_V10 = 0.005
MIN_RAW_AFTER_REPAIR = 1.0
UNCORROBORATED_MULTIPLIER_CAP = 100.0
MIN_MEDIAN_N = 3


def _is_missing(value):
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


@dataclass
class GateDecision:
    dropped: bool
    drop_reason: str | None
    raw_effective: float
    raw_confidence: str  # "observed" or "repaired"
    repair_source: str | None  # "ungraded_sold_median" or "near_mint_price"


def apply_gates(card):
    """Run the DROP/REPAIR ladder on one card (a mapping of fields).

    Expected fields: v10, v9, raw_price, grade10_sold_median,
    grade10_sold_n, ungraded_sold_median, ungraded_sold_n,
    feed_sales_lifetime, near_mint_price.
    """
    v10 = float(card["v10"])
    v9 = card.get("v9")
    raw = float(card["raw_price"])
    grade10_median = card.get("grade10_sold_median")
    grade10_n = int(card.get("grade10_sold_n") or 0)
    ungraded_median = card.get("ungraded_sold_median")
    ungraded_n = int(card.get("ungraded_sold_n") or 0)
    feed_sales = int(card.get("feed_sales_lifetime") or 0)
    near_mint = card.get("near_mint_price")

    has_v9 = not _is_missing(v9)
    has_grade10_median = grade10_n >= MIN_MEDIAN_N and not _is_missing(grade10_median)

    # 1. Wrong-card match: the feed's V10 is far above what graded copies
    #    actually sold for.
    if has_grade10_median and v10 > WRONG_CARD_MEDIAN_MULT * float(grade10_median):
        return GateDecision(True, "wrong_card_match", raw, "observed", None)

    # 2. V10/V9 above the measured genuine ceiling.
    if has_v9 and float(v9) > 0 and v10 / float(v9) > RATIO_CEILING:
        return GateDecision(True, "ratio_above_ceiling", raw, "observed", None)

    # 3. Thin outlier: no V9, almost no sales history, elevated V10.
    if (not has_v9) and feed_sales <= THIN_OUTLIER_MAX_SALES and v10 > THIN_OUTLIER_MIN_V10:
        return GateDecision(True, "thin_outlier", raw, "observed", None)

    # 4. Unsupported high value: a four-figure V10 nothing corroborates.
    if (
        v10 >= UNSUPPORTED_MIN_V10
        and grade10_n == 0
        and feed_sales <= UNSUPPORTED_MAX_FEED_SALES
    ):
        return GateDecision(True, "unsupported_high_value", raw, "observed", None)

    # 5. Raw repair. Primary truth: the ungraded sold median.
    raw_effective = raw
    raw_confidence = "observed"
    repair_source = None
    usable_ungraded = None
    has_ungraded_median = ungraded_n >= MIN_MEDIAN_N and not _is_missing(ungraded_median)
    if has_ungraded_median and float(ungraded_median) < v10:  # at/above V10 is noise
        usable_ungraded = float(ungraded_median)
    if usable_ungraded is not None:
        if raw < usable_ungraded / REPAIR_MEDIAN_DIVISOR:
            raw_effective = usable_ungraded
            raw_confidence = "repaired"
            repair_source = "ungraded_sold_median"
    else:
        if has_v9:
            below_floor = raw < FLOOR_FRACTION_OF_V9 * float(v9)
        else:
            below_floor = raw < FLOOR_FRACTION_OF_V10 * v10
        if below_floor:
            if not _is_missing(near_mint) and float(near_mint) > 0:
                raw_effective = float(near_mint)
                raw_confidence = "repaired"
                repair_source = "near_mint_price"
            else:
                return GateDecision(True, "raw_below_floor_no_signal", raw, "observed", None)

    # 6. Post-repair sanity: a raw below $1 is not a real market price.
    if raw_effective < MIN_RAW_AFTER_REPAIR:
        return GateDecision(
            True, "raw_below_one_dollar", raw_effective, raw_confidence, repair_source
        )

    # 7. Uncorroborated multiplier cap; corroborated grails are exempt.
    multiplier = v10 / raw_effective
    if multiplier > UNCORROBORATED_MULTIPLIER_CAP and grade10_n == 0:
        return GateDecision(
            True, "uncorroborated_multiplier", raw_effective, raw_confidence, repair_source
        )

    return GateDecision(False, None, raw_effective, raw_confidence, repair_source)


@dataclass
class Rankability:
    rankable: bool
    reasons: list


def rankability(npv_value, grade10_sales_recent, raw_effective, era, cfg):
    """Post-score gates. A card that fails stays scored but unranked."""
    reasons = []
    if npv_value < cfg.min_npv:
        reasons.append("npv_below_min")
    if grade10_sales_recent < cfg.min_liquidity:
        reasons.append("insufficient_liquidity")
    floor = cfg.era_raw_floors.get(era, 1.0)
    if raw_effective < floor:
        reasons.append("raw_below_era_floor")
    is_rankable = len(reasons) == 0
    return Rankability(is_rankable, reasons)
