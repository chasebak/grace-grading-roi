"""Every gate hit on a crafted case AND missed on a near-boundary clean case."""

import math

from grace import gates
from grace.config import DEFAULT_CONFIG


def make_card(**overrides):
    """A clean baseline card no gate should touch."""
    card = {
        "v10": 300.0,
        "v9": 120.0,
        "raw_price": 30.0,
        "grade10_sold_median": 290.0,
        "grade10_sold_n": 5,
        "ungraded_sold_median": 31.0,
        "ungraded_sold_n": 6,
        "feed_sales_lifetime": 10,
        "near_mint_price": 32.0,
    }
    card.update(overrides)
    return card


def test_clean_card_passes():
    decision = gates.apply_gates(make_card())
    assert decision.dropped is False
    assert decision.raw_confidence == "observed"
    assert decision.raw_effective == 30.0


# 1. wrong-card match -------------------------------------------------------


def test_wrong_card_match_hit():
    card = make_card(v10=900.1, grade10_sold_median=300.0, grade10_sold_n=3)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "wrong_card_match"


def test_wrong_card_match_missed_at_boundary():
    # exactly 3.0x is NOT above the threshold
    card = make_card(v10=900.0, grade10_sold_median=300.0, grade10_sold_n=3)
    assert gates.apply_gates(card).dropped is False


def test_wrong_card_match_needs_median_n3():
    card = make_card(v10=900.1, grade10_sold_median=300.0, grade10_sold_n=2)
    assert gates.apply_gates(card).drop_reason != "wrong_card_match"


# 2. ratio ceiling ----------------------------------------------------------


def test_ratio_ceiling_hit():
    card = make_card(v10=6100.0, v9=100.0, grade10_sold_median=6000.0)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "ratio_above_ceiling"


def test_ratio_ceiling_missed_at_60x():
    card = make_card(v10=6000.0, v9=100.0, grade10_sold_median=5900.0)
    assert gates.apply_gates(card).dropped is False


def test_ratio_ceiling_ignores_missing_v9():
    card = make_card(v10=6000.0, v9=math.nan, grade10_sold_median=5900.0)
    assert gates.apply_gates(card).drop_reason != "ratio_above_ceiling"


# 3. thin outlier -----------------------------------------------------------


def test_thin_outlier_hit():
    card = make_card(v9=math.nan, feed_sales_lifetime=1, v10=200.01,
                     grade10_sold_median=200.0)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "thin_outlier"


def test_thin_outlier_missed_with_v9():
    card = make_card(v9=120.0, feed_sales_lifetime=1, v10=250.0)
    assert gates.apply_gates(card).dropped is False


def test_thin_outlier_missed_with_two_sales():
    card = make_card(v9=math.nan, feed_sales_lifetime=2, v10=250.0)
    assert gates.apply_gates(card).drop_reason != "thin_outlier"


def test_thin_outlier_missed_at_200():
    card = make_card(v9=math.nan, feed_sales_lifetime=0, v10=200.0)
    assert gates.apply_gates(card).drop_reason != "thin_outlier"


# 4. unsupported high value --------------------------------------------------


def test_unsupported_hit():
    card = make_card(v10=1000.0, grade10_sold_n=0, grade10_sold_median=math.nan,
                     feed_sales_lifetime=2)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "unsupported_high_value"


def test_unsupported_missed_below_1000():
    card = make_card(v10=999.99, grade10_sold_n=0, grade10_sold_median=math.nan,
                     feed_sales_lifetime=2)
    assert gates.apply_gates(card).drop_reason != "unsupported_high_value"


def test_unsupported_missed_with_corroboration():
    card = make_card(v10=5000.0, grade10_sold_n=3, grade10_sold_median=4800.0,
                     feed_sales_lifetime=0)
    assert gates.apply_gates(card).dropped is False


def test_unsupported_missed_with_three_feed_sales():
    card = make_card(v10=5000.0, grade10_sold_n=0, grade10_sold_median=math.nan,
                     feed_sales_lifetime=3, ungraded_sold_median=55.0,
                     ungraded_sold_n=5, raw_price=50.0)
    assert gates.apply_gates(card).drop_reason != "unsupported_high_value"


# 5. repair ------------------------------------------------------------------


def test_repair_from_ungraded_median():
    # raw 9.99 < 30/3 -> repaired to 30
    card = make_card(raw_price=9.99, ungraded_sold_median=30.0, ungraded_sold_n=3)
    decision = gates.apply_gates(card)
    assert decision.dropped is False
    assert decision.raw_confidence == "repaired"
    assert decision.repair_source == "ungraded_sold_median"
    assert decision.raw_effective == 30.0


def test_no_repair_at_median_over_three():
    card = make_card(raw_price=10.0, ungraded_sold_median=30.0, ungraded_sold_n=3)
    decision = gates.apply_gates(card)
    assert decision.raw_confidence == "observed"
    assert decision.raw_effective == 10.0


def test_median_discarded_when_at_or_above_v10():
    # median >= V10 is noise -> structural-floor path; raw is fine there
    card = make_card(raw_price=10.0, ungraded_sold_median=300.0, ungraded_sold_n=5)
    decision = gates.apply_gates(card)
    assert decision.raw_confidence == "observed"


def test_median_needs_n3():
    card = make_card(raw_price=5.0, ungraded_sold_median=30.0, ungraded_sold_n=2)
    decision = gates.apply_gates(card)
    # no usable signal; raw 5 is above the 0.02*V9=2.4 floor -> untouched
    assert decision.raw_confidence == "observed"


def test_structural_floor_repairs_from_near_mint():
    # raw 2 < 0.02*V9 = 2.4; near-mint available -> repaired
    card = make_card(raw_price=2.0, ungraded_sold_n=0, ungraded_sold_median=math.nan,
                     near_mint_price=32.0)
    decision = gates.apply_gates(card)
    assert decision.dropped is False
    assert decision.raw_confidence == "repaired"
    assert decision.repair_source == "near_mint_price"
    assert decision.raw_effective == 32.0


def test_structural_floor_drops_without_near_mint():
    card = make_card(raw_price=2.0, ungraded_sold_n=0, ungraded_sold_median=math.nan,
                     near_mint_price=math.nan)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "raw_below_floor_no_signal"


def test_structural_floor_uses_v10_without_v9():
    # no V9: floor is 0.005*V10 = 1.5 ; raw 1.4 is below it
    card = make_card(v9=math.nan, feed_sales_lifetime=5, raw_price=1.4,
                     ungraded_sold_n=0, ungraded_sold_median=math.nan,
                     near_mint_price=math.nan)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "raw_below_floor_no_signal"


def test_structural_floor_missed_just_above():
    card = make_card(raw_price=2.5, ungraded_sold_n=0, ungraded_sold_median=math.nan,
                     near_mint_price=math.nan)
    assert gates.apply_gates(card).dropped is False


# 6. post-repair raw below $1 -----------------------------------------------


def test_post_repair_raw_below_one_dollar():
    # repaired to a sub-$1 median -> still dropped
    card = make_card(v10=5.0, v9=2.0, raw_price=0.2, grade10_sold_median=4.9,
                     ungraded_sold_median=0.9, ungraded_sold_n=5)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "raw_below_one_dollar"


def test_observed_raw_below_one_dollar_above_floor():
    # raw 0.9 >= 0.02*V9 (0.6) so no repair, but still below $1 -> dropped
    card = make_card(v10=40.0, v9=30.0, raw_price=0.9, grade10_sold_median=39.0,
                     ungraded_sold_n=0, ungraded_sold_median=math.nan)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "raw_below_one_dollar"


# 7. uncorroborated multiplier cap -------------------------------------------


def test_uncorroborated_multiplier_hit():
    card = make_card(v10=900.0, v9=400.0, raw_price=8.9, grade10_sold_n=0,
                     grade10_sold_median=math.nan, ungraded_sold_median=9.0,
                     ungraded_sold_n=5, feed_sales_lifetime=9)
    decision = gates.apply_gates(card)
    assert decision.dropped and decision.drop_reason == "uncorroborated_multiplier"


def test_corroborated_grail_exempt_from_cap():
    # 150x multiplier but graded-sold corroboration exists -> kept
    card = make_card(v10=900.0, v9=400.0, raw_price=6.0, grade10_sold_n=4,
                     grade10_sold_median=880.0, ungraded_sold_median=6.2,
                     ungraded_sold_n=5)
    assert gates.apply_gates(card).dropped is False


def test_multiplier_missed_at_100x():
    card = make_card(v10=890.0, v9=400.0, raw_price=8.9, grade10_sold_n=0,
                     grade10_sold_median=math.nan, ungraded_sold_median=9.0,
                     ungraded_sold_n=5)
    assert gates.apply_gates(card).dropped is False


def test_multiplier_uses_post_repair_raw():
    # raw 0.5 would be 1800x, but repair to the $30 median first -> 15x, kept
    card = make_card(v10=450.0, v9=200.0, raw_price=0.5, grade10_sold_n=0,
                     grade10_sold_median=math.nan, ungraded_sold_median=30.0,
                     ungraded_sold_n=5)
    decision = gates.apply_gates(card)
    assert decision.dropped is False
    assert decision.raw_confidence == "repaired"


# rankability ----------------------------------------------------------------


def test_rankability_all_gates():
    cfg = DEFAULT_CONFIG
    ok = gates.rankability(25.0, 3, 2.0, "mid", cfg)
    assert ok.rankable is True and ok.reasons == []

    low_npv = gates.rankability(24.99, 3, 2.0, "mid", cfg)
    assert low_npv.rankable is False and "npv_below_min" in low_npv.reasons

    illiquid = gates.rankability(25.0, 2, 2.0, "mid", cfg)
    assert illiquid.rankable is False and "insufficient_liquidity" in illiquid.reasons

    below_floor = gates.rankability(25.0, 3, 1.99, "mid", cfg)
    assert below_floor.rankable is False and "raw_below_era_floor" in below_floor.reasons

    modern_floor_is_1 = gates.rankability(25.0, 3, 1.0, "modern", cfg)
    assert modern_floor_is_1.rankable is True
