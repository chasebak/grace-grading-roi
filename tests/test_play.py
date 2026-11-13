"""The Grading Play Calculator's engine (grace/play.py) and its page files.

Why: the page computes in the browser, so two things must hold. The Python
engine must be right (hand-computed timelines and present values, the
invariants a discounting model has to obey, flags that fire exactly at
their boundaries, blank BGS rows that never reach an output), and the
committed page must stay tied to it (data files equal a fresh build, the
service tables equal the published reading typed below, the page follows
the page kit). tests/test_play_parity.py covers the third thing: calc.js
equals this engine on the fixtures.

The hand-computed values below were worked out on paper (the arithmetic is
in the comments), not by calling the code under test.
Not covered here: whether a published price is still current (only a fresh
reading of the official pages can tell), and whether the browser draws the
numbers correctly (the screenshot click-through does that, outside CI).
"""

import json
import re
from pathlib import Path

import pytest
import test_privacy_patterns as privacy

from grace import play, report, viz

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_FILES = (play.INDEX_PATH, play.APP_PATH, play.CALC_PATH, play.TIERS_PATH, play.PARITY_PATH)

# The published tables as read on 2026-11-12, typed a second time on
# purpose: an accidental edit to grace/play.py fails here.
# (level, $ per card, max value, fastest days, slowest days, minimum cards)
PSA_PUBLISHED = [
    ("Value", 39.99, 500, 90, 100, None),
    ("Standard", 59.99, 1000, 70, 80, None),
    ("Priority", 79.99, 1500, 50, 60, None),
    ("Express", 199.00, 2500, 20, 30, None),
    ("Super Express", 349.00, 5000, 10, 15, None),
    ("Premier", 599.00, 10000, 7, 10, None),
]
TAG_PUBLISHED = [
    ("Value", 29.95, 250, 90, 90, 10),
    ("Base", 39.95, 500, 60, 60, 5),
    ("Standard", 59.95, 1000, 45, 45, None),
    ("Priority", 79.95, 1500, 30, 30, None),
    ("Express", 149.95, 2500, 10, 10, None),
    ("Premier", 299.95, 5000, 3, 5, None),
]


def page_text(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fresh_outputs():
    return play.build_outputs()


@pytest.fixture(scope="module")
def committed_tiers():
    return json.loads(page_text(play.TIERS_PATH))


@pytest.fixture(scope="module")
def committed_parity():
    return json.loads(page_text(play.PARITY_PATH))


def solo_company(cap_rule=play.CAP_INFORMATIONAL):
    """One made-up level: $100, 5 business days (= 7 calendar days)."""
    return {
        "key": "solo",
        "name": "Solo",
        "cap_rule": cap_rule,
        "tiers": [play.service_tier("Only", 100.0, None, 5, 5, None)],
    }


def solo_inputs(fee_timing, weekly_rate_pct):
    """Round numbers so every date is a whole week and every PV is exact.

    A certain $1,000 sale with no selling fee; $10 out, $25 back; 7 days
    each way, 7 days to sell. So: graded on day 14, sold on day 28.
    """
    return {
        "raw_value_now": 200.0,
        "quantity": 1,
        "sell_fee_pct": 0.0,
        "ship_to_buyer": 0.0,
        "days_out": 7,
        "days_back": 7,
        "ship_out_batch": 10.0,
        "ship_back_batch": 25.0,
        "sell_days": 7,
        "weekly_rate_pct": weekly_rate_pct,
        "range_mode": play.RANGE_MIDPOINT,
        "companies": {
            "solo": {
                "p_top_pct": 100.0,
                "p_second_pct": 0.0,
                "price_top": 1000.0,
                "price_second": 0.0,
                "price_lower": 0.0,
                "fee_timing": fee_timing,
            }
        },
    }


def tier_by_key(result, key):
    for tier in result["tiers"]:
        if tier["key"] == key:
            return tier
    return None


def default_result():
    return play.evaluate(play.default_inputs(), play.service_companies())


# ---- hand-computed formulas ----------------------------------------------


def test_present_value_hand_check():
    # 100 two weeks out at 10% a week: 100 / 1.1^2 = 100 / 1.21 = 82.6446281
    assert play.present_value(100.0, 14, 0.10) == pytest.approx(82.6446281, abs=1e-7)
    # today's dollar is a dollar; at rate 0 so is any later one
    assert play.present_value(100.0, 0, 0.10) == 100.0
    assert play.present_value(100.0, 365, 0.0) == 100.0


def test_business_days_become_calendar_days_at_seven_fifths():
    # 95 x 7/5 = 133 ; 5 x 7/5 = 7 ; 8.5 x 7/5 = 11.9
    assert play.business_to_calendar_days(95) == 133
    assert play.business_to_calendar_days(5) == 7
    assert play.business_to_calendar_days(8.5) == pytest.approx(11.9)


def test_range_mode_picks_the_stated_end_or_the_midpoint():
    wide = play.service_tier("x", 1.0, None, 90, 100, None)
    assert play.turnaround_business_days(wide, play.RANGE_FASTEST) == 90
    assert play.turnaround_business_days(wide, play.RANGE_MIDPOINT) == 95
    assert play.turnaround_business_days(wide, play.RANGE_SLOWEST) == 100
    single = play.service_tier("x", 1.0, None, None, 8, None)
    for mode in play.RANGE_MODES:
        assert play.turnaround_business_days(single, mode) == 8


def test_expected_net_sale_and_sell_raw_hand_check():
    # The page's example: odds 50/35/15, prices 1500/500/250, fee 13%, ship 5
    #   top    0.50 * (1500*0.87 - 5) = 0.50 * 1300.0 = 650.000
    #   second 0.35 * ( 500*0.87 - 5) = 0.35 *  430.0 = 150.500
    #   lower  0.15 * ( 250*0.87 - 5) = 0.15 *  212.5 =  31.875
    #   expected net sale = 832.375 ; sell raw today = 250*0.87 - 5 = 212.5
    inputs = play.default_inputs()
    company_inputs = inputs["companies"]["psa"]
    assert play.lower_odds_pct(company_inputs) == 15.0
    assert play.expected_net_sale(company_inputs, 0.13, 5.0) == pytest.approx(832.375, abs=1e-9)
    assert play.sell_raw_today(inputs) == pytest.approx(212.5, abs=1e-9)


def test_weekly_rate_presets_hand_check():
    # (1 + w)^52 = 1.12  ->  w = exp(ln(1.12)/52) - 1 = 0.0021817744 (0.218%)
    weekly_pct = play.weekly_pct_from_annual(12.0)
    assert weekly_pct == pytest.approx(0.21817744, abs=1e-7)
    assert play.annual_pct_from_weekly(weekly_pct) == pytest.approx(12.0, abs=1e-9)
    presets = play.rate_presets()
    assert presets[0] == {"label": "another project at 6% a week", "weekly_rate_pct": 6.0}
    assert presets[1]["label"] == "stock market, 12% a year"
    assert presets[1]["weekly_rate_pct"] == weekly_pct


def test_timeline_hand_check_on_the_slowest_published_level():
    # PSA Value, midpoint: 95 business days = 133 calendar days.
    # received day 5 ; graded 5 + 133 = 138 ; back 143 ; sold 143 + 45 = 188
    tier = tier_by_key(default_result(), "psa/0")
    assert tier["label"] == "PSA Value"
    assert tier["turnaround_business_days"] == 95
    assert tier["turnaround_calendar_days"] == 133
    assert tier["day_received"] == 5
    assert tier["day_graded"] == 138
    assert tier["day_back"] == 143
    assert tier["day_sale"] == 188
    # PSA charges after grading by default: the fee and return shipping on day 138
    days = []
    for flow in tier["flows"]:
        days.append((flow["what"], flow["day"]))
    assert days == [("ship_out", 0), ("grading_fee", 138), ("ship_back", 138), ("sale", 188)]


def test_fee_after_grading_hand_check():
    # 25% a week: 1.25^2 = 1.5625 (day 14), 1.25^4 = 2.44140625 (day 28)
    #   ship out   10 on day 0   ->  10
    #   fee       100 on day 14  ->  100 / 1.5625 = 64
    #   ship back  25 on day 14  ->   25 / 1.5625 = 16
    #   sale     1000 on day 28  -> 1000 / 2.44140625 = 409.6
    #   NPV = 409.6 - (10 + 64 + 16) = 319.6 ; raw today 200 ; value = 119.6
    inputs = solo_inputs(play.FEE_AFTER_GRADING, 25.0)
    result = play.evaluate(inputs, [solo_company()])
    tier = result["tiers"][0]
    assert (tier["day_received"], tier["day_graded"], tier["day_back"], tier["day_sale"]) == (
        7,
        14,
        21,
        28,
    )
    assert tier["fee_day"] == 14
    assert tier["fee_value_today"] == pytest.approx(64.0, abs=1e-9)
    assert tier["pv_costs"] == pytest.approx(90.0, abs=1e-9)
    assert tier["pv_proceeds"] == pytest.approx(409.6, abs=1e-9)
    assert tier["npv"] == pytest.approx(319.6, abs=1e-9)
    assert result["sell_raw_today"] == 200.0
    assert tier["value_of_play"] == pytest.approx(119.6, abs=1e-9)
    assert result["best"] == "solo/0"
    assert result["play"] == play.PLAY_GRADE


def test_fee_at_checkout_hand_check():
    # Same card, fee paid on day 0: costs 10 + 100 + 16 = 126
    #   NPV = 409.6 - 126 = 283.6 ; value = 83.6
    inputs = solo_inputs(play.FEE_AT_CHECKOUT, 25.0)
    tier = play.evaluate(inputs, [solo_company()])["tiers"][0]
    assert tier["fee_day"] == 0
    assert tier["fee_value_today"] == 100.0
    assert tier["pv_costs"] == pytest.approx(126.0, abs=1e-9)
    assert tier["npv"] == pytest.approx(283.6, abs=1e-9)
    assert tier["value_of_play"] == pytest.approx(83.6, abs=1e-9)


def test_rate_zero_npv_is_the_plain_sum():
    # No discounting: NPV = 1000 - (10 + 100 + 25) = 865, whenever the fee is paid
    for fee_timing in play.FEE_TIMINGS:
        tier = play.evaluate(solo_inputs(fee_timing, 0.0), [solo_company()])["tiers"][0]
        assert tier["npv"] == 865.0
        assert tier["npv"] == tier["undiscounted_profit"]
        assert tier["value_of_play"] == tier["undiscounted_value_of_play"] == 665.0
    # and on the published tables: PSA Value = 832.375 - (15 + 39.99 + 20) = 757.385
    inputs = play.default_inputs()
    inputs["weekly_rate_pct"] = 0.0
    tier = tier_by_key(play.evaluate(inputs, play.service_companies()), "psa/0")
    assert tier["npv"] == pytest.approx(757.385, abs=1e-9)
    assert tier["pv_costs"] == pytest.approx(tier["sticker_costs"], abs=1e-12)


def test_fee_paid_later_is_worth_less_today():
    """The same $100 fee: on day 0 it costs $100 today, on day 14 less."""
    upfront = play.evaluate(solo_inputs(play.FEE_AT_CHECKOUT, 6.0), [solo_company()])["tiers"][0]
    later = play.evaluate(solo_inputs(play.FEE_AFTER_GRADING, 6.0), [solo_company()])["tiers"][0]
    assert upfront["fee_value_today"] == 100.0
    # 100 / 1.06^2 = 100 / 1.1236 = 88.9996
    assert later["fee_value_today"] == pytest.approx(88.9996, abs=1e-4)
    assert later["fee_value_today"] < upfront["fee_value_today"]
    assert later["npv"] > upfront["npv"]


def test_batch_shipping_is_shared_per_card():
    # $15 out and $20 back over 10 cards = $1.50 and $2.00 a card
    inputs = play.default_inputs()
    inputs["quantity"] = 10
    result = play.evaluate(inputs, play.service_companies())
    tier = tier_by_key(result, "psa/0")
    amounts = {}
    for flow in tier["flows"]:
        amounts[flow["what"]] = flow["amount"]
    assert amounts["ship_out"] == 1.5
    assert amounts["ship_back"] == 2.0
    assert amounts["grading_fee"] == 39.99
    assert result["best_batch_value_of_play"] == result["best_value_of_play"] * 10


# ---- invariants -------------------------------------------------------------


def test_higher_rate_never_raises_the_pv_of_proceeds():
    companies = play.service_companies()
    rates = (0.0, 0.25, 1.0, 3.0, 6.0, 10.0, 25.0)
    previous = None
    for rate in rates:
        inputs = play.default_inputs()
        inputs["weekly_rate_pct"] = rate
        result = play.evaluate(inputs, companies)
        current = {}
        for tier in result["tiers"]:
            current[tier["key"]] = tier["pv_proceeds"]
        assert len(current) == 12
        if previous is not None:
            keys = sorted(current.keys())
            for key in keys:
                assert current[key] < previous[key], (rate, key)
        previous = current


def test_longer_sell_time_never_raises_npv():
    companies = play.service_companies()
    for rate in (0.0, 6.0):
        previous = None
        for sell_days in (0, 10, 45, 120):
            inputs = play.default_inputs()
            inputs["weekly_rate_pct"] = rate
            inputs["sell_days"] = sell_days
            result = play.evaluate(inputs, companies)
            current = {}
            for tier in result["tiers"]:
                current[tier["key"]] = tier["npv"]
            if previous is not None:
                keys = sorted(current.keys())
                for key in keys:
                    if rate > 0:
                        assert current[key] < previous[key], (sell_days, key)
                    else:
                        assert current[key] == previous[key], (sell_days, key)
            previous = current


def test_paying_after_grading_is_never_worse_than_paying_up_front():
    companies = play.service_companies()
    for rate in (0.0, 0.5, 6.0):
        npv_by_timing = {}
        for fee_timing in play.FEE_TIMINGS:
            inputs = play.default_inputs()
            inputs["weekly_rate_pct"] = rate
            company_keys = sorted(inputs["companies"].keys())
            for key in company_keys:
                inputs["companies"][key]["fee_timing"] = fee_timing
            result = play.evaluate(inputs, companies)
            values = {}
            for tier in result["tiers"]:
                values[tier["key"]] = tier["npv"]
            npv_by_timing[fee_timing] = values
        after = npv_by_timing[play.FEE_AFTER_GRADING]
        upfront = npv_by_timing[play.FEE_AT_CHECKOUT]
        keys = sorted(after.keys())
        assert len(keys) == 12
        for key in keys:
            if rate > 0:
                assert after[key] > upfront[key], (rate, key)
            else:
                assert after[key] == upfront[key], key


def test_identical_tiers_give_identical_values_across_companies():
    twins = play.twin_companies()
    inputs = play.default_inputs()
    inputs["companies"] = {
        "north": play.example_company_inputs(play.FEE_AFTER_GRADING),
        "south": play.example_company_inputs(play.FEE_AFTER_GRADING),
    }
    inputs["quantity"] = 2
    result = play.evaluate(inputs, twins)
    assert len(result["tiers"]) == 4
    compared = ("npv", "value_of_play", "pv_costs", "pv_proceeds", "day_sale", "eligible")
    for tier_index in (0, 1):
        north = tier_by_key(result, f"north/{tier_index}")
        south = tier_by_key(result, f"south/{tier_index}")
        for name in compared:
            assert north[name] == south[name], name
        north_curve = result["rate_grid"]["value_of_play"][north["key"]]
        south_curve = result["rate_grid"]["value_of_play"][south["key"]]
        assert north_curve == south_curve
    # a tie keeps table order: north is listed first
    assert result["best"].startswith("north/")


def test_value_of_the_play_is_npv_minus_selling_raw_today():
    result = default_result()
    assert result["sell_raw_today"] == pytest.approx(212.5)
    for tier in result["tiers"]:
        assert tier["value_of_play"] == tier["npv"] - result["sell_raw_today"]
        assert tier["npv"] == tier["pv_proceeds"] - tier["pv_costs"]


def test_ranking_is_sorted_and_gaps_are_against_the_best():
    result = default_result()
    ranking = result["ranking"]
    assert len(ranking) == 8
    assert result["best"] == ranking[0]["key"]
    assert ranking[0]["gap_to_best"] == 0
    previous_value = ranking[0]["value_of_play"]
    for entry in ranking:
        assert entry["value_of_play"] <= previous_value
        assert entry["gap_to_best"] == entry["value_of_play"] - ranking[0]["value_of_play"]
        assert entry["gap_to_best"] <= 0
        tier = tier_by_key(result, entry["key"])
        assert tier["eligible"]
        assert tier["gap_to_best"] == entry["gap_to_best"]
        previous_value = entry["value_of_play"]
    for tier in result["tiers"]:
        if not tier["eligible"]:
            assert tier["gap_to_best"] is None


def test_rate_grid_matches_single_evaluations():
    """Chart 2's lines are the same model as chart 1's bars, rate by rate."""
    companies = play.service_companies()
    result = default_result()
    grid = result["rate_grid"]
    assert grid["rates_pct"][0] == 0.0
    assert grid["rates_pct"][1] == 0.25
    assert grid["rates_pct"][-1] == 10.0
    assert len(grid["rates_pct"]) == 41
    eligible_keys = []
    for tier in result["tiers"]:
        if tier["eligible"]:
            eligible_keys.append(tier["key"])
    assert grid["keys"] == eligible_keys
    rate_count = len(grid["rates_pct"])
    positions = range(rate_count)
    for grid_index in positions:
        inputs = play.default_inputs()
        inputs["weekly_rate_pct"] = grid["rates_pct"][grid_index]
        single = play.evaluate(inputs, companies)
        for key in grid["keys"]:
            expected = tier_by_key(single, key)["value_of_play"]
            assert grid["value_of_play"][key][grid_index] == expected, (grid_index, key)
        ranked = play.rank_at_rate(grid, grid_index)
        assert ranked == single["ranking"]


def test_best_by_rate_marks_where_the_better_level_changes():
    result = default_result()
    segments = result["best_by_rate"]
    grid = result["rate_grid"]
    assert segments[0]["from_pct"] == 0.0
    assert segments[-1]["to_pct"] == 10.0
    assert len(segments) > 1  # the example has crossings; a single run would hide them
    rates = grid["rates_pct"]
    rate_count = len(rates)
    positions = range(rate_count)
    for grid_index in positions:
        ranked = play.rank_at_rate(grid, grid_index)
        expected = ranked[0]["key"]
        if ranked[0]["value_of_play"] <= 0:
            expected = play.RAW_KEY
        covering = []
        for segment in segments:
            if segment["from_pct"] <= rates[grid_index] <= segment["to_pct"]:
                covering.append(segment["winner"])
        assert covering == [expected], rates[grid_index]


def test_sell_raw_is_the_play_when_no_level_beats_it():
    # A card worth more raw than any graded outcome can return.
    inputs = play.default_inputs()
    inputs["raw_value_now"] = 5000.0
    result = play.evaluate(inputs, play.service_companies())
    assert result["play"] == play.PLAY_SELL_RAW
    assert result["best_value_of_play"] < 0
    assert result["best_by_rate"] == [{"winner": play.RAW_KEY, "from_pct": 0.0, "to_pct": 10.0}]


# ---- eligibility flags --------------------------------------------------------


def evaluate_with(quantity, price_top):
    inputs = play.default_inputs()
    inputs["quantity"] = quantity
    company_keys = sorted(inputs["companies"].keys())
    for key in company_keys:
        inputs["companies"][key]["price_top"] = price_top
    return play.evaluate(inputs, play.service_companies())


def test_flags_fire_exactly_at_the_boundaries():
    # minimum cards: TAG Value needs 10, TAG Base needs 5
    nine = tier_by_key(evaluate_with(9, 200.0), "tag/0")
    ten = tier_by_key(evaluate_with(10, 200.0), "tag/0")
    assert nine["below_minimum"] and not nine["eligible"]
    assert nine["flags"] == [play.FLAG_BELOW_MINIMUM]
    assert not ten["below_minimum"] and ten["eligible"] and ten["flags"] == []
    four = tier_by_key(evaluate_with(4, 200.0), "tag/1")
    five = tier_by_key(evaluate_with(5, 200.0), "tag/1")
    assert four["below_minimum"] and not four["eligible"]
    assert not five["below_minimum"] and five["eligible"]

    # PSA's max insured value: at the cap is fine, a cent over moves it up a level
    at_cap = tier_by_key(evaluate_with(1, 500.0), "psa/0")
    over_cap = tier_by_key(evaluate_with(1, 500.01), "psa/0")
    assert not at_cap["over_cap"] and at_cap["eligible"] and at_cap["flags"] == []
    assert over_cap["over_cap"] and not over_cap["eligible"]
    assert over_cap["flags"] == [play.FLAG_OVER_CAP]

    # TAG's protection amount is informational: flagged, still a play
    tag_at = tier_by_key(evaluate_with(10, 250.0), "tag/0")
    tag_over = tier_by_key(evaluate_with(10, 250.01), "tag/0")
    assert not tag_at["over_cap"] and tag_at["flags"] == []
    assert tag_over["over_cap"] and tag_over["eligible"]
    assert tag_over["flags"] == [play.FLAG_OVER_CAP]


def test_flagged_levels_stay_in_the_table_and_leave_the_ranking():
    """Never silently dropped: a blocked level is listed with its flag."""
    result = evaluate_with(1, 1500.0)
    assert len(result["tiers"]) == 12
    blocked = []
    for tier in result["tiers"]:
        if not tier["eligible"]:
            blocked.append(tier["key"])
            assert tier["flags"] != []
    # PSA Value and Standard are over their caps; TAG Value and Base need a bigger batch
    assert blocked == ["psa/0", "psa/1", "tag/0", "tag/1"]
    ranked_keys = []
    for entry in result["ranking"]:
        ranked_keys.append(entry["key"])
    for key in blocked:
        assert key not in ranked_keys
        assert key not in result["rate_grid"]["keys"]
        assert key not in result["rate_grid"]["value_of_play"]
    companies = play.service_companies()
    assert companies[0]["cap_flag_text"] == (
        "over this level's max insured value — PSA would move it up a level"
    )
    assert companies[1]["cap_flag_text"] == "above the protection amount"


def test_impossible_odds_skip_the_company_and_say_so():
    inputs = play.default_inputs()
    inputs["companies"]["tag"]["p_top_pct"] = 70.0
    inputs["companies"]["tag"]["p_second_pct"] = 45.0
    result = play.evaluate(inputs, play.service_companies())
    assert result["skipped_companies"] == ["tag"]
    for tier in result["tiers"]:
        assert tier["company"] == "psa"
    # exactly 100% is possible; typed decimals that sum to 100 are too
    assert play.odds_are_valid({"p_top_pct": 60.0, "p_second_pct": 40.0})
    assert play.odds_are_valid({"p_top_pct": 33.3, "p_second_pct": 66.7})
    assert not play.odds_are_valid({"p_top_pct": 60.0, "p_second_pct": 40.5})
    assert not play.odds_are_valid({"p_top_pct": -1.0, "p_second_pct": 40.0})


# ---- BGS ships unverified and empty --------------------------------------------


def all_output_keys(result):
    keys = []
    for tier in result["tiers"]:
        keys.append(tier["key"])
    for entry in result["ranking"]:
        keys.append(entry["key"])
    keys.extend(result["rate_grid"]["keys"])
    keys.extend(sorted(result["rate_grid"]["value_of_play"].keys()))
    for segment in result["best_by_rate"]:
        keys.append(segment["winner"])
    if result["best"] is not None:
        keys.append(result["best"])
    return keys


def test_empty_bgs_rows_never_appear():
    companies = play.service_companies()
    bgs = companies[2]
    assert bgs["key"] == "bgs"
    assert bgs["verified"] is False
    assert len(bgs["tiers"]) == play.BGS_EMPTY_ROWS
    for tier in bgs["tiers"]:
        assert tier == play.empty_tier()
        assert not play.tier_is_complete(tier)
    for mode in play.RANGE_MODES:
        inputs = play.default_inputs()
        inputs["range_mode"] = mode
        result = play.evaluate(inputs, companies)
        output_keys = all_output_keys(result)
        assert len(output_keys) > 12
        for key in output_keys:
            assert not key.startswith("bgs"), key
        assert result["skipped_companies"] == []


def test_a_half_typed_row_stays_out_and_a_complete_one_comes_in():
    companies = play.service_companies()
    # made-up test values, typed the way a user would
    companies[2]["tiers"][0] = play.service_tier("fee only", 50.0, None, None, None, None)
    companies[2]["tiers"][1] = play.service_tier("days only", None, None, 10, 20, None)
    companies[2]["tiers"][2] = play.service_tier("negative fee", -5.0, None, 10, 20, None)
    result = play.evaluate(play.default_inputs(), companies)
    output_keys = all_output_keys(result)
    for key in output_keys:
        assert not key.startswith("bgs"), key
    companies[2]["tiers"][3] = play.service_tier("", 50.0, None, 10, None, None)
    result = play.evaluate(play.default_inputs(), companies)
    typed = tier_by_key(result, "bgs/3")
    assert typed is not None
    assert typed["label"] == "BGS row 4"
    assert typed["turnaround_business_days"] == 10
    assert "bgs/3" in result["rate_grid"]["keys"]


def test_the_data_file_carries_no_bgs_number(committed_tiers):
    bgs = committed_tiers["companies"][2]
    assert bgs["key"] == "bgs"
    assert bgs["verified"] is False
    assert bgs["as_of"] is None
    assert bgs["note"] == (
        "BGS prices were not read from beckett.com — enter the current ones from "
        "https://www.beckett.com/grading/pricing"
    )
    assert bgs["fee_timing_source"] == "unverified — set it"
    for tier in bgs["tiers"]:
        assert tier == {
            "name": "",
            "fee": None,
            "max_value": None,
            "days_min": None,
            "days_max": None,
            "min_cards": None,
        }
    assert play.bundle_problems(committed_tiers, {"fixtures": [None] * play.MIN_FIXTURES}) == []


def test_control_a_number_on_an_unverified_tier_is_caught(committed_tiers):
    planted = json.loads(json.dumps(committed_tiers))
    planted["companies"][2]["tiers"][0]["fee"] = 1.0
    problems = play.bundle_problems(planted, {"fixtures": [None] * play.MIN_FIXTURES})
    assert problems == ["bgs: an unverified tier carries a number"]
    few = play.bundle_problems(committed_tiers, {"fixtures": [None] * (play.MIN_FIXTURES - 1)})
    assert few == [f"parity.json holds {play.MIN_FIXTURES - 1} fixtures, under {play.MIN_FIXTURES}"]


# ---- the published tables and the data files ------------------------------------


def published_rows(company):
    rows = []
    for tier in company["tiers"]:
        rows.append(
            (
                tier["name"],
                tier["fee"],
                tier["max_value"],
                tier["days_min"],
                tier["days_max"],
                tier["min_cards"],
            )
        )
    return rows


def test_service_tables_are_the_published_reading(committed_tiers):
    psa, tag, _bgs = committed_tiers["companies"]
    assert published_rows(psa) == PSA_PUBLISHED
    assert published_rows(tag) == TAG_PUBLISHED
    assert psa["source_url"] == "https://www.psacard.com/services/tradingcardgrading"
    assert tag["source_url"] == "https://taggrading.com/pages/pricing"
    assert psa["as_of"] == tag["as_of"] == committed_tiers["as_of"] == "2026-11-12"
    assert psa["verified"] is True and tag["verified"] is True
    assert psa["source_quotes"] == [
        "Estimated Turnaround Times are estimates only and not guaranteed"
    ]
    assert tag["source_quotes"] == [
        "Turnaround times are estimates in business days from intake and are not guaranteed",
        "No value upcharges",
    ]
    assert published_rows(play.service_companies()[0]) == PSA_PUBLISHED
    assert published_rows(play.service_companies()[1]) == TAG_PUBLISHED


def test_fee_timing_defaults_and_their_source_labels(committed_tiers):
    psa, tag, bgs = committed_tiers["companies"]
    assert psa["fee_timing_default"] == play.FEE_AFTER_GRADING
    assert "author's own PSA orders" in psa["fee_timing_source"]
    assert "not from a PSA page" in psa["fee_timing_source"]
    assert tag["fee_timing_default"] == play.FEE_AT_CHECKOUT
    assert tag["fee_timing_source"] == "TAG's order flow is a store checkout (help.taggrading.com)"
    assert bgs["fee_timing_default"] == play.FEE_AT_CHECKOUT
    defaults = committed_tiers["defaults"]
    for company in (psa, tag, bgs):
        assert defaults["companies"][company["key"]]["fee_timing"] == company["fee_timing_default"]


def test_defaults_in_the_data_file_are_the_engines(committed_tiers):
    defaults = committed_tiers["defaults"]
    assert defaults == report.round_floats(play.default_inputs())
    assert defaults["sell_fee_pct"] == 13.0
    assert defaults["ship_to_buyer"] == 5.0
    assert (defaults["days_out"], defaults["days_back"]) == (5, 5)
    assert (defaults["ship_out_batch"], defaults["ship_back_batch"]) == (15.0, 20.0)
    assert defaults["sell_days"] == 45
    assert defaults["weekly_rate_pct"] == 6.0
    assert defaults["range_mode"] == play.RANGE_MIDPOINT
    # the same example for every company: no company-specific price premium
    psa_example = dict(defaults["companies"]["psa"])
    psa_example.pop("fee_timing")
    for key in ("tag", "bgs"):
        other = dict(defaults["companies"][key])
        other.pop("fee_timing")
        assert other == psa_example
    assert committed_tiers["example_label"] == "example — enter your own comps per company"
    assert committed_tiers["presets"] == report.round_floats(play.rate_presets())


def test_committed_data_match_a_fresh_build(fresh_outputs):
    assert play.check(REPO_ROOT, fresh_outputs) == []


def test_build_is_deterministic(fresh_outputs):
    assert play.build_outputs() == fresh_outputs


def copy_page_tree(tmp_path, outputs):
    """A throwaway docs/ tree holding the given outputs and the static files."""
    static_paths = (play.INDEX_PATH, play.APP_PATH, play.CALC_PATH, play.MAIN_INDEX_PATH)
    for path in static_paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / path).read_bytes())
    paths = sorted(outputs.keys())
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(outputs[path], encoding="utf-8")


def test_control_stale_and_missing_data_are_caught(tmp_path, fresh_outputs):
    copy_page_tree(tmp_path, fresh_outputs)
    assert play.check(tmp_path, fresh_outputs) == []

    tiers_path = tmp_path / play.TIERS_PATH
    text = tiers_path.read_text(encoding="utf-8")
    mutated = text.replace("39.99", "38.99", 1)
    assert mutated != text
    tiers_path.write_text(mutated, encoding="utf-8")
    assert play.check(tmp_path, fresh_outputs) == [
        f"{play.TIERS_PATH} is stale; run python -m grace.play build"
    ]

    tiers_path.write_text(text, encoding="utf-8")
    parity_path = tmp_path / play.PARITY_PATH
    parity_text = parity_path.read_text(encoding="utf-8")
    changed_number = parity_text.replace('"day_sale": 188', '"day_sale": 189', 1)
    assert changed_number != parity_text
    parity_path.write_text(changed_number, encoding="utf-8")
    assert play.check(tmp_path, fresh_outputs) == [
        f"{play.PARITY_PATH} is stale; run python -m grace.play build"
    ]

    parity_path.unlink()
    problems = play.check(tmp_path, fresh_outputs)
    assert f"{play.PARITY_PATH} is missing; run python -m grace.play build" in problems


def test_check_reads_parity_numbers_at_the_parity_tolerance(tmp_path, fresh_outputs):
    """Another platform's pow() may differ in the last bit, which can move a
    stored 10th digit; that is not a stale file. A real change still is."""
    copy_page_tree(tmp_path, fresh_outputs)
    parity_path = tmp_path / play.PARITY_PATH
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    first_tier = parity["fixtures"][0]["expected"]["tiers"][0]
    original_npv = first_tier["npv"]
    first_tier["npv"] = original_npv * (1 + 1e-11)
    parity_path.write_text(report.to_json_text(parity), encoding="utf-8")
    assert parity_path.read_text(encoding="utf-8") != fresh_outputs[play.PARITY_PATH]
    assert play.check(tmp_path, fresh_outputs) == []

    first_tier["npv"] = original_npv * (1 + 1e-8)
    parity_path.write_text(report.to_json_text(parity), encoding="utf-8")
    assert play.check(tmp_path, fresh_outputs) == [
        f"{play.PARITY_PATH} is stale; run python -m grace.play build"
    ]


def test_control_broken_static_files_are_caught(tmp_path, fresh_outputs):
    copy_page_tree(tmp_path, fresh_outputs)
    index = tmp_path / play.INDEX_PATH
    html = index.read_text(encoding="utf-8")
    html = html.replace("integrity=", "data-x=", 1)
    html = html.replace("../tokens.css", "tokens.css", 1)
    index.write_text(html, encoding="utf-8")
    main_index = tmp_path / play.MAIN_INDEX_PATH
    main_html = main_index.read_text(encoding="utf-8")
    main_index.write_text(main_html.replace(play.LINK_TO_PLAY, 'href="#"'), encoding="utf-8")
    assert play.check(tmp_path, fresh_outputs) == [
        f"{play.INDEX_PATH} must carry the pinned Chart.js tag exactly once",
        f"{play.INDEX_PATH} must link ../tokens.css exactly once",
        f"{play.MAIN_INDEX_PATH} must link to the calculator",
    ]
    (tmp_path / play.CALC_PATH).unlink()
    assert play.check(tmp_path, fresh_outputs) == [f"{play.CALC_PATH} is missing"]


def test_cli_check_exits_zero_on_the_committed_tree(capsys):
    assert play.main(["build", "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


# ---- page kit conventions -----------------------------------------------------


def test_chart_js_tag_and_hash_match_the_main_page():
    html = page_text(play.INDEX_PATH)
    main_html = page_text(viz.INDEX_PATH)
    assert html.count(viz.CHART_JS_TAG) == 1
    assert main_html.count(viz.CHART_JS_TAG) == 1
    integrity = re.findall(r'integrity="([^"]+)"', html)
    assert integrity == re.findall(r'integrity="([^"]+)"', main_html)
    assert len(integrity) == 1


def test_page_links_the_shared_tokens_and_loads_no_other_external_code():
    html = page_text(play.INDEX_PATH)
    links = re.findall(r'<link[^>]*href="([^"]+)"', html)
    assert links == ["../tokens.css"]
    sources = re.findall(r'<script[^>]*src="([^"]+)"', html)
    chart_src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
    assert sources == [chart_src, "calc.js", "app.js"]
    for path in (play.APP_PATH, play.CALC_PATH):
        script = page_text(path)
        assert "http://" not in script
        assert "https://" not in script


def test_the_two_pages_link_to_each_other():
    assert play.LINK_TO_MAIN in page_text(play.INDEX_PATH)
    assert play.LINK_TO_PLAY in page_text(viz.INDEX_PATH)


def test_calc_js_is_pure_and_loadable_by_node():
    """Formulas only: no DOM, no network, and a module.exports guard."""
    calc = page_text(play.CALC_PATH)
    assert "document" not in calc
    assert "fetch(" not in calc
    assert "module.exports = api" in calc
    window_lines = []
    calc_lines = calc.splitlines()
    for line in calc_lines:
        if "window" in line and not line.lstrip().startswith("//"):
            window_lines.append(line.strip())
    assert window_lines == ['})(typeof window !== "undefined" ? window : this);']


def test_app_js_reads_and_draws_but_holds_no_formulas():
    """A coarse guard on the split: the discounting and the expectation live
    in calc.js. Formatting and layout arithmetic are allowed in app.js."""
    app = page_text(play.APP_PATH)
    assert "calc.evaluate(state.inputs, state.companies)" in app
    assert "calc.rankAtRate(" in app
    assert "Math.pow(" not in app
    assert "**" not in app
    assert ".reduce(" not in app
    assert "/ 7" not in app
    assert "7 / 5" not in app


def test_fallback_message_is_the_kit_text():
    html = page_text(play.INDEX_PATH)
    assert html.count("chart unavailable, table shown") == 2
    app = page_text(play.APP_PATH)
    assert "window.Chart" in app
    assert "showChartFallback" in app


def test_page_text_the_spec_requires(committed_tiers):
    html = page_text(play.INDEX_PATH)
    assert play.PROVENANCE in html
    assert committed_tiers["provenance"] == play.PROVENANCE
    assert "What this does NOT claim" in html
    assert "estimates, not guarantees" in html
    assert "Not investment advice" in html
    assert "BGS is not verified" in html
    assert "are your inputs" in html
    assert f"<title>{play.TITLE}</title>" in html
    assert f"<h1>{play.TITLE}</h1>" in html
    app = page_text(play.APP_PATH)
    assert "company.note" in app  # the BGS note is shown from the data file
    assert "company.source_url" in app
    assert "company.fee_timing_source" in app


def test_every_result_widget_has_a_why_box_and_a_provenance_pill(committed_tiers):
    html = page_text(play.INDEX_PATH)
    why_keys = re.findall(r'data-why="([a-z_]+)"', html)
    assert sorted(why_keys) == sorted(play.PLAY_WHY.keys())
    assert committed_tiers["why"] == play.PLAY_WHY
    prov_keys = re.findall(r'data-prov="([a-z_]+)"', html)
    assert sorted(set(prov_keys)) == sorted(committed_tiers["provenance_labels"].keys())
    why_entries = list(play.PLAY_WHY.values())
    for entry in why_entries:
        assert sorted(entry.keys()) == ["kind", "lie", "question"]


def test_why_boxes_name_tests_that_exist():
    """A why box that cites a guard must cite a real test."""
    sources = page_text("tests/test_play.py") + page_text("tests/test_play_parity.py")
    cited = []
    why_entries = list(play.PLAY_WHY.values())
    for entry in why_entries:
        cited.extend(re.findall(r"test_[a-z_]+", entry["lie"]))
    assert len(cited) >= 4
    for name in cited:
        if name in ("test_play", "test_play_parity"):
            continue
        assert f"def {name}(" in sources, name


def test_page_files_carry_no_leak_shapes():
    """The generic privacy patterns, on the calculator's files by name."""
    lines = []
    for path in PAGE_FILES:
        text_lines = page_text(path).splitlines()
        numbered = enumerate(text_lines, start=1)
        for line_number, line in numbered:
            lines.append((path, line_number, line))
    assert len(lines) > 1000
    assert privacy.find_hits(lines, privacy.HOME_PATH_PATTERN, "home-directory path") == []
    assert privacy.find_hits(lines, privacy.EMAIL_PATTERN, "email address") == []
    assert privacy.find_hits(lines, privacy.HEX64_PATTERN, "64-hex string") == []
    assert privacy.find_hits(lines, privacy.CERT_PATTERN, "cert-like digit run") == []


def squash(text):
    pieces = text.split()
    return " ".join(pieces)


def test_doc_claims_match_the_data(committed_tiers):
    """The fixture count DECISIONS.md quotes is the committed one."""
    claims = play.doc_claims(committed_tiers)
    assert len(claims) >= 1
    for doc_name, claim in claims:
        assert re.search(r"\d", claim), claim
        assert squash(claim) in squash(page_text(doc_name)), claim
    changed = json.loads(json.dumps(committed_tiers))
    changed["parity"]["fixtures"] += 1
    changed_claims = play.doc_claims(changed)
    for doc_name, claim in changed_claims:
        assert squash(claim) not in squash(page_text(doc_name))
