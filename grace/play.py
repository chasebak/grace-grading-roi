"""The Grading Play Calculator: the engine and python -m grace.play build [--check].

Question: I have this card. Which grading company and service level is the
best play, or should I sell it raw today, once every payment and payout is
valued at the day it actually happens?

Timeline of one service level (days from today, t = 0):
    t = 0        ship the card out and pay the ship-out cost; a company
                 that charges at checkout takes its grading fee now
    t_received   = days_out
    t_graded     = t_received + turnaround business days * 7 / 5; return
                 shipping is paid here, and a company that charges after
                 grading takes its grading fee here
    t_back       = t_graded + days_back
    t_sale       = t_back + sell_days; the expected net sale arrives
Present value of an amount at day t = amount / (1 + weekly_rate) ** (t / 7).
    NPV of grading    = PV(expected net sale) - PV(every cost line)
    sell raw today    = raw_value_now * (1 - fee) - ship_to_buyer
    value of the play = NPV of grading - sell raw today
All dollar results are per card; a batch shares its two shipping costs.

This module is the source of truth. The page runs the same formulas in the
browser (docs/play/calc.js), because a calculator has to answer inputs
nobody precomputed. tests/test_play_parity.py runs calc.js under node on
the fixtures in docs/play/data/parity.json and compares every number with
this module's output.

build writes docs/play/data/tiers.json (the published service tables, the
input defaults and the page's label text) and docs/play/data/parity.json
(fixture inputs with this module's outputs at 10 significant digits).
--check exits 1 if either file is stale or missing, or the page files
break the page kit (pinned Chart.js tag, ../tokens.css, calc.js, a link
each way between the two pages). tiers.json is compared byte for byte.
parity.json is compared value by value at the parity tolerance, because
pow() may differ in its last bit between platforms and a stored 10th digit
could then round the other way on another machine.

Failure modes: the service tables are a dated reading of the companies'
public pages and go stale without notice (the page shows the date and says
so); every odds and price input is the user's own; the parity test proves
the browser formulas equal these on the fixtures, not on every input.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from grace import report, viz

REPO_ROOT = Path(__file__).resolve().parent.parent
TIERS_PATH = "docs/play/data/tiers.json"
PARITY_PATH = "docs/play/data/parity.json"
DATA_PATHS = (TIERS_PATH, PARITY_PATH)
INDEX_PATH = "docs/play/index.html"
APP_PATH = "docs/play/app.js"
CALC_PATH = "docs/play/calc.js"
MAIN_INDEX_PATH = viz.INDEX_PATH
TOKENS_LINK = '<link rel="stylesheet" href="../tokens.css">'
LINK_TO_PLAY = 'href="play/"'
LINK_TO_MAIN = 'href="../"'

TITLE = "Grading Play Calculator"
AS_OF = "2026-11-12"
PROVENANCE = (
    "Service prices and estimated turnaround times were read from the official PSA and "
    "TAG pages on 2026-11-12; they change without notice"
)
FOOTER_TEMPLATE = (
    "This page computes in your browser. Its formulas live in calc.js, and a test runs "
    "them against this repo's tested Python on {fixtures} committed fixture input sets."
)

FEE_AFTER_GRADING = "after_grading"
FEE_AT_CHECKOUT = "at_checkout"
FEE_TIMINGS = (FEE_AT_CHECKOUT, FEE_AFTER_GRADING)
RANGE_FASTEST = "fastest"
RANGE_MIDPOINT = "midpoint"
RANGE_SLOWEST = "slowest"
RANGE_MODES = (RANGE_FASTEST, RANGE_MIDPOINT, RANGE_SLOWEST)
# What a level's value cap does when the card's top-grade price is above it.
CAP_MOVES_UP = "moves_up"  # the company moves the card up a level: not a play at this level
CAP_INFORMATIONAL = "informational"  # still a play; the cover just stops at the cap
FLAG_BELOW_MINIMUM = "below_minimum"
FLAG_OVER_CAP = "over_cap"
PLAY_GRADE = "grade"
PLAY_SELL_RAW = "sell_raw"
RAW_KEY = "raw"

BUSINESS_DAYS_PER_WEEK = 5
CALENDAR_DAYS_PER_WEEK = 7
WEEKS_PER_YEAR = 52
PERCENT = 100.0
# Typed odds like 33.3 + 66.7 can land a hair above 100 in floating point.
ODDS_SLACK_PCT = 1e-9
# Chart 2's grid: the weekly opportunity rate from 0% to 10% in 0.25% steps.
RATE_GRID_STEP_PCT = 0.25
RATE_GRID_STEPS = 40
STOCK_MARKET_ANNUAL_PCT = 12.0
PROJECT_WEEKLY_PCT = 6.0
BGS_EMPTY_ROWS = 4
SIGNIFICANT_DIGITS = report.SIGNIFICANT_DIGITS
PARITY_RELATIVE_TOLERANCE = 1e-9
# Lets an exact zero compare with a rounding-sized remainder (dollars).
PARITY_ABSOLUTE_FLOOR = 1e-9
MIN_FIXTURES = 12
# Grid positions whose chart-2 tooltip ranking each fixture pins: 0%, 6%, 10%.
RANK_SAMPLE_INDEXES = (0, 24, 40)
# Rates the two conversion helpers are pinned at.
ANNUAL_SAMPLE_PCTS = (0.0, 12.0, 50.0)
WEEKLY_SAMPLE_PCTS = (0.0, 0.25, 6.0)

EXAMPLE_LABEL = "example — enter your own comps per company"
ILLUSTRATIVE_LABEL = "illustrative — enter your own"
PROVENANCE_LABELS = {
    "inputs": "every field is yours to change · nothing you type leaves this page",
    "example": EXAMPLE_LABEL,
    "illustrative": ILLUSTRATIVE_LABEL,
    "computed": "computed in your browser from your inputs · formulas tested against the Python",
}
BGS_NOTE = (
    "BGS prices were not read from beckett.com — enter the current ones from "
    "https://www.beckett.com/grading/pricing"
)

FEE_TIMING_LABELS = {
    FEE_AT_CHECKOUT: "up front (day 0)",
    FEE_AFTER_GRADING: "after grading",
}
RANGE_MODE_LABELS = {
    RANGE_FASTEST: "fastest end",
    RANGE_MIDPOINT: "midpoint",
    RANGE_SLOWEST: "slowest end",
}
FLOW_LABELS = {
    "ship_out": "ship the card to the grader",
    "grading_fee": "grading fee",
    "ship_back": "return shipping",
    "sale": "expected net sale",
}
FLAG_TEXT = {
    FLAG_BELOW_MINIMUM: "batch is below this level's minimum of {minimum} cards",
}

PLAY_WHY = {
    "play_bars": {
        "question": (
            "Which company and service level leaves me with the most money in today's "
            "dollars, and does any of them beat selling the card raw today?"
        ),
        "kind": (
            "Modelled: your odds and prices, the published fee and estimated turnaround, "
            "and every payment and payout discounted from the day it happens at your "
            "weekly opportunity rate. Zero is selling raw today."
        ),
        "lie": (
            "the odds or the sale prices are wrong (they are your inputs), or the grader "
            "runs slower than its estimate; a browser formula that drifted from the Python "
            "would also lie (guard: tests/test_play_parity.py)."
        ),
    },
    "rate_lines": {
        "question": (
            "How good does my other use of the money have to be before a faster, dearer "
            "level wins, or before selling raw today wins?"
        ),
        "kind": (
            "Modelled: the same value of the play, recomputed at every weekly rate from "
            "0% to 10% in 0.25% steps. Where two lines cross, the better level changes."
        ),
        "lie": (
            "A crossing between two grid steps is only located to the nearest step, and "
            "one flat rate stands in for a project that may not always be available "
            "(guard: test_rate_grid_matches_single_evaluations in tests/test_play.py)."
        ),
    },
    "results_table": {
        "question": "What is every level's fee, timing and value, including the ones I cannot use?",
        "kind": (
            "Modelled per card: the present value of the costs and of the expected sale, "
            "and their difference against selling raw today. A level you cannot use "
            "(the batch is below its minimum, or the card is over PSA's max insured "
            "value) stays in the table and leaves the charts and the ranking. A level "
            "that is only above TAG's protection amount (or a BGS max value you typed) "
            "stays in the table, the charts and the ranking, with its note."
        ),
        "lie": (
            "A level dropped without a flag would hide a choice (guards: "
            "test_flags_fire_exactly_at_the_boundaries, "
            "test_empty_bgs_rows_never_appear in tests/test_play.py)."
        ),
    },
    "cash_flows": {
        "question": "When does each dollar leave and come back, and what is it worth today?",
        "kind": (
            "Modelled: each dated payment and payout of one level with its sticker "
            "amount and its present value at your weekly rate."
        ),
        "lie": (
            "the company charges at a different step than the setting says; the PSA "
            "default comes from the author's own orders, not a PSA page (guard: "
            "test_fee_paid_later_is_worth_less_today in tests/test_play.py)."
        ),
    },
}


def empty_tier():
    """One blank, user-editable service row (nothing here is a published number)."""
    return {
        "name": "",
        "fee": None,
        "max_value": None,
        "days_min": None,
        "days_max": None,
        "min_cards": None,
    }


def service_tier(name, fee, max_value, days_min, days_max, min_cards):
    return {
        "name": name,
        "fee": fee,
        "max_value": max_value,
        "days_min": days_min,
        "days_max": days_max,
        "min_cards": min_cards,
    }


# The published tables, read from the official pages on AS_OF. These are
# the only service numbers in the repo's calculator; edit them only from a
# fresh reading of the same pages, and move AS_OF when you do.
PSA_TIERS = (
    service_tier("Value", 39.99, 500, 90, 100, None),
    service_tier("Standard", 59.99, 1000, 70, 80, None),
    service_tier("Priority", 79.99, 1500, 50, 60, None),
    service_tier("Express", 199.00, 2500, 20, 30, None),
    service_tier("Super Express", 349.00, 5000, 10, 15, None),
    service_tier("Premier", 599.00, 10000, 7, 10, None),
)
TAG_TIERS = (
    service_tier("Value", 29.95, 250, 90, 90, 10),
    service_tier("Base", 39.95, 500, 60, 60, 5),
    service_tier("Standard", 59.95, 1000, 45, 45, None),
    service_tier("Priority", 79.95, 1500, 30, 30, None),
    service_tier("Express", 149.95, 2500, 10, 10, None),
    service_tier("Premier", 299.95, 5000, 3, 5, None),
)


def service_companies():
    """The three companies' service data as fresh, plain (JSON-ready) dicts.

    BGS ships unverified: blank rows the user fills in, so no BGS number
    exists in this repo. A blank row is never evaluated (tier_is_complete).
    """
    psa_tiers = []
    for tier in PSA_TIERS:
        psa_tiers.append(dict(tier))
    tag_tiers = []
    for tier in TAG_TIERS:
        tag_tiers.append(dict(tier))
    bgs_tiers = []
    blank_rows = range(BGS_EMPTY_ROWS)
    for _row in blank_rows:
        bgs_tiers.append(empty_tier())
    psa = {
        "key": "psa",
        "name": "PSA",
        "verified": True,
        "editable": False,
        "source_url": "https://www.psacard.com/services/tradingcardgrading",
        "as_of": AS_OF,
        "source_quotes": ["Estimated Turnaround Times are estimates only and not guaranteed"],
        "note": "The clock starts when the order is received and scanned in.",
        "cap_label": "max insured value",
        "cap_rule": CAP_MOVES_UP,
        "cap_flag_text": "over this level's max insured value — PSA would move it up a level",
        "fee_timing_default": FEE_AFTER_GRADING,
        "fee_timing_source": (
            "author's own PSA orders (orders wait at an accounting step after grading); "
            "not from a PSA page"
        ),
        "tiers": psa_tiers,
    }
    tag = {
        "key": "tag",
        "name": "TAG",
        "verified": True,
        "editable": False,
        "source_url": "https://taggrading.com/pages/pricing",
        "as_of": AS_OF,
        "source_quotes": [
            "Turnaround times are estimates in business days from intake and are not guaranteed",
            "No value upcharges",
        ],
        "note": "Protection covers each card up to the amount shown.",
        "cap_label": "protection per card",
        "cap_rule": CAP_INFORMATIONAL,
        "cap_flag_text": "above the protection amount",
        "fee_timing_default": FEE_AT_CHECKOUT,
        "fee_timing_source": "TAG's order flow is a store checkout (help.taggrading.com)",
        "tiers": tag_tiers,
    }
    bgs = {
        "key": "bgs",
        "name": "BGS",
        "verified": False,
        "editable": True,
        "source_url": "https://www.beckett.com/grading/pricing",
        "as_of": None,
        "source_quotes": [],
        "note": BGS_NOTE,
        "cap_label": "max value (your entry)",
        "cap_rule": CAP_INFORMATIONAL,
        "cap_flag_text": "above the max value you entered",
        "fee_timing_default": FEE_AT_CHECKOUT,
        "fee_timing_source": "unverified — set it",
        "tiers": bgs_tiers,
    }
    return [psa, tag, bgs]


def example_company_inputs(fee_timing):
    """One company's odds and prices: the same example for every company.

    No company-specific price premium is assumed anywhere; the user enters
    their own comps per company.
    """
    return {
        "p_top_pct": 50.0,
        "p_second_pct": 35.0,
        "price_top": 1500.0,
        "price_second": 500.0,
        "price_lower": 250.0,
        "fee_timing": fee_timing,
    }


def default_inputs():
    """The page's starting inputs (examples and illustrative shipping)."""
    companies = service_companies()
    company_inputs = {}
    for company in companies:
        company_inputs[company["key"]] = example_company_inputs(company["fee_timing_default"])
    return {
        "raw_value_now": 250.0,
        "quantity": 1,
        "sell_fee_pct": 13.0,
        "ship_to_buyer": 5.0,
        "days_out": 5,
        "days_back": 5,
        "ship_out_batch": 15.0,
        "ship_back_batch": 20.0,
        "sell_days": 45,
        "weekly_rate_pct": PROJECT_WEEKLY_PCT,
        "range_mode": RANGE_MIDPOINT,
        "companies": company_inputs,
    }


# ---- the formulas (mirrored one for one in docs/play/calc.js) -------------


def is_number(value):
    """True for a finite int or float; None, bools and text are not numbers."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def tier_day_range(tier):
    """(fastest, slowest) stated business days, or None when neither is given."""
    days_min = tier["days_min"]
    days_max = tier["days_max"]
    has_min = is_number(days_min) and days_min >= 0
    has_max = is_number(days_max) and days_max >= 0
    if has_min and has_max:
        return (min(days_min, days_max), max(days_min, days_max))
    if has_min:
        return (days_min, days_min)
    if has_max:
        return (days_max, days_max)
    return None


def tier_is_complete(tier):
    """A row is usable only with a fee and a turnaround; blank rows are not."""
    fee = tier["fee"]
    if not is_number(fee) or fee < 0:
        return False
    return tier_day_range(tier) is not None


def turnaround_business_days(tier, range_mode):
    """The stated business days at the chosen end of the range."""
    fastest, slowest = tier_day_range(tier)
    if range_mode == RANGE_FASTEST:
        return fastest
    if range_mode == RANGE_SLOWEST:
        return slowest
    return (fastest + slowest) / 2


def business_to_calendar_days(business_days):
    """Business days to calendar days: x 7/5."""
    return business_days * CALENDAR_DAYS_PER_WEEK / BUSINESS_DAYS_PER_WEEK


def present_value(amount, day, weekly_rate):
    """amount / (1 + weekly_rate) ** (day / 7)."""
    weeks = day / CALENDAR_DAYS_PER_WEEK
    return amount / (1 + weekly_rate) ** weeks


def weekly_pct_from_annual(annual_pct):
    """The weekly rate that compounds to annual_pct over 52 weeks."""
    growth = 1 + annual_pct / PERCENT
    weekly_growth = growth ** (1 / WEEKS_PER_YEAR)
    return (weekly_growth - 1) * PERCENT


def annual_pct_from_weekly(weekly_pct):
    """What weekly_pct compounds to over 52 weeks."""
    growth = (1 + weekly_pct / PERCENT) ** WEEKS_PER_YEAR
    return (growth - 1) * PERCENT


def clean_quantity(inputs):
    """Cards in the batch: at least one."""
    quantity = inputs["quantity"]
    if not is_number(quantity) or quantity < 1:
        return 1
    return quantity


def odds_are_valid(company_inputs):
    """Top and second odds are not negative and add up to at most 100%."""
    p_top_pct = company_inputs["p_top_pct"]
    p_second_pct = company_inputs["p_second_pct"]
    if p_top_pct < 0 or p_second_pct < 0:
        return False
    return p_top_pct + p_second_pct <= PERCENT + ODDS_SLACK_PCT


def lower_odds_pct(company_inputs):
    """The remainder after the top and second grades."""
    lower_pct = PERCENT - company_inputs["p_top_pct"] - company_inputs["p_second_pct"]
    if lower_pct < 0:
        return 0.0
    return lower_pct


def net_sale(price, sell_fee, ship_to_buyer):
    """What one sale leaves after the selling fee and shipping to the buyer."""
    return price * (1 - sell_fee) - ship_to_buyer


def expected_net_sale(company_inputs, sell_fee, ship_to_buyer):
    """Sum over the three outcomes of probability x net sale."""
    p_top = company_inputs["p_top_pct"] / PERCENT
    p_second = company_inputs["p_second_pct"] / PERCENT
    p_lower = lower_odds_pct(company_inputs) / PERCENT
    net_top = net_sale(company_inputs["price_top"], sell_fee, ship_to_buyer)
    net_second = net_sale(company_inputs["price_second"], sell_fee, ship_to_buyer)
    net_lower = net_sale(company_inputs["price_lower"], sell_fee, ship_to_buyer)
    return p_top * net_top + p_second * net_second + p_lower * net_lower


def sell_raw_today(inputs):
    """raw_value_now x (1 - fee) - ship_to_buyer: the money in hand today."""
    sell_fee = inputs["sell_fee_pct"] / PERCENT
    return net_sale(inputs["raw_value_now"], sell_fee, inputs["ship_to_buyer"])


def tier_plan(company, tier, inputs):
    """One complete tier's dated sticker cash flows, before any discounting."""
    company_inputs = inputs["companies"][company["key"]]
    sell_fee = inputs["sell_fee_pct"] / PERCENT
    quantity = clean_quantity(inputs)
    business_days = turnaround_business_days(tier, inputs["range_mode"])
    calendar_days = business_to_calendar_days(business_days)
    day_received = inputs["days_out"]
    day_graded = day_received + calendar_days
    day_back = day_graded + inputs["days_back"]
    day_sale = day_back + inputs["sell_days"]
    fee_timing = company_inputs["fee_timing"]
    fee_day = day_graded
    if fee_timing == FEE_AT_CHECKOUT:
        fee_day = 0
    ship_out = inputs["ship_out_batch"] / quantity
    ship_back = inputs["ship_back_batch"] / quantity
    sale = expected_net_sale(company_inputs, sell_fee, inputs["ship_to_buyer"])
    flows = [
        {"what": "ship_out", "kind": "cost", "day": 0, "amount": ship_out},
        {"what": "grading_fee", "kind": "cost", "day": fee_day, "amount": tier["fee"]},
        {"what": "ship_back", "kind": "cost", "day": day_graded, "amount": ship_back},
        {"what": "sale", "kind": "proceeds", "day": day_sale, "amount": sale},
    ]
    return {
        "fee_timing": fee_timing,
        "turnaround_business_days": business_days,
        "turnaround_calendar_days": calendar_days,
        "day_received": day_received,
        "day_graded": day_graded,
        "day_back": day_back,
        "day_sale": day_sale,
        "flows": flows,
    }


def value_flows(flows, weekly_rate):
    """Each flow's value today, and the totals by kind (costs added in order)."""
    valued = []
    pv_costs = 0.0
    pv_proceeds = 0.0
    sticker_costs = 0.0
    sticker_proceeds = 0.0
    for flow in flows:
        value_today = present_value(flow["amount"], flow["day"], weekly_rate)
        entry = {
            "what": flow["what"],
            "kind": flow["kind"],
            "day": flow["day"],
            "amount": flow["amount"],
            "value_today": value_today,
        }
        valued.append(entry)
        if flow["kind"] == "cost":
            pv_costs = pv_costs + value_today
            sticker_costs = sticker_costs + flow["amount"]
        else:
            pv_proceeds = pv_proceeds + value_today
            sticker_proceeds = sticker_proceeds + flow["amount"]
    return {
        "flows": valued,
        "pv_costs": pv_costs,
        "pv_proceeds": pv_proceeds,
        "sticker_costs": sticker_costs,
        "sticker_proceeds": sticker_proceeds,
    }


def tier_flags(company, tier, inputs):
    """Eligibility of one tier for this batch and this card's top-grade price.

    below_minimum: the batch has fewer cards than the level's minimum.
    over_cap: the top-grade price is above the level's value cap. With a
    "moves up" cap (PSA's max insured value) the level is not a play; with
    an informational cap (TAG's protection amount) it still is.
    """
    company_inputs = inputs["companies"][company["key"]]
    quantity = clean_quantity(inputs)
    minimum = tier["min_cards"]
    below_minimum = is_number(minimum) and quantity < minimum
    cap = tier["max_value"]
    over_cap = is_number(cap) and company_inputs["price_top"] > cap
    flags = []
    if below_minimum:
        flags.append(FLAG_BELOW_MINIMUM)
    if over_cap:
        flags.append(FLAG_OVER_CAP)
    blocked_by_cap = over_cap and company["cap_rule"] == CAP_MOVES_UP
    eligible = not below_minimum and not blocked_by_cap
    return {
        "below_minimum": below_minimum,
        "over_cap": over_cap,
        "flags": flags,
        "eligible": eligible,
    }


def tier_key(company, tier_index):
    return company["key"] + "/" + str(tier_index)


def tier_label(company, tier, tier_index):
    name = tier["name"]
    if not name:
        name = "row " + str(tier_index + 1)
    return company["name"] + " " + name


def evaluate_tier(company, tier_index, inputs):
    """Everything the page shows for one complete tier, per card."""
    tier = company["tiers"][tier_index]
    weekly_rate = inputs["weekly_rate_pct"] / PERCENT
    plan = tier_plan(company, tier, inputs)
    valued = value_flows(plan["flows"], weekly_rate)
    raw_today = sell_raw_today(inputs)
    npv = valued["pv_proceeds"] - valued["pv_costs"]
    undiscounted_profit = valued["sticker_proceeds"] - valued["sticker_costs"]
    fee_flow = valued["flows"][1]
    flags = tier_flags(company, tier, inputs)
    return {
        "key": tier_key(company, tier_index),
        "company": company["key"],
        "label": tier_label(company, tier, tier_index),
        "fee": tier["fee"],
        "fee_timing": plan["fee_timing"],
        "fee_day": fee_flow["day"],
        "fee_value_today": fee_flow["value_today"],
        "turnaround_business_days": plan["turnaround_business_days"],
        "turnaround_calendar_days": plan["turnaround_calendar_days"],
        "day_received": plan["day_received"],
        "day_graded": plan["day_graded"],
        "day_back": plan["day_back"],
        "day_sale": plan["day_sale"],
        "expected_net_sale": valued["sticker_proceeds"],
        "sticker_costs": valued["sticker_costs"],
        "pv_costs": valued["pv_costs"],
        "pv_proceeds": valued["pv_proceeds"],
        "npv": npv,
        "value_of_play": npv - raw_today,
        "undiscounted_profit": undiscounted_profit,
        "undiscounted_value_of_play": undiscounted_profit - raw_today,
        "min_cards": tier["min_cards"],
        "max_value": tier["max_value"],
        "below_minimum": flags["below_minimum"],
        "over_cap": flags["over_cap"],
        "flags": flags["flags"],
        "eligible": flags["eligible"],
        "gap_to_best": None,
        "flows": valued["flows"],
    }


def value_of_play_at_rate(company, tier_index, inputs, weekly_rate):
    """One tier's value of the play at another weekly rate (same plan)."""
    tier = company["tiers"][tier_index]
    plan = tier_plan(company, tier, inputs)
    valued = value_flows(plan["flows"], weekly_rate)
    npv = valued["pv_proceeds"] - valued["pv_costs"]
    return npv - sell_raw_today(inputs)


def rate_sort_key(entry):
    return (-entry["value_of_play"], entry["position"])


def rank_values(keys, values):
    """[{key, value_of_play, gap_to_best}] best first; ties keep table order."""
    entries = []
    key_count = len(keys)
    positions = range(key_count)
    for position in positions:
        entry = {"key": keys[position], "value_of_play": values[position], "position": position}
        entries.append(entry)
    ordered = sorted(entries, key=rate_sort_key)
    ranking = []
    for entry in ordered:
        gap = entry["value_of_play"] - ordered[0]["value_of_play"]
        ranking.append(
            {"key": entry["key"], "value_of_play": entry["value_of_play"], "gap_to_best": gap}
        )
    return ranking


def rate_grid_pcts():
    """0%, 0.25%, ..., 10% a week."""
    pcts = []
    steps = range(RATE_GRID_STEPS + 1)
    for step in steps:
        pcts.append(step * RATE_GRID_STEP_PCT)
    return pcts


def rank_at_rate(rate_grid, grid_index):
    """The eligible tiers ranked at one grid rate (chart 2's tooltip)."""
    keys = rate_grid["keys"]
    values = []
    for key in keys:
        values.append(rate_grid["value_of_play"][key][grid_index])
    return rank_values(keys, values)


def best_by_rate(rate_grid):
    """Runs of grid rates with one winner: a tier key, or "raw" when no
    eligible tier beats selling raw today at that rate."""
    rates = rate_grid["rates_pct"]
    keys = rate_grid["keys"]
    segments = []
    rate_count = len(rates)
    positions = range(rate_count)
    for grid_index in positions:
        winner = RAW_KEY
        best_value = 0.0
        for key in keys:
            value = rate_grid["value_of_play"][key][grid_index]
            if value > best_value:
                winner = key
                best_value = value
        if segments and segments[-1]["winner"] == winner:
            segments[-1]["to_pct"] = rates[grid_index]
        else:
            segments.append(
                {"winner": winner, "from_pct": rates[grid_index], "to_pct": rates[grid_index]}
            )
    return segments


def evaluate(inputs, companies):
    """Every complete tier of every company for one card, plus the ranking.

    A blank or half-filled row is skipped (tier_is_complete). A company
    whose odds add up to more than 100% is listed in skipped_companies
    instead of being evaluated on impossible odds. Flagged tiers stay in
    "tiers" with their flags; only eligible tiers enter the ranking, the
    rate grid and best_by_rate.
    """
    raw_today = sell_raw_today(inputs)
    tiers = []
    skipped_companies = []
    eligible_keys = []
    eligible_values = []
    eligible_sources = []
    for company in companies:
        company_inputs = inputs["companies"][company["key"]]
        valid_odds = odds_are_valid(company_inputs)
        tier_rows = company["tiers"]
        tier_count = len(tier_rows)
        positions = range(tier_count)
        complete_count = 0
        for tier_index in positions:
            if not tier_is_complete(tier_rows[tier_index]):
                continue
            complete_count += 1
            if not valid_odds:
                continue
            result = evaluate_tier(company, tier_index, inputs)
            tiers.append(result)
            if result["eligible"]:
                eligible_keys.append(result["key"])
                eligible_values.append(result["value_of_play"])
                eligible_sources.append((company, tier_index))
        if complete_count > 0 and not valid_odds:
            skipped_companies.append(company["key"])

    ranking = rank_values(eligible_keys, eligible_values)
    gap_by_key = {}
    for entry in ranking:
        gap_by_key[entry["key"]] = entry["gap_to_best"]
    for result in tiers:
        if result["key"] in gap_by_key:
            result["gap_to_best"] = gap_by_key[result["key"]]

    quantity = clean_quantity(inputs)
    best = None
    best_value = None
    best_batch_value = None
    play = PLAY_SELL_RAW
    if ranking:
        best = ranking[0]["key"]
        best_value = ranking[0]["value_of_play"]
        best_batch_value = best_value * quantity
        if best_value > 0:
            play = PLAY_GRADE

    rates_pct = rate_grid_pcts()
    curves = {}
    eligible_count = len(eligible_keys)
    eligible_positions = range(eligible_count)
    for position in eligible_positions:
        company, tier_index = eligible_sources[position]
        curve = []
        for rate_pct in rates_pct:
            weekly_rate = rate_pct / PERCENT
            curve.append(value_of_play_at_rate(company, tier_index, inputs, weekly_rate))
        curves[eligible_keys[position]] = curve
    rate_grid = {"rates_pct": rates_pct, "keys": eligible_keys, "value_of_play": curves}

    return {
        "weekly_rate_pct": inputs["weekly_rate_pct"],
        "annual_rate_pct": annual_pct_from_weekly(inputs["weekly_rate_pct"]),
        "quantity": quantity,
        "sell_raw_today": raw_today,
        "tiers": tiers,
        "skipped_companies": skipped_companies,
        "ranking": ranking,
        "best": best,
        "best_value_of_play": best_value,
        "best_batch_value_of_play": best_batch_value,
        "play": play,
        "rate_grid": rate_grid,
        "best_by_rate": best_by_rate(rate_grid),
    }


# ---- the page data ---------------------------------------------------------


def rate_presets():
    """The two preset buttons under the opportunity rate."""
    return [
        {"label": "another project at 6% a week", "weekly_rate_pct": PROJECT_WEEKLY_PCT},
        {
            "label": "stock market, 12% a year",
            "weekly_rate_pct": weekly_pct_from_annual(STOCK_MARKET_ANNUAL_PCT),
        },
    ]


def build_tiers_bundle(fixture_count):
    """docs/play/data/tiers.json as a plain dict (not yet rounded)."""
    return {
        "generated_by": "python -m grace.play build",
        "title": TITLE,
        "as_of": AS_OF,
        "provenance": PROVENANCE,
        "provenance_labels": PROVENANCE_LABELS,
        "footer": FOOTER_TEMPLATE.format(fixtures=fixture_count),
        "companies": service_companies(),
        "defaults": default_inputs(),
        "example_label": EXAMPLE_LABEL,
        "illustrative_label": ILLUSTRATIVE_LABEL,
        "fee_timings": list(FEE_TIMINGS),
        "fee_timing_labels": FEE_TIMING_LABELS,
        "range_modes": list(RANGE_MODES),
        "range_mode_labels": RANGE_MODE_LABELS,
        "flow_labels": FLOW_LABELS,
        "flag_text": FLAG_TEXT,
        "presets": rate_presets(),
        "rate_grid": {
            "min_pct": 0.0,
            "max_pct": RATE_GRID_STEPS * RATE_GRID_STEP_PCT,
            "step_pct": RATE_GRID_STEP_PCT,
        },
        "why": PLAY_WHY,
        "parity": {
            "fixtures": fixture_count,
            "relative_tolerance": PARITY_RELATIVE_TOLERANCE,
            "significant_digits": SIGNIFICANT_DIGITS,
        },
    }


def fixture(name, what, inputs, companies=None):
    """One parity fixture. companies=None means the shipped service data."""
    return {"name": name, "what": what, "inputs": inputs, "companies": companies}


def twin_companies():
    """Two made-up companies with identical tiers (a symmetry fixture).

    The fee and days are round test values, not any company's prices. The
    second row has no name, so the fixtures still cover the "row N" label.
    """
    twins = []
    for key in ("north", "south"):
        twins.append(
            {
                "key": key,
                "name": key.capitalize(),
                "cap_rule": CAP_INFORMATIONAL,
                "tiers": [
                    service_tier("Slow", 20.0, 1000, 40, 60, None),
                    service_tier("", 100.0, None, 5, 5, 2),
                ],
            }
        )
    return twins


def build_fixtures():
    """Fixture inputs for the Python/JavaScript parity test (no outputs yet).

    Covers: all three companies, both fee timings on both verified
    companies, rate 0, a rate above the chart grid, quantities below, at and
    above a minimum, a top price above and exactly at a cap, each range
    mode, typed-in BGS rows (one left half blank), impossible odds, and two
    identical made-up companies.
    """
    fixtures = []
    fixtures.append(fixture("defaults", "the page's starting inputs", default_inputs()))

    inputs = default_inputs()
    inputs["weekly_rate_pct"] = 0.0
    fixtures.append(fixture("rate_zero", "no discounting: NPV is the plain sum", inputs))

    inputs = default_inputs()
    inputs["weekly_rate_pct"] = 25.0
    fixtures.append(fixture("rate_high", "25% a week, above the chart's grid", inputs))

    inputs = default_inputs()
    inputs["weekly_rate_pct"] = weekly_pct_from_annual(STOCK_MARKET_ANNUAL_PCT)
    fixtures.append(fixture("rate_stock_market", "the 12% a year preset", inputs))

    inputs = default_inputs()
    inputs["range_mode"] = RANGE_FASTEST
    fixtures.append(fixture("range_fastest", "fastest end of each stated range", inputs))

    inputs = default_inputs()
    inputs["range_mode"] = RANGE_SLOWEST
    fixtures.append(fixture("range_slowest", "slowest end of each stated range", inputs))

    inputs = default_inputs()
    inputs["quantity"] = 4
    fixtures.append(fixture("quantity_below_minimums", "4 cards: under both TAG minimums", inputs))

    inputs = default_inputs()
    inputs["quantity"] = 5
    fixtures.append(fixture("quantity_at_a_minimum", "5 cards: exactly TAG Base's minimum", inputs))

    inputs = default_inputs()
    inputs["quantity"] = 10
    inputs["weekly_rate_pct"] = 1.0
    fixtures.append(fixture("quantity_ten", "10 cards share the batch shipping", inputs))

    inputs = default_inputs()
    company_keys = sorted(inputs["companies"].keys())
    for key in company_keys:
        inputs["companies"][key]["price_top"] = 6000.0
    inputs["raw_value_now"] = 900.0
    fixtures.append(fixture("price_above_caps", "a $6,000 top grade: over most caps", inputs))

    inputs = default_inputs()
    for key in company_keys:
        inputs["companies"][key]["price_top"] = 500.0
        inputs["companies"][key]["price_second"] = 200.0
        inputs["companies"][key]["price_lower"] = 90.0
    inputs["raw_value_now"] = 100.0
    inputs["weekly_rate_pct"] = 0.5
    fixtures.append(fixture("price_at_a_cap", "a $500 top grade: exactly at two caps", inputs))

    inputs = default_inputs()
    inputs["companies"]["psa"]["fee_timing"] = FEE_AT_CHECKOUT
    inputs["companies"]["tag"]["fee_timing"] = FEE_AFTER_GRADING
    fixtures.append(fixture("fee_timings_swapped", "PSA up front, TAG after grading", inputs))

    inputs = default_inputs()
    inputs["companies"]["psa"]["p_top_pct"] = 30.0
    inputs["companies"]["psa"]["price_top"] = 2800.0
    inputs["companies"]["tag"]["p_top_pct"] = 55.0
    inputs["companies"]["tag"]["p_second_pct"] = 33.3
    inputs["companies"]["tag"]["price_top"] = 1900.0
    inputs["companies"]["tag"]["price_second"] = 640.0
    inputs["sell_days"] = 21
    inputs["weekly_rate_pct"] = 2.5
    fixtures.append(fixture("mixed_companies", "different odds and comps per company", inputs))

    inputs = default_inputs()
    inputs["companies"]["bgs"]["fee_timing"] = FEE_AFTER_GRADING
    inputs["weekly_rate_pct"] = 1.5
    typed = service_companies()
    typed[2]["tiers"][0] = service_tier("made-up row A", 25.0, 400, 30, 40, 3)
    typed[2]["tiers"][1] = service_tier("made-up row B", 150.0, None, None, 8, None)
    typed[2]["tiers"][2] = service_tier("made-up row C", 75.0, None, None, None, None)
    fixtures.append(
        fixture("bgs_typed_in", "made-up test rows typed into BGS; one half blank", inputs, typed)
    )

    inputs = default_inputs()
    inputs["companies"]["tag"]["p_top_pct"] = 70.0
    inputs["companies"]["tag"]["p_second_pct"] = 45.0
    fixtures.append(fixture("impossible_odds", "TAG's odds add up to 115%: skipped", inputs))

    inputs = default_inputs()
    inputs["sell_fee_pct"] = 0.0
    inputs["ship_to_buyer"] = 0.0
    inputs["ship_out_batch"] = 0.0
    inputs["ship_back_batch"] = 0.0
    inputs["days_out"] = 0
    inputs["days_back"] = 0
    inputs["sell_days"] = 0
    inputs["weekly_rate_pct"] = 3.0
    fixtures.append(fixture("no_frictions", "no fees or shipping; sold the day it is back", inputs))

    twins = twin_companies()
    inputs = default_inputs()
    inputs["companies"] = {
        "north": example_company_inputs(FEE_AFTER_GRADING),
        "south": example_company_inputs(FEE_AFTER_GRADING),
    }
    inputs["quantity"] = 2
    inputs["weekly_rate_pct"] = 4.0
    fixtures.append(
        fixture("twin_companies", "two made-up companies with identical tiers", inputs, twins)
    )
    return fixtures


def fixture_companies(entry):
    """The service data a fixture runs on."""
    if entry["companies"] is None:
        return service_companies()
    return entry["companies"]


def build_parity_bundle():
    """docs/play/data/parity.json as a plain dict (not yet rounded)."""
    fixtures = build_fixtures()
    entries = []
    for entry in fixtures:
        expected = evaluate(entry["inputs"], fixture_companies(entry))
        rankings = {}
        for grid_index in RANK_SAMPLE_INDEXES:
            rankings[grid_index] = rank_at_rate(expected["rate_grid"], grid_index)
        entries.append(
            {
                "name": entry["name"],
                "what": entry["what"],
                "inputs": entry["inputs"],
                "companies": entry["companies"],
                "expected": expected,
                "expected_rank_at_rate": rankings,
            }
        )
    weekly_from_annual = []
    for annual_pct in ANNUAL_SAMPLE_PCTS:
        weekly_from_annual.append(
            {"annual_pct": annual_pct, "expected": weekly_pct_from_annual(annual_pct)}
        )
    annual_from_weekly = []
    for weekly_pct in WEEKLY_SAMPLE_PCTS:
        annual_from_weekly.append(
            {"weekly_pct": weekly_pct, "expected": annual_pct_from_weekly(weekly_pct)}
        )
    return {
        "generated_by": "python -m grace.play build",
        "significant_digits": SIGNIFICANT_DIGITS,
        "relative_tolerance": PARITY_RELATIVE_TOLERANCE,
        "fixtures": entries,
        "helpers": {
            "weekly_pct_from_annual": weekly_from_annual,
            "annual_pct_from_weekly": annual_from_weekly,
        },
    }


def build_outputs():
    """{repo-relative path: file text} for the two data files."""
    parity = build_parity_bundle()
    tiers = build_tiers_bundle(len(parity["fixtures"]))
    return {
        TIERS_PATH: report.to_json_text(report.round_floats(tiers)),
        PARITY_PATH: report.to_json_text(report.round_floats(parity)),
    }


def is_plain_number(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def json_differences(expected, actual, where, expected_name, actual_name):
    """Every place two JSON values disagree, as 'path: what' lines.

    Rule: numbers must agree to PARITY_RELATIVE_TOLERANCE (with the small
    absolute floor); text, booleans, nulls, list lengths and dict keys must
    match exactly. Used for the Python/JavaScript parity test and for
    --check's comparison of parity.json.
    """
    if is_plain_number(expected) and is_plain_number(actual):
        close = math.isclose(
            expected,
            actual,
            rel_tol=PARITY_RELATIVE_TOLERANCE,
            abs_tol=PARITY_ABSOLUTE_FLOOR,
        )
        if close:
            return []
        return [f"{where}: {expected_name} {expected!r} != {actual_name} {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = sorted(expected.keys())
        actual_keys = sorted(actual.keys())
        if expected_keys != actual_keys:
            return [
                f"{where}: keys differ: {expected_name} {expected_keys} != "
                f"{actual_name} {actual_keys}"
            ]
        problems = []
        for key in expected_keys:
            inner_where = f"{where}.{key}"
            problems += json_differences(
                expected[key], actual[key], inner_where, expected_name, actual_name
            )
        return problems
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [
                f"{where}: length {expected_name} {len(expected)} != "
                f"{actual_name} {len(actual)}"
            ]
        problems = []
        item_count = len(expected)
        positions = range(item_count)
        for position in positions:
            inner_where = f"{where}[{position}]"
            problems += json_differences(
                expected[position], actual[position], inner_where, expected_name, actual_name
            )
        return problems
    same_type = type(expected) is type(actual)
    if same_type and expected == actual:
        return []
    return [f"{where}: {expected_name} {expected!r} != {actual_name} {actual!r}"]


def bundle_problems(tiers, parity):
    """Shape problems in the page data (empty when clean).

    Rule: an unverified company ships no service number at all; every
    verified tier is complete; there are at least MIN_FIXTURES fixtures.
    Failure mode: it checks shape, not whether a published price is still
    current; only a fresh reading of the official pages can tell that.
    """
    problems = []
    for company in tiers["companies"]:
        for tier in company["tiers"]:
            complete = tier_is_complete(tier)
            if company["verified"] and not complete:
                problems.append(f"{company['key']}: a verified tier is incomplete")
            if company["verified"]:
                continue
            field_names = sorted(tier.keys())
            for field_name in field_names:
                if is_number(tier[field_name]):
                    problems.append(f"{company['key']}: an unverified tier carries a number")
    fixture_count = len(parity["fixtures"])
    if fixture_count < MIN_FIXTURES:
        problems.append(f"parity.json holds {fixture_count} fixtures, under {MIN_FIXTURES}")
    return problems


def static_problems(root):
    """Problems with the hand-written page files the build does not write."""
    root_path = Path(root)
    problems = []
    for path in (INDEX_PATH, APP_PATH, CALC_PATH, MAIN_INDEX_PATH):
        if not (root_path / path).is_file():
            problems.append(f"{path} is missing")
    if problems:
        return problems
    index = (root_path / INDEX_PATH).read_text(encoding="utf-8")
    if index.count(viz.CHART_JS_TAG) != 1:
        problems.append(f"{INDEX_PATH} must carry the pinned Chart.js tag exactly once")
    if index.count(TOKENS_LINK) != 1:
        problems.append(f"{INDEX_PATH} must link ../tokens.css exactly once")
    if LINK_TO_MAIN not in index:
        problems.append(f"{INDEX_PATH} must link back to the main page")
    main_index = (root_path / MAIN_INDEX_PATH).read_text(encoding="utf-8")
    if LINK_TO_PLAY not in main_index:
        problems.append(f"{MAIN_INDEX_PATH} must link to the calculator")
    return problems


def check(root=REPO_ROOT, outputs=None):
    """Every problem --check reports (empty when the page data are fresh)."""
    if outputs is None:
        outputs = build_outputs()
    root_path = Path(root)
    problems = []
    paths = sorted(outputs.keys())
    for path in paths:
        target = root_path / path
        if not target.is_file():
            problems.append(f"{path} is missing; run python -m grace.play build")
            continue
        committed = target.read_text(encoding="utf-8")
        if committed == outputs[path]:
            continue
        stale = True
        if path == PARITY_PATH:
            fresh_value = json.loads(outputs[path])
            committed_value = json.loads(committed)
            differences = json_differences(fresh_value, committed_value, "parity", "fresh", "file")
            stale = differences != []
        if stale:
            problems.append(f"{path} is stale; run python -m grace.play build")
    tiers = json.loads(outputs[TIERS_PATH])
    parity = json.loads(outputs[PARITY_PATH])
    problems += bundle_problems(tiers, parity)
    problems += static_problems(root)
    return problems


def write(root=REPO_ROOT):
    outputs = build_outputs()
    root_path = Path(root)
    (root_path / "docs" / "play" / "data").mkdir(parents=True, exist_ok=True)
    paths = sorted(outputs.keys())
    for path in paths:
        (root_path / path).write_text(outputs[path], encoding="utf-8")
    return paths


def doc_claims(tiers):
    """(doc, substring) pairs for the calculator numbers the docs quote.

    tests/test_play.py checks each appears verbatim, so a changed fixture
    set that leaves DECISIONS.md behind fails CI.
    """
    parity = tiers["parity"]
    claims = [
        (
            "DECISIONS.md",
            (
                f"{parity['fixtures']} fixture input sets agree to "
                f"{parity['relative_tolerance']:g} relative"
            ),
        ),
    ]
    return claims


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"], help="build the calculator's data")
    parser.add_argument("--check", action="store_true", help="fail if the data are stale")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.root)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print("calculator data are up to date")
        return 0
    paths = write(args.root)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
