"""Seeded synthetic data: pop ladders plus price tables with planted pathologies.

Everything here is synthetic; column names are generic (no vendor or
marketplace names). The truth table records what was planted so the gate
ladder can be scored as a classifier.

Planted pathologies (label column in truth.csv):
  wrong_card    feed V10 inflated versus the graded-10 sold median
                (a bad identity match); inflation factor 2.2x-9x, so some
                plants sit below the 3x gate threshold on purpose
  fake_grail    V10 drawn in $700-$6,000 with zero graded-sold
                corroboration and <=2 feed sales; the sub-$1,000 part of
                the range is deliberately below the gate threshold
  thin_outlier  no V9, <=1 lifetime feed sale, V10 in $120-$900 (the
                sub-$200 part is deliberately below the gate threshold)
  bad_raw       raw price divided by 2x-150x; ~20% of divisors are drawn
                sub-threshold in 1.5x-2.8x, deliberately too mild for the
                repair rules to see (the plants straddle the gate)
  genuine_grail a REAL high-multiplier card (80x-300x) WITH graded-sold
                corroboration — the gates must keep every one of these

Borderline-legit clean subpopulation (label stays "clean",
should_flag=False, detail starts with "borderline_clean"): legitimate
cards whose honest numbers approach the gate thresholds, so gate
precision is measured against something that can actually be lost:
  hot_market      the graded-10 sold median lags a hot feed V10
                  (v10 = median * 2.0x-4.5x, n>=3): the part above 3x
                  legitimately trips the wrong-card gate
  sparse_release  a new release: no V9, 0-1 lifetime feed sales, V10 in
                  $150-$900, no graded-sold history yet — most of these
                  sit above the $200 thin-outlier gate
  ratio_tail      an honest V10/V9 ratio drawn in 40-75 against the 60
                  ceiling
  exempt_grail    a genuine 80x-120x multiplier WITH graded-sold
                  corroboration: exercises the corroboration exemption
                  from the legit side

Submission-selection effect: the observed gem rate used to draw each pop
ladder is the latent true gem rate plus a boost that shrinks with card
value — submitters pre-screen low-value cards harder, so observed gem
rates overstate P(10) for an unscreened raw card, most strongly at the
low end.

Knobs (generate() parameters; defaults reproduce synth/*.csv byte for byte,
which tests/test_golden_data.py pins):
  selection_boost_max  size of the submission-selection boost at V10 = 0
                       (0.0 is the null control: observed = latent rates)
  multiplier_scale     multiplies every clean card's drawn V10/raw
                       multiplier before the [1.5, 55] clip (below 1.0 is a
                       harsher market where grading pays less)

True prices. generate_world() also returns each card's TRUE prices, for
the decision backtest's truth model. They are plain copies of values the
generator already drew, so capturing them adds no random draws:
  clean (incl. borderline), genuine_grail  true = feed (the plant is real)
  wrong_card, fake_grail, thin_outlier,    true = the base card's prices
  bad_raw                                  before the plant overwrote them
                                           (a junk feed row replaced a real
                                           card; the card you hold is that
                                           base card)
Failure mode: a junk plant whose base card had no V9 has true_v9 = NaN;
the truth model then uses the engine's 0.5*V10 fallback, so V9 truth for
those cards is only as good as that fallback.
"""

import math
import os

import numpy as np
import pandas as pd

DEFAULT_SEED = 7
DEFAULT_N_CARDS = 600

ERA_NAMES = ("modern", "mid", "vintage")
ERA_SHARES = (0.5, 0.3, 0.2)

# Latent (true) gem-rate distribution per era: Beta with this mean and
# concentration (alpha + beta). Deliberately BELOW the engine's
# illustrative priors, because the priors are fit to observed (selection-
# boosted) pop ladders.
ERA_LATENT_MEAN = {"modern": 0.40, "mid": 0.25, "vintage": 0.14}
ERA_LATENT_CONCENTRATION = {"modern": 30.0, "mid": 30.0, "vintage": 30.0}

ERA_RAW_MEDIAN = {"modern": 18.0, "mid": 30.0, "vintage": 60.0}
ERA_RAW_SIGMA = {"modern": 0.9, "mid": 0.9, "vintage": 1.0}
ERA_MULT_MEDIAN = {"modern": 9.0, "mid": 8.0, "vintage": 7.0}
ERA_MULT_SIGMA = {"modern": 0.7, "mid": 0.7, "vintage": 0.7}

SELECTION_BOOST_MAX = 0.18
SELECTION_BOOST_SCALE = 400.0  # dollars of V10 over which the boost decays
MULTIPLIER_SCALE = 1.0  # default knob: the multiplier draws as fitted above
MULTIPLIER_CLIP = (1.5, 55.0)

# Truth labels whose feed prices are junk: the true prices are the base
# card's, captured before the plant.
JUNK_FEED_LABELS = ("wrong_card", "fake_grail", "thin_outlier", "bad_raw")

PATHOLOGY_COUNTS = {
    "wrong_card": 25,
    "fake_grail": 15,
    "thin_outlier": 15,
    "bad_raw": 30,
    "genuine_grail": 15,
}

# Borderline-legit clean cards (truth label stays "clean"). Sized so the
# default 600-card draw carries ~68 of them among its 500 clean cards.
BORDERLINE_COUNTS = {
    "hot_market": 18,
    "sparse_release": 16,
    "ratio_tail": 18,
    "exempt_grail": 16,
}

BAD_RAW_SUBTHRESHOLD_FRACTION = 0.2

# Share of the non-gem copies in a ladder that grade 9 (the rest grade <=8).
# The decision backtest's truth model reuses this exact value.
NINE_SHARE = 0.6


def selection_boost(v10, boost_max=SELECTION_BOOST_MAX):
    """Submission-selection inflation of the observed gem rate.

    boost = boost_max * exp(-V10 / SELECTION_BOOST_SCALE); boost_max = 0
    turns the effect off (the null control).
    """
    return boost_max * math.exp(-v10 / SELECTION_BOOST_SCALE)


def _draw_base_card(rng, card_id, era, boost_max, multiplier_scale):
    """One clean card as a plain dict.

    The knobs change values, not which draws are made. They do NOT keep a
    knob run aligned card by card with the default world: NumPy's binomial
    sampler uses a value-dependent number of uniforms, so once a ladder's
    gem rate moves, every later draw (and the plant assignment) shifts. A
    knob run is its own seeded world.
    """
    latent_mean = ERA_LATENT_MEAN[era]
    concentration = ERA_LATENT_CONCENTRATION[era]
    alpha = latent_mean * concentration
    beta = (1.0 - latent_mean) * concentration
    latent_p10 = rng.beta(alpha, beta)

    raw_price = rng.lognormal(math.log(ERA_RAW_MEDIAN[era]), ERA_RAW_SIGMA[era])
    multiplier = rng.lognormal(math.log(ERA_MULT_MEDIAN[era]), ERA_MULT_SIGMA[era])
    multiplier = multiplier * multiplier_scale
    multiplier = min(max(multiplier, MULTIPLIER_CLIP[0]), MULTIPLIER_CLIP[1])
    v10 = raw_price * multiplier

    v9 = v10 * rng.uniform(0.3, 0.6)
    if rng.uniform() < 0.12:
        v9 = math.nan

    near_mint_price = raw_price * rng.uniform(0.9, 1.3)
    if rng.uniform() < 0.4:
        near_mint_price = math.nan

    grade10_sold_median = v10 * rng.uniform(0.85, 1.15)
    grade10_sold_n = int(rng.poisson(5))
    if grade10_sold_n == 0:
        grade10_sold_median = math.nan
    ungraded_sold_median = raw_price * rng.uniform(0.85, 1.2)
    ungraded_sold_n = int(rng.poisson(6))
    if ungraded_sold_n == 0:
        ungraded_sold_median = math.nan

    feed_sales_lifetime = int(rng.poisson(9))
    grade10_sales_180d = int(rng.poisson(4))

    n_ladder = round(rng.lognormal(math.log(50.0), 1.0))
    if rng.uniform() < 0.08:
        n_ladder = 0

    observed_p10 = latent_p10 + selection_boost(v10, boost_max)
    observed_p10 = min(max(observed_p10, 0.01), 0.95)
    if n_ladder > 0:
        n10 = int(rng.binomial(n_ladder, observed_p10))
        n9 = int(rng.binomial(n_ladder - n10, NINE_SHARE))
    else:
        n10 = 0
        n9 = 0
    n_le8 = n_ladder - n10 - n9

    card = {
        "card_id": f"card_{card_id:04d}",
        "era": era,
        "qty": 1,
        "raw_price": raw_price,
        "v10": v10,
        "v9": v9,
        "near_mint_price": near_mint_price,
        "grade10_sold_median": grade10_sold_median,
        "grade10_sold_n": grade10_sold_n,
        "ungraded_sold_median": ungraded_sold_median,
        "ungraded_sold_n": ungraded_sold_n,
        "feed_sales_lifetime": feed_sales_lifetime,
        "grade10_sales_180d": grade10_sales_180d,
        "n10": n10,
        "n9": n9,
        "n_le8": n_le8,
    }
    extras = {
        "latent_p10": latent_p10,
        "observed_p10": observed_p10,
    }
    return card, extras


def _plant_wrong_card(rng, card):
    if card["grade10_sold_n"] < 3 or not math.isfinite(card["grade10_sold_median"]):
        card["grade10_sold_n"] = int(rng.integers(3, 9))
        card["grade10_sold_median"] = card["v10"] * rng.uniform(0.9, 1.1)
    factor = rng.uniform(2.2, 9.0)
    card["v10"] = card["grade10_sold_median"] * factor


def _plant_fake_grail(rng, card):
    card["v10"] = rng.uniform(700.0, 6000.0)
    card["raw_price"] = rng.uniform(8.0, 40.0)
    card["v9"] = card["v10"] * rng.uniform(0.35, 0.5)
    card["grade10_sold_n"] = 0
    card["grade10_sold_median"] = math.nan
    card["feed_sales_lifetime"] = int(rng.integers(0, 3))
    card["ungraded_sold_median"] = card["raw_price"] * rng.uniform(0.85, 1.2)
    card["ungraded_sold_n"] = int(rng.integers(3, 9))
    card["near_mint_price"] = card["raw_price"] * rng.uniform(0.9, 1.3)


def _plant_thin_outlier(rng, card):
    card["v10"] = rng.uniform(120.0, 900.0)
    card["raw_price"] = card["v10"] / rng.uniform(5.0, 30.0)
    card["v9"] = math.nan
    card["feed_sales_lifetime"] = int(rng.integers(0, 2))
    card["grade10_sold_n"] = 0
    card["grade10_sold_median"] = math.nan
    card["ungraded_sold_median"] = card["raw_price"] * rng.uniform(0.85, 1.2)
    card["ungraded_sold_n"] = int(rng.integers(3, 9))


def _plant_bad_raw(rng, card, position):
    """Corrupt the raw price downward; vary which rescue signal exists.

    position cycles 0..4: 0-2 keep the ungraded-sold signal (primary
    repair path), 3 has no signal but a near-mint price (structural-floor
    repair), 4 has neither (structural-floor DROP).

    About 20% of divisors are drawn sub-threshold (1.5x-2.8x), too mild
    for the repair rules to see — the straddle the module docstring
    promises.
    """
    if rng.uniform() < BAD_RAW_SUBTHRESHOLD_FRACTION:
        divisor = rng.uniform(1.5, 2.8)
    else:
        divisor = rng.uniform(2.0, 150.0)
    original_raw = card["raw_price"]
    card["raw_price"] = original_raw / divisor
    if math.isnan(card["v9"]):
        card["v9"] = card["v10"] * rng.uniform(0.3, 0.6)
    if position <= 2:
        card["ungraded_sold_median"] = original_raw * rng.uniform(0.85, 1.2)
        card["ungraded_sold_n"] = int(rng.integers(3, 9))
    elif position == 3:
        card["ungraded_sold_n"] = 0
        card["ungraded_sold_median"] = math.nan
        card["near_mint_price"] = original_raw * rng.uniform(0.9, 1.3)
    else:
        card["ungraded_sold_n"] = 0
        card["ungraded_sold_median"] = math.nan
        card["near_mint_price"] = math.nan
    return divisor


def _plant_genuine_grail(rng, card):
    multiplier = rng.uniform(80.0, 300.0)
    card["raw_price"] = rng.uniform(3.0, 20.0)
    card["v10"] = card["raw_price"] * multiplier
    card["v9"] = card["v10"] * rng.uniform(0.35, 0.55)
    card["grade10_sold_median"] = card["v10"] * rng.uniform(0.9, 1.1)
    card["grade10_sold_n"] = int(rng.integers(3, 12))
    card["feed_sales_lifetime"] = int(rng.integers(3, 15))
    card["ungraded_sold_median"] = card["raw_price"] * rng.uniform(0.85, 1.2)
    card["ungraded_sold_n"] = int(rng.integers(3, 10))
    card["near_mint_price"] = card["raw_price"] * rng.uniform(0.9, 1.3)
    return multiplier


def _plant_borderline_hot_market(rng, card):
    """A legit card the market just ran up on: the sold median lags V10."""
    if card["grade10_sold_n"] < 3 or not math.isfinite(card["grade10_sold_median"]):
        card["grade10_sold_n"] = int(rng.integers(3, 9))
        card["grade10_sold_median"] = card["v10"] * rng.uniform(0.9, 1.1)
    factor = rng.uniform(2.0, 4.5)
    card["v10"] = card["grade10_sold_median"] * factor
    card["v9"] = card["v10"] * rng.uniform(0.3, 0.6)
    card["feed_sales_lifetime"] = int(rng.integers(4, 20))
    return f"borderline_clean;kind=hot_market;factor={factor:.2f}"


def _plant_borderline_sparse_release(rng, card):
    """A legit new release: thin history, no V9, no graded-sold data yet."""
    card["v10"] = rng.uniform(150.0, 900.0)
    card["raw_price"] = card["v10"] / rng.uniform(5.0, 30.0)
    card["v9"] = math.nan
    card["feed_sales_lifetime"] = int(rng.integers(0, 2))
    card["grade10_sold_n"] = 0
    card["grade10_sold_median"] = math.nan
    card["ungraded_sold_median"] = card["raw_price"] * rng.uniform(0.85, 1.2)
    card["ungraded_sold_n"] = int(rng.integers(3, 9))
    card["near_mint_price"] = card["raw_price"] * rng.uniform(0.9, 1.3)
    return f"borderline_clean;kind=sparse_release;v10={card['v10']:.0f}"


def _plant_borderline_ratio_tail(rng, card):
    """A legit card whose honest V10/V9 ratio sits in the 40-75 tail."""
    ratio = rng.uniform(40.0, 75.0)
    card["v9"] = card["v10"] / ratio
    return f"borderline_clean;kind=ratio_tail;ratio={ratio:.1f}"


def _plant_borderline_exempt_grail(rng, card):
    """A legit 80x-120x multiplier WITH graded-sold corroboration."""
    multiplier = rng.uniform(80.0, 120.0)
    card["raw_price"] = rng.uniform(3.0, 20.0)
    card["v10"] = card["raw_price"] * multiplier
    card["v9"] = card["v10"] * rng.uniform(0.35, 0.55)
    card["grade10_sold_median"] = card["v10"] * rng.uniform(0.9, 1.1)
    card["grade10_sold_n"] = int(rng.integers(3, 12))
    card["feed_sales_lifetime"] = int(rng.integers(3, 15))
    card["ungraded_sold_median"] = card["raw_price"] * rng.uniform(0.85, 1.2)
    card["ungraded_sold_n"] = int(rng.integers(3, 10))
    card["near_mint_price"] = card["raw_price"] * rng.uniform(0.9, 1.3)
    return f"borderline_clean;kind=exempt_grail;multiplier={multiplier:.0f}"


def _refresh_ladder(rng, card, latent_p10, boost_max):
    """Redraw the pop ladder after a plant moved V10, keeping the world
    self-consistent: the selection boost depends on the (new) V10."""
    observed_p10 = latent_p10 + selection_boost(card["v10"], boost_max)
    observed_p10 = min(max(observed_p10, 0.01), 0.95)
    n_ladder = card["n10"] + card["n9"] + card["n_le8"]
    if n_ladder > 0:
        n10 = int(rng.binomial(n_ladder, observed_p10))
        n9 = int(rng.binomial(n_ladder - n10, NINE_SHARE))
    else:
        n10 = 0
        n9 = 0
    card["n10"] = n10
    card["n9"] = n9
    card["n_le8"] = n_ladder - n10 - n9
    return observed_p10


def generate_world(
    n_cards=DEFAULT_N_CARDS,
    seed=DEFAULT_SEED,
    selection_boost_max=SELECTION_BOOST_MAX,
    multiplier_scale=MULTIPLIER_SCALE,
):
    """Return (cards, truth, true_prices) DataFrames, fully seeded.

    true_prices has card_id, true_raw, true_v10, true_v9 (see the module
    docstring for the rule). It lives in memory only: write() keeps the
    committed CSVs unchanged, and the base-price capture below copies
    values without drawing, so cards and truth equal generate()'s exactly.
    """
    rng = np.random.default_rng(seed)

    cards = []
    latent_list = []
    observed_list = []
    base_prices = []
    card_ids = range(n_cards)
    for card_id in card_ids:
        era = str(rng.choice(ERA_NAMES, p=ERA_SHARES))
        card, extras = _draw_base_card(rng, card_id, era, selection_boost_max, multiplier_scale)
        cards.append(card)
        latent_list.append(extras["latent_p10"])
        observed_list.append(extras["observed_p10"])
        # A copy, not a draw: the plants below overwrite these fields.
        base_price = {"raw": card["raw_price"], "v10": card["v10"], "v9": card["v9"]}
        base_prices.append(base_price)

    labels = ["clean"] * n_cards
    details = [""] * n_cards

    total_planted = sum(PATHOLOGY_COUNTS.values())
    total_borderline = sum(BORDERLINE_COUNTS.values())
    shuffled = rng.permutation(n_cards)
    planted_indices = shuffled[:total_planted]
    borderline_indices = shuffled[total_planted : total_planted + total_borderline]
    cursor = 0
    for pathology_name, pathology_count in PATHOLOGY_COUNTS.items():
        chosen = planted_indices[cursor : cursor + pathology_count]
        cursor += pathology_count
        for position, index in enumerate(chosen):
            index = int(index)
            card = cards[index]
            labels[index] = pathology_name
            if pathology_name == "wrong_card":
                _plant_wrong_card(rng, card)
            elif pathology_name == "fake_grail":
                _plant_fake_grail(rng, card)
            elif pathology_name == "thin_outlier":
                _plant_thin_outlier(rng, card)
            elif pathology_name == "bad_raw":
                divisor = _plant_bad_raw(rng, card, position % 5)
                details[index] = f"divisor={divisor:.1f};rescue={position % 5}"
            elif pathology_name == "genuine_grail":
                multiplier = _plant_genuine_grail(rng, card)
                details[index] = f"multiplier={multiplier:.0f}"

    # Borderline-legit clean cards: the label stays "clean" and
    # should_flag stays False — any gate that fires on one of these is a
    # false positive by construction.
    cursor = 0
    for borderline_kind, borderline_count in BORDERLINE_COUNTS.items():
        chosen = borderline_indices[cursor : cursor + borderline_count]
        cursor += borderline_count
        for index in chosen:
            index = int(index)
            card = cards[index]
            if borderline_kind == "hot_market":
                detail = _plant_borderline_hot_market(rng, card)
            elif borderline_kind == "sparse_release":
                detail = _plant_borderline_sparse_release(rng, card)
            elif borderline_kind == "ratio_tail":
                detail = _plant_borderline_ratio_tail(rng, card)
            else:
                detail = _plant_borderline_exempt_grail(rng, card)
            details[index] = detail
            if borderline_kind != "ratio_tail":  # these plants moved V10
                observed_list[index] = _refresh_ladder(
                    rng, card, latent_list[index], selection_boost_max
                )

    truth_rows = []
    for index in range(n_cards):
        label = labels[index]
        should_flag = label in ("wrong_card", "fake_grail", "thin_outlier", "bad_raw")
        truth_row = {
            "card_id": cards[index]["card_id"],
            "label": label,
            "should_flag": should_flag,
            "latent_p10": latent_list[index],
            "observed_p10": observed_list[index],
            "detail": details[index],
        }
        truth_rows.append(truth_row)

    price_rows = []
    for index in card_ids:
        label = labels[index]
        if label in JUNK_FEED_LABELS:
            source = base_prices[index]
            true_raw = source["raw"]
            true_v10 = source["v10"]
            true_v9 = source["v9"]
        else:
            card = cards[index]
            true_raw = card["raw_price"]
            true_v10 = card["v10"]
            true_v9 = card["v9"]
        price_row = {
            "card_id": cards[index]["card_id"],
            "true_raw": true_raw,
            "true_v10": true_v10,
            "true_v9": true_v9,
        }
        price_rows.append(price_row)

    cards_df = pd.DataFrame(cards)
    truth_df = pd.DataFrame(truth_rows)
    true_prices_df = pd.DataFrame(price_rows)
    return cards_df, truth_df, true_prices_df


def generate(
    n_cards=DEFAULT_N_CARDS,
    seed=DEFAULT_SEED,
    selection_boost_max=SELECTION_BOOST_MAX,
    multiplier_scale=MULTIPLIER_SCALE,
):
    """Return (cards DataFrame, truth DataFrame), fully seeded.

    At the default knobs this reproduces synth/cards.csv and
    synth/truth.csv byte for byte once written (tests/test_golden_data.py).
    """
    cards_df, truth_df, _true_prices_df = generate_world(
        n_cards=n_cards,
        seed=seed,
        selection_boost_max=selection_boost_max,
        multiplier_scale=multiplier_scale,
    )
    return cards_df, truth_df


def write(out_dir="synth", n_cards=DEFAULT_N_CARDS, seed=DEFAULT_SEED):
    """Generate and write synth/cards.csv and synth/truth.csv."""
    cards_df, truth_df = generate(n_cards=n_cards, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    cards_path = os.path.join(out_dir, "cards.csv")
    truth_path = os.path.join(out_dir, "truth.csv")
    cards_df.to_csv(cards_path, index=False, float_format="%.6g")
    truth_df.to_csv(truth_path, index=False, float_format="%.6g")
    return cards_path, truth_path


def load(synth_dir="synth"):
    """Load the committed synthetic data, regenerating in memory if absent."""
    cards_path = os.path.join(synth_dir, "cards.csv")
    truth_path = os.path.join(synth_dir, "truth.csv")
    if os.path.exists(cards_path) and os.path.exists(truth_path):
        cards_df = pd.read_csv(cards_path)
        truth_df = pd.read_csv(truth_path)
        return cards_df, truth_df
    return generate()


if __name__ == "__main__":
    written = write()
    print(f"wrote {written[0]} and {written[1]}")
