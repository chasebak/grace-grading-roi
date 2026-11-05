"""Golden data: the generator still produces the committed CSVs, the knobs
work, and capturing true prices adds no random draws.

Why: every README number is computed from synth/*.csv; if generate()
drifts from them, a regenerated dataset would silently change every
result. The knob controls prove the byte comparison can fail.
"""

import math
from pathlib import Path

import pytest

from grace import synth

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_DIR = REPO_ROOT / "synth"
CSV_NAMES = ("cards.csv", "truth.csv")


def written_bytes(out_dir, **knobs):
    """Write a generated world with the repo's own writer and read it back."""
    cards_df, truth_df = synth.generate(**knobs)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = out_dir / "cards.csv"
    truth_path = out_dir / "truth.csv"
    cards_df.to_csv(cards_path, index=False, float_format="%.6g")
    truth_df.to_csv(truth_path, index=False, float_format="%.6g")
    contents = {}
    for name in CSV_NAMES:
        path = out_dir / name
        contents[name] = path.read_bytes()
    return contents


def committed_bytes():
    contents = {}
    for name in CSV_NAMES:
        path = COMMITTED_DIR / name
        contents[name] = path.read_bytes()
    return contents


def test_generate_matches_committed_csvs_byte_for_byte(tmp_path):
    generated = written_bytes(tmp_path / "default")
    committed = committed_bytes()
    for name in CSV_NAMES:
        assert generated[name] == committed[name], name


def test_write_matches_committed_csvs_byte_for_byte(tmp_path):
    out_dir = tmp_path / "via_write"
    synth.write(out_dir=str(out_dir))
    committed = committed_bytes()
    for name in CSV_NAMES:
        path = out_dir / name
        assert path.read_bytes() == committed[name], name


@pytest.mark.parametrize(
    "knobs",
    [{"selection_boost_max": 0.0}, {"multiplier_scale": 0.8}],
)
def test_control_a_changed_knob_breaks_the_golden_match(tmp_path, knobs):
    """The byte check must be able to fail: each knob moves the output."""
    generated = written_bytes(tmp_path / "knob", **knobs)
    committed = committed_bytes()
    assert generated["cards.csv"] != committed["cards.csv"]


def test_explicit_default_knobs_equal_the_implicit_defaults():
    cards_a, truth_a = synth.generate()
    cards_b, truth_b = synth.generate(
        selection_boost_max=synth.SELECTION_BOOST_MAX,
        multiplier_scale=synth.MULTIPLIER_SCALE,
    )
    assert cards_a.equals(cards_b)
    assert truth_a.equals(truth_b)


def test_capturing_true_prices_adds_no_random_draws():
    """generate_world() and generate() must return the same cards and truth."""
    cards_plain, truth_plain = synth.generate()
    cards_world, truth_world, true_prices = synth.generate_world()
    assert cards_world.equals(cards_plain)
    assert truth_world.equals(truth_plain)
    assert len(true_prices) == len(cards_world)
    assert true_prices["card_id"].tolist() == cards_world["card_id"].tolist()


def test_knob_runs_are_seeded_and_keep_the_plant_counts():
    """A knob run is its own seeded world (the binomial sampler's draw count
    depends on the rate, so it is not card-aligned with the default), but it
    is deterministic and plants the same number of each pathology."""
    cards_a, truth_a = synth.generate(multiplier_scale=0.8)
    cards_b, truth_b = synth.generate(multiplier_scale=0.8)
    assert cards_a.equals(cards_b)
    assert truth_a.equals(truth_b)
    _cards_null, truth_null = synth.generate(selection_boost_max=0.0)
    counts = truth_null["label"].value_counts()
    planted_counts = list(synth.PATHOLOGY_COUNTS.items())
    for label, expected in planted_counts:
        assert counts[label] == expected
    first_latent = truth_null["latent_p10"].iloc[0]
    _cards_default, truth_default = synth.generate()
    assert first_latent == truth_default["latent_p10"].iloc[0]  # same seed, same first draw


def test_null_control_sets_observed_rate_to_latent_rate():
    """With the boost off, observed = latent (inside the generator's clip)."""
    _cards, truth = synth.generate(selection_boost_max=0.0)
    latent = truth["latent_p10"].tolist()
    observed = truth["observed_p10"].tolist()
    indices = range(len(latent))
    for index in indices:
        clipped = min(max(latent[index], 0.01), 0.95)
        assert observed[index] == pytest.approx(clipped, abs=1e-12)


def test_planted_boost_makes_observed_rates_exceed_latent():
    _cards, truth = synth.generate()
    gap = (truth["observed_p10"] - truth["latent_p10"]).mean()
    assert gap > 0.0


def clean_multipliers(cards_df, truth_df):
    """V10/raw of the plain clean cards, selected by THIS world's own labels."""
    merged = cards_df.merge(truth_df[["card_id", "label", "detail"]], on="card_id")
    clean_mask = merged["detail"].fillna("").eq("") & merged["label"].eq("clean")
    clean = merged[clean_mask]
    return clean["v10"] / clean["raw_price"]


def test_multiplier_scale_lowers_clean_multipliers():
    """Knob worlds are not card-aligned with the default, so each world's
    clean cards are picked from its own truth labels."""
    cards_default, truth_default = synth.generate()
    cards_scaled, truth_scaled = synth.generate(multiplier_scale=0.8)
    default_ratio = clean_multipliers(cards_default, truth_default)
    scaled_ratio = clean_multipliers(cards_scaled, truth_scaled)
    assert len(scaled_ratio) > 0
    assert scaled_ratio.median() < default_ratio.median()


def world_frame():
    cards_df, truth_df, true_prices_df = synth.generate_world()
    merged = cards_df.merge(truth_df, on="card_id").merge(true_prices_df, on="card_id")
    return merged


def same_price(feed, true):
    if isinstance(feed, float) and math.isnan(feed):
        return isinstance(true, float) and math.isnan(true)
    return feed == true


def test_true_price_equals_feed_for_clean_borderline_and_genuine_grails():
    merged = world_frame()
    real_rows = merged[merged["label"].isin(["clean", "genuine_grail"])]
    borderline = real_rows["detail"].fillna("").str.startswith("borderline_clean")
    assert int(borderline.sum()) == sum(synth.BORDERLINE_COUNTS.values())
    records = real_rows.to_dict("records")
    for row in records:
        assert row["true_raw"] == row["raw_price"]
        assert row["true_v10"] == row["v10"]
        assert same_price(row["v9"], row["true_v9"])


def test_junk_plants_true_prices_are_the_base_card():
    """Each junk plant's true price is the card it replaced, not the junk."""
    merged = world_frame()
    wrong = merged[merged["label"] == "wrong_card"]
    assert (wrong["true_v10"] < wrong["v10"]).all()  # the feed V10 was inflated

    bad_raw = merged[merged["label"] == "bad_raw"]
    records = bad_raw.to_dict("records")
    for row in records:
        divisor_part = row["detail"].split(";")[0]
        divisor = float(divisor_part.split("=")[1])
        observed_ratio = row["true_raw"] / row["raw_price"]
        assert observed_ratio == pytest.approx(divisor, abs=0.051)  # detail has 1 decimal

    # Base cards are drawn with the multiplier clipped to [1.5, 55], so every
    # junk row's TRUE prices must sit inside that range; the junk feed rows
    # need not (a fake grail's feed V10 is drawn on its own).
    junk = merged[merged["label"].isin(synth.JUNK_FEED_LABELS)]
    true_multiplier = junk["true_v10"] / junk["true_raw"]
    assert (true_multiplier <= synth.MULTIPLIER_CLIP[1] + 1e-9).all()
    assert (true_multiplier >= synth.MULTIPLIER_CLIP[0] - 1e-9).all()
    feed_multiplier = junk["v10"] / junk["raw_price"]
    assert (feed_multiplier > synth.MULTIPLIER_CLIP[1]).any()  # control: the feed breaks it


def test_control_junk_rows_differ_from_feed():
    """If the capture copied the feed, every junk row would match: it must not."""
    merged = world_frame()
    junk = merged[merged["label"].isin(synth.JUNK_FEED_LABELS)]
    differs = (junk["true_v10"] != junk["v10"]) | (junk["true_raw"] != junk["raw_price"])
    assert bool(differs.all())
