"""The synthetic generator: determinism, planted properties, selection effect."""

import numpy as np

from grace import synth


def test_deterministic_for_same_seed():
    cards_a, truth_a = synth.generate(n_cards=150, seed=11)
    cards_b, truth_b = synth.generate(n_cards=150, seed=11)
    assert cards_a.equals(cards_b)
    assert truth_a.equals(truth_b)


def test_different_seed_differs():
    cards_a, _ = synth.generate(n_cards=150, seed=11)
    cards_b, _ = synth.generate(n_cards=150, seed=12)
    assert not cards_a.equals(cards_b)


def test_planted_counts():
    _cards, truth = synth.generate()
    counts = truth["label"].value_counts()
    for label, expected in synth.PATHOLOGY_COUNTS.items():
        assert counts[label] == expected


def test_fake_grails_have_no_corroboration():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    grails = merged[merged["label"] == "fake_grail"]
    assert (grails["grade10_sold_n"] == 0).all()
    assert (grails["feed_sales_lifetime"] <= 2).all()
    assert (grails["v10"] >= 700.0).all()


def test_thin_outliers_have_no_v9_and_no_history():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    thin = merged[merged["label"] == "thin_outlier"]
    assert thin["v9"].isna().all()
    assert (thin["feed_sales_lifetime"] <= 1).all()


def test_wrong_cards_inflated_vs_graded_median():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    wrong = merged[merged["label"] == "wrong_card"]
    ratio = wrong["v10"] / wrong["grade10_sold_median"]
    assert (ratio > 2.0).all()
    assert (wrong["grade10_sold_n"] >= 3).all()


def test_genuine_grails_are_corroborated():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    grails = merged[merged["label"] == "genuine_grail"]
    multiplier = grails["v10"] / grails["raw_price"]
    assert (multiplier >= 79.0).all()
    assert (grails["grade10_sold_n"] >= 3).all()


def test_borderline_clean_subpopulation():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    borderline = merged[merged["detail"].str.startswith("borderline_clean", na=False)]
    # ~60-80 of the 500 clean cards; label stays clean, should_flag False
    assert len(borderline) == sum(synth.BORDERLINE_COUNTS.values())
    assert 60 <= len(borderline) <= 80
    assert (borderline["label"] == "clean").all()
    assert (~borderline["should_flag"]).all()

    exempt = borderline[borderline["detail"].str.contains("kind=exempt_grail")]
    multiplier = exempt["v10"] / exempt["raw_price"]
    assert (multiplier >= 79.0).all()
    assert (multiplier <= 121.0).all()
    assert (exempt["grade10_sold_n"] >= 3).all()

    sparse = borderline[borderline["detail"].str.contains("kind=sparse_release")]
    assert sparse["v9"].isna().all()
    assert (sparse["feed_sales_lifetime"] <= 1).all()
    assert (sparse["grade10_sold_n"] == 0).all()

    ratio_tail = borderline[borderline["detail"].str.contains("kind=ratio_tail")]
    ratio = ratio_tail["v10"] / ratio_tail["v9"]
    assert (ratio >= 39.0).all()
    assert (ratio <= 76.0).all()

    hot = borderline[borderline["detail"].str.contains("kind=hot_market")]
    hot_ratio = hot["v10"] / hot["grade10_sold_median"]
    assert (hot_ratio >= 1.9).all()
    assert (hot_ratio <= 4.6).all()
    assert (hot["grade10_sold_n"] >= 3).all()


def test_bad_raw_divisors_straddle_the_repair_threshold():
    _cards, truth = synth.generate()
    bad = truth[truth["label"] == "bad_raw"]
    divisors = []
    for detail in bad["detail"].tolist():
        divisor_text = detail.split(";")[0].split("=")[1]
        divisors.append(float(divisor_text))
    sub_threshold = [divisor for divisor in divisors if divisor < 2.8]
    assert len(sub_threshold) >= 3  # ~20% of 30 drawn in 1.5x-2.8x
    assert max(divisors) > 10.0  # the easy-to-catch end is still there


def test_selection_effect_inflates_low_value_observed_rates():
    cards, truth = synth.generate()
    merged = cards.merge(truth, on="card_id")
    clean = merged[merged["label"] == "clean"]
    gap = clean["observed_p10"] - clean["latent_p10"]
    value_median = clean["v10"].median()
    low_gap = gap[clean["v10"] <= value_median].mean()
    high_gap = gap[clean["v10"] > value_median].mean()
    assert low_gap > high_gap > 0.0
    assert low_gap > 0.05


def test_ladder_counts_consistent():
    cards, _ = synth.generate()
    total = cards["n10"] + cards["n9"] + cards["n_le8"]
    assert (cards["n10"] >= 0).all()
    assert (cards["n9"] >= 0).all()
    assert (cards["n_le8"] >= 0).all()
    assert (total >= 0).all()
    some_zero = int((total == 0).sum())
    assert some_zero > 0  # the no-pop prior path gets exercised


def test_write_and_load_round_trip(tmp_path):
    out_dir = str(tmp_path / "synthdata")
    _cards_path, _truth_path = synth.write(out_dir=out_dir, n_cards=80, seed=3)
    cards, truth = synth.load(out_dir)
    assert len(cards) == 80
    assert len(truth) == 80
    assert np.isfinite(cards["v10"]).all()
