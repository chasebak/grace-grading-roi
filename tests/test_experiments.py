"""The two experiments run end-to-end and their findings hold."""

import math
import os

from grace import synth
from grace.experiments import gates_eval, m_estimation


def test_gates_classifier_metrics():
    cards, truth = synth.generate()
    metrics = gates_eval.evaluate(cards, truth)

    for pathology in gates_eval.PATHOLOGY_LABELS:
        entry = metrics["per_pathology"][pathology]
        assert entry["planted"] > 0
        # plants straddle the gate thresholds on purpose, so recall is
        # not 1.0 — but the ladder must catch well over half of each
        assert entry["recall"] >= 0.5, pathology

    # no genuine grail may be dropped — and the corroboration exemption
    # must also hold from the legit side (borderline 80x-120x grails)
    assert metrics["grail_survived"] == metrics["grail_total"]
    assert metrics["exempt_grail_total"] > 0
    assert metrics["exempt_grail_survived"] == metrics["exempt_grail_total"]
    # the borderline-legit clean subpopulation makes precision a real,
    # non-degenerate number: some clean cards must get flagged, so
    # precision sits strictly below 1.0 but stays usefully high
    assert metrics["clean_flagged"] > 0
    assert 0.5 <= metrics["flag_precision"] < 1.0
    clean_rate = metrics["clean_flagged"] / metrics["clean_total"]
    assert clean_rate <= 0.12


def test_gates_figure_written(tmp_path):
    cards, truth = synth.generate(n_cards=200, seed=2)
    metrics = gates_eval.evaluate(cards, truth)
    out_path = str(tmp_path / "gates.png")
    gates_eval.make_figure(metrics, out_path)
    assert os.path.getsize(out_path) > 0


def test_m_estimate_hand_check_no_overdispersion():
    # identical rates -> zero variance -> no overdispersion -> infinite m
    gem_counts = [5, 5, 5, 5]
    ladder_sizes = [10, 10, 10, 10]
    estimate = m_estimation.m_estimate(gem_counts, ladder_sizes)
    assert math.isinf(estimate)


def test_m_estimates_by_era_finite_and_positive():
    cards, _truth = synth.generate()
    estimates = m_estimation.m_estimates_by_era(cards)
    assert set(estimates.keys()) == {"modern", "mid", "vintage"}
    for era, estimate in estimates.items():
        assert estimate > 0, era
        assert estimate < 500, era  # real overdispersion is present


def test_calibration_shows_planted_optimism():
    cards, truth = synth.generate()
    per_card, binned = m_estimation.calibration_table(cards, truth)
    assert len(per_card) > 100
    assert len(binned) == m_estimation.CALIBRATION_BINS
    gap = (per_card["predicted_p10"] - per_card["realized_gem_rate"]).mean()
    assert gap > 0.0  # predictions sit above realized: the selection effect
    low = per_card[per_card["value_group"] == "low_value"]
    high = per_card[per_card["value_group"] == "high_value"]
    low_gap = (low["predicted_p10"] - low["realized_gem_rate"]).mean()
    high_gap = (high["predicted_p10"] - high["realized_gem_rate"]).mean()
    assert low_gap > high_gap  # strongest for low-value cards


def test_calibration_figure_written(tmp_path):
    cards, truth = synth.generate(n_cards=300, seed=4)
    per_card, binned = m_estimation.calibration_table(cards, truth)
    out_path = str(tmp_path / "calibration.png")
    m_estimation.make_figure(per_card, binned, out_path)
    assert os.path.getsize(out_path) > 0


def test_prior_sensitivity_rankings_stable():
    cards, _truth = synth.generate(n_cards=300, seed=9)
    correlations = m_estimation.prior_sensitivity(cards)
    for shift, rho in correlations.items():
        assert rho > 0.8, shift
