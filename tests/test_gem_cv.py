"""Experiment 3 logic checks: metrics, folds and estimator endpoints.

Each check is written so it can fail: the metric fixtures are hand
computed, the fold-leak check has a control that changes a training card
and must move the fit, and the endpoint checks would catch the scratch
prototype's NaN at m = 0 with an empty ladder.
"""

import math

import numpy as np
import pytest

from grace import pooling, synth
from grace.config import DEFAULT_CONFIG
from grace.experiments import gem_cv


@pytest.fixture(scope="module")
def world():
    cards_df, truth_df = synth.load()
    frame = gem_cv.build_frame(cards_df, truth_df)
    folds = gem_cv.assign_folds(len(frame))
    return frame, folds


@pytest.fixture(scope="module")
def summary():
    cards_df, truth_df = synth.load()
    return gem_cv.compute_summary(cards_df, truth_df)


# --- metrics -----------------------------------------------------------------


def test_log_loss_at_one_half_is_ln_2_for_any_outcome():
    n = 20
    outcomes = range(n + 1)
    for k in outcomes:
        assert gem_cv.binomial_log_loss(0.5, k, n) == pytest.approx(math.log(2.0), abs=1e-12)


def test_log_loss_hand_case():
    # 3 gems in 4 trials at p = 0.75: -(3 ln 0.75 + ln 0.25) / 4
    expected = -(3.0 * math.log(0.75) + math.log(0.25)) / 4.0
    assert gem_cv.binomial_log_loss(0.75, 3, 4) == pytest.approx(expected, abs=1e-12)


def test_brier_two_trial_example():
    # p = 0.7, one gem in two trials: ((1 - 0.7)^2 + 0.7^2) / 2 = (0.09 + 0.49) / 2
    assert gem_cv.binomial_brier(0.7, 1, 2) == pytest.approx(0.29, abs=1e-12)
    # a perfect prediction of a sure thing scores 0
    assert gem_cv.binomial_brier(1.0, 2, 2) == 0.0


def test_ece_is_zero_on_perfectly_calibrated_bins():
    predictions = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75]
    gem_counts = [0, 2, 1, 1, 3, 3]  # rates 0, .5, .25, .25 (mean .25); .75, .75
    assert gem_cv.expected_calibration_error(predictions, gem_counts, 4) == pytest.approx(0.0)


def test_ece_control_a_miscalibrated_bin_is_caught():
    predictions = [0.25, 0.25, 0.75, 0.75]
    gem_counts = [1, 1, 1, 1]  # realized 0.25 everywhere: the top bin is 0.5 too high
    error = gem_cv.expected_calibration_error(predictions, gem_counts, 4)
    assert error == pytest.approx(0.5 * 0.5)  # half the cards, gap 0.5


def test_ece_puts_p_equal_one_in_the_top_bin():
    error = gem_cv.expected_calibration_error([1.0], [4], 4)
    assert error == pytest.approx(0.0)


def test_clipping_fixture():
    big_loss = -math.log(gem_cv.CLIP)
    # p = 0 with a gem observed costs -ln(1e-6) per gem trial, not infinity
    assert gem_cv.binomial_log_loss(0.0, 1, 1) == pytest.approx(big_loss)
    assert gem_cv.binomial_log_loss(1.0, 0, 1) == pytest.approx(big_loss)
    # p = 0 with no gems costs almost nothing
    assert gem_cv.binomial_log_loss(0.0, 0, 5) == pytest.approx(-math.log(1.0 - gem_cv.CLIP))
    # the count: exactly the predictions the clip moves
    predictions = [0.0, 1.0, gem_cv.CLIP, 1.0 - gem_cv.CLIP, 0.5, 1e-9]
    assert gem_cv.clipped_count(predictions) == 3


# --- folds -------------------------------------------------------------------


def test_each_card_is_in_exactly_one_test_fold(world):
    frame, folds = world
    n_cards = len(frame)
    seen = np.zeros(n_cards, dtype=int)
    fold_ids = range(gem_cv.K_FOLDS)
    sizes = []
    for test_fold in fold_ids:
        test_mask = folds == test_fold
        seen += test_mask.astype(int)
        sizes.append(int(test_mask.sum()))
    assert seen.tolist() == [1] * n_cards
    assert max(sizes) - min(sizes) <= 1
    card_ids = frame["card_id"].tolist()
    assert len(set(card_ids)) == n_cards  # one row per card


def test_changing_the_test_fold_leaves_the_fit_unchanged(world):
    frame, folds = world
    test_fold = 2
    before = gem_cv.fit_training_fold(frame, folds, test_fold)
    mutated = frame.copy()
    test_mask = folds == test_fold
    target_names = list(gem_cv.TARGETS.keys())
    for target in target_names:
        column = f"k_{target}"
        mutated.loc[test_mask, column] = gem_cv.SUBMISSION_SIZE - mutated.loc[test_mask, column]
    mutated.loc[test_mask, "n10"] = mutated.loc[test_mask, "n_ladder"]
    after = gem_cv.fit_training_fold(mutated, folds, test_fold)
    assert after == before


def test_control_changing_a_training_card_moves_the_fit(world):
    frame, folds = world
    test_fold = 2
    before = gem_cv.fit_training_fold(frame, folds, test_fold)
    mutated = frame.copy()
    training_positions = np.flatnonzero(folds != test_fold)
    row = int(training_positions[0])
    era = mutated.loc[row, "era"]
    mutated.loc[row, "n_ladder"] = 400
    mutated.loc[row, "n10"] = 400
    after = gem_cv.fit_training_fold(mutated, folds, test_fold)
    assert after[era]["mu_hat"] != before[era]["mu_hat"]
    assert after[era]["m_hat"] != before[era]["m_hat"]


def test_outcomes_never_enter_the_fit(world):
    frame, folds = world
    before = gem_cv.fit_training_fold(frame, folds, 0)
    mutated = frame.copy()
    target_names = list(gem_cv.TARGETS.keys())
    for target in target_names:
        mutated[f"k_{target}"] = 0
    assert gem_cv.fit_training_fold(mutated, folds, 0) == before


# --- estimator endpoints -------------------------------------------------------


def test_m_zero_is_the_raw_rate():
    assert gem_cv.shrunk_p10(7, 20, 0.4, 0.0) == pytest.approx(0.35)
    assert gem_cv.shrunk_p10(20, 20, 0.4, 0.0) == 1.0  # an all-gem ladder is fine at m = 0


def test_m_zero_with_an_empty_ladder_falls_back_to_the_era_mean():
    value = gem_cv.shrunk_p10(0, 0, 0.4, 0.0)
    assert not math.isnan(value)
    assert value == 0.4


def test_huge_m_is_the_era_mean():
    assert gem_cv.shrunk_p10(30, 31, 0.2, 1e12) == pytest.approx(0.2, abs=1e-9)
    assert gem_cv.shrunk_p10(30, 31, 0.2, math.inf) == 0.2
    assert gem_cv.shrunk_p10(30, 31, 0.2, math.nan) == 0.2


def test_shipped_m40_matches_the_engine_pooling():
    ladders = [(0, 0, 0), (3, 4, 5), (20, 0, 0), (0, 10, 30), (17, 20, 10)]
    for n10, n9, n_le8 in ladders:
        engine = pooling.pooled_rates(n10, n9, n_le8, 0.3, 40.0).p10
        mine = gem_cv.shrunk_p10(n10, n10 + n9 + n_le8, 0.3, 40.0)
        assert mine == pytest.approx(engine, abs=1e-15)


def test_raw_rate_estimator_uses_the_era_mean_for_empty_ladders(world):
    frame, folds = world
    fit = gem_cv.fit_training_fold(frame, folds, 0)
    empty = frame[(frame["n_ladder"] == 0) & (folds == 0)]
    assert len(empty) > 0
    records = empty.to_dict("records")
    for card in records:
        predictions = gem_cv.predict(card, fit, "unscreened")
        assert predictions["raw_rate"] == fit[card["era"]]["mu_hat"]
        assert predictions["eb_mom"] == fit[card["era"]]["mu_hat"]
        assert predictions["m40_era_mean"] == fit[card["era"]]["mu_hat"]
        assert predictions["eb_prior"] == DEFAULT_CONFIG.era_priors[card["era"]]


# --- results ---------------------------------------------------------------------


def test_no_estimator_beats_the_true_rate(summary):
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        estimators = summary["targets"][target]["estimators"]
        floor = estimators[gem_cv.REFERENCE]["log_loss"]["mean"]
        for name in gem_cv.ESTIMATORS:
            assert estimators[name]["log_loss"]["mean"] > floor


def test_only_the_raw_rate_is_clipped(summary):
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        estimators = summary["targets"][target]["estimators"]
        assert estimators["raw_rate"]["clipped"] > 0
        for name in gem_cv.ESTIMATORS:
            if name != "raw_rate":
                assert estimators[name]["clipped"] == 0


def test_baseline_paired_difference_is_zero(summary):
    targets = list(gem_cv.TARGETS.keys())
    for target in targets:
        paired = summary["targets"][target]["estimators"][gem_cv.BASELINE]
        assert paired["log_loss_diff_vs_m40"]["mean"] == 0.0
        assert paired["log_loss_diff_vs_m40"]["folds_better"] == 0


def test_verdict_rule():
    assert gem_cv.verdict({"mean": -0.002, "sd": 0.001}) == "better"
    assert gem_cv.verdict({"mean": 0.002, "sd": 0.001}) == "worse"
    assert gem_cv.verdict({"mean": -0.002, "sd": 0.003}) == "null"
    assert gem_cv.verdict({"mean": 0.02, "sd": 0.03}) == "null"


def test_cv_is_deterministic(summary):
    cards_df, truth_df = synth.load()
    again = gem_cv.compute_summary(cards_df, truth_df)
    assert again == summary


def test_cli_prints_every_target(capsys):
    assert gem_cv.main([]) == 0
    printed = capsys.readouterr().out
    target_names = list(gem_cv.TARGETS.keys())
    for target in target_names:
        assert f"target: {target}" in printed


# --- wiring and pinned constants (each would pass a broken build otherwise) ---


def test_predict_pooled_m40_is_the_engine_pooling_on_real_cards(world):
    """predict()'s m = 40 entry must be the engine's own pooled p10 at the
    fixed era prior and cfg.m, card by card; shrunk_p10 alone being right
    does not prove predict() feeds it the prior. Control: the era mean is a
    different target, so wiring mu_hat in would fail here."""
    frame, folds = world
    cfg = DEFAULT_CONFIG
    assert cfg.m == gem_cv.SHIPPED_M
    fit = gem_cv.fit_training_fold(frame, folds, 0)
    test = frame[folds == 0]
    records = test.to_dict("records")
    moved_by_mu_hat = 0
    for card in records:
        era = card["era"]
        engine = pooling.pooled_rates(
            int(card["n10"]), int(card["n9"]), int(card["n_le8"]), cfg.era_priors[era], cfg.m
        ).p10
        predictions = gem_cv.predict(card, fit, "unscreened", cfg)
        assert predictions["pooled_m40"] == pytest.approx(engine, abs=1e-15)
        if abs(predictions["m40_era_mean"] - engine) > 1e-6:
            moved_by_mu_hat += 1
    assert moved_by_mu_hat > 0


def test_cross_arms_change_one_half_of_empirical_bayes_each(world):
    """m40_era_mean = (n10 + 40 mu_hat) / (n + 40); eb_prior = (n10 + m_hat
    mu) / (n + m_hat). Each must differ from both parents on some card, or
    it is a copy and splits nothing."""
    frame, folds = world
    cfg = DEFAULT_CONFIG
    fit = gem_cv.fit_training_fold(frame, folds, 1)
    test = frame[folds == 1]
    records = test.to_dict("records")
    differs = {"m40_era_mean": 0, "eb_prior": 0}
    for card in records:
        era = card["era"]
        n10 = int(card["n10"])
        n_ladder = int(card["n_ladder"])
        mu_hat = fit[era]["mu_hat"]
        m_hat = fit[era]["m_hat"]
        prior = cfg.era_priors[era]
        predictions = gem_cv.predict(card, fit, "next_pop", cfg)
        target_only = gem_cv.shrunk_p10(n10, n_ladder, mu_hat, 40.0)
        strength_only = gem_cv.shrunk_p10(n10, n_ladder, prior, m_hat)
        assert predictions["m40_era_mean"] == pytest.approx(target_only, abs=1e-15)
        assert predictions["eb_prior"] == pytest.approx(strength_only, abs=1e-15)
        parents = (predictions["pooled_m40"], predictions["eb_mom"])
        for name in differs:
            value = predictions[name]
            if abs(value - parents[0]) > 1e-9 and abs(value - parents[1]) > 1e-9:
                differs[name] += 1
    assert differs["m40_era_mean"] > 0
    assert differs["eb_prior"] > 0


def test_clip_is_pinned_at_one_in_a_million():
    """The fixture above reads gem_cv.CLIP, so a changed clip would pass it;
    this pins the value and the loss it sets: -ln(1e-6) = 13.8155."""
    assert gem_cv.CLIP == 1e-6
    assert gem_cv.binomial_log_loss(0.0, 1, 1) == pytest.approx(13.8155, abs=1e-4)


def test_ece_control_over_and_under_prediction_both_count():
    """Without abs(), an over- and an under-predicting bin would cancel.
    Bin [0.2, 0.3): two cards at p = 0.25, realized 0.75 (k = 3 of 4):
    gap 0.5 under. Bin [0.7, 0.8): two cards at p = 0.75, realized 0.25
    (k = 1 of 4): gap 0.5 over. ECE = 0.5 * 0.5 + 0.5 * 0.5 = 0.5; a signed
    sum would give 0."""
    predictions = [0.25, 0.25, 0.75, 0.75]
    gem_counts = [3, 3, 1, 1]
    error = gem_cv.expected_calibration_error(predictions, gem_counts, 4)
    assert error == pytest.approx(0.5)
