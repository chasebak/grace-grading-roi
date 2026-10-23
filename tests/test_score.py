"""Hand-checked sigma and GRACE score."""

import math

import pytest

from grace import score

# Worked example (hand arithmetic), matching the NPV test example:
#   p = (0.5, 0.3, 0.2), V = (1000, 400, 50), f = 0.13, s = 5
#   nets = (865, 343, 38.5); U = 0.9748243
#   discounted payoffs = (843.22, 334.37, 37.53)
#   mean = 529.43
#   var  = 0.5*313.79^2 + 0.3*(-195.07)^2 + 0.2*(-491.90)^2 = 109,041
#   sigma = 330.2
U = 0.9748243


def test_sigma_hand_check():
    sigma = score.discounted_payoff_sigma(
        0.5, 0.3, 0.2, 1000.0, 400.0, 50.0, 0.13, 5.0, U
    )
    assert sigma == pytest.approx(330.2, rel=1e-3)


def test_sigma_zero_when_payoffs_equal():
    sigma = score.discounted_payoff_sigma(
        0.5, 0.3, 0.2, 100.0, 100.0, 100.0, 0.13, 5.0, U
    )
    assert sigma == pytest.approx(0.0, abs=1e-9)


def test_sigma_scales_with_discount_factor():
    sigma_full = score.discounted_payoff_sigma(
        0.5, 0.3, 0.2, 1000.0, 400.0, 50.0, 0.13, 5.0, 1.0
    )
    sigma_half = score.discounted_payoff_sigma(
        0.5, 0.3, 0.2, 1000.0, 400.0, 50.0, 0.13, 5.0, 0.5
    )
    assert sigma_half == pytest.approx(0.5 * sigma_full)


def test_grace_default_is_npv_over_sqrt_capital():
    # lambda=0, a=0.5: 200 / sqrt(400) = 10
    assert score.grace_score(200.0, 999.0, 400.0) == pytest.approx(10.0)


def test_grace_lambda_penalty():
    # (200 - 0.5*100) / 400^0.5 = 150/20 = 7.5
    assert score.grace_score(200.0, 100.0, 400.0, lam=0.5, a=0.5) == pytest.approx(7.5)


def test_grace_nan_on_nonpositive_capital():
    assert math.isnan(score.grace_score(100.0, 10.0, 0.0))
