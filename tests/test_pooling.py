"""Hand-checked partial pooling."""

import pytest

from grace import pooling


def test_pooled_rates_hand_check():
    # n10=8, n9=10, n_le8=2, mu=0.3, m=40
    # p10 = (8 + 12) / (20 + 40) = 1/3
    # share9 = (10 + 20) / (12 + 40) = 30/52
    # p9 = (2/3) * 30/52 = 0.3846154 ; p_le8 = (2/3) * 22/52 = 0.2820513
    rates = pooling.pooled_rates(8, 10, 2, 0.3, 40.0)
    assert rates.p10 == pytest.approx(1.0 / 3.0)
    assert rates.p9 == pytest.approx(0.3846154, abs=1e-6)
    assert rates.p_le8 == pytest.approx(0.2820513, abs=1e-6)
    assert rates.source == "pooled"


def test_pooled_rates_sum_to_one():
    rates = pooling.pooled_rates(123, 45, 67, 0.18, 40.0)
    assert rates.p10 + rates.p9 + rates.p_le8 == pytest.approx(1.0)


def test_no_pop_falls_back_to_prior():
    rates = pooling.pooled_rates(0, 0, 0, 0.45, 40.0)
    assert rates.p10 == 0.45
    assert rates.p9 == pytest.approx(0.275)
    assert rates.p_le8 == pytest.approx(0.275)
    assert rates.source == "prior"


def test_huge_ladder_dominates_prior():
    # 900 gems of 1000: prior barely matters.
    rates = pooling.pooled_rates(900, 80, 20, 0.18, 40.0)
    assert rates.p10 == pytest.approx((900 + 40 * 0.18) / 1040)
    assert rates.p10 > 0.85


def test_tiny_ladder_shrinks_to_prior():
    # 2 gems of 2: far from 1.0, close-ish to the prior.
    rates = pooling.pooled_rates(2, 0, 0, 0.30, 40.0)
    assert rates.p10 == pytest.approx((2 + 12) / 42)
    assert rates.p10 < 0.5
