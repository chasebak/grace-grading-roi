"""Hand-checked NPV formulas.

The expected values below were computed by hand (calculator arithmetic),
not by calling the code under test.
"""

import math

import pytest

from grace import npv
from grace.config import DEFAULT_CONFIG, EngineConfig, Tier

# Shared worked example:
#   p = (0.5, 0.3, 0.2), V = (1000, 400, 50), f = 0.13, s = 5
#   leg10 = 0.5 * (870 - 5)  = 432.50
#   leg9  = 0.3 * (348 - 5)  = 102.90
#   leg8  = 0.2 * (43.5 - 5) =   7.70
#   Q = 543.10
P10, P9, PLE8 = 0.5, 0.3, 0.2
V10, V9, VLE8 = 1000.0, 400.0, 50.0
FEE = 0.13
SHIP = 5.0
Q_EXPECTED = 543.10


def test_expected_net_sale_hand_check():
    q = npv.expected_net_sale(P10, P9, PLE8, V10, V9, VLE8, FEE, SHIP)
    assert q == pytest.approx(Q_EXPECTED, abs=1e-9)


def test_capital_hand_check():
    # R = 2 * (50 + 79.99) + 10 = 269.98
    r = npv.capital(2, 50.0, 79.99, 10.0)
    assert r == pytest.approx(269.98, abs=1e-9)


def test_discount_factor_hand_check():
    # U = 1 / 1.12^(2.7/12) = 0.9748243 (hand: exp(-0.225 * ln 1.12))
    u = npv.discount_factor(2.7, 0.12)
    assert u == pytest.approx(0.9748243, abs=1e-6)


def test_npv_hand_check():
    # NPV = 543.10 * 2 * 0.9748243 - 269.98 = 788.873
    u = npv.discount_factor(2.7, 0.12)
    value = npv.npv(Q_EXPECTED, 2, u, 269.98)
    assert value == pytest.approx(788.873, abs=0.01)


def test_annualized_roi_hand_check():
    # pv/R = 1058.853/269.98 = 3.92197; 3.92197^(12/2.7) - 1 = 433.3 (huge)
    u = npv.discount_factor(2.7, 0.12)
    roi = npv.annualized_roi(Q_EXPECTED, 2, u, 269.98, 2.7)
    assert roi == pytest.approx(432.6, rel=0.01)


def test_annualized_roi_total_loss_and_bad_capital():
    assert npv.annualized_roi(-10.0, 1, 1.0, 100.0, 2.0) == -1.0
    assert math.isnan(npv.annualized_roi(10.0, 1, 1.0, 0.0, 2.0))


def test_raw_net_now_floor_at_zero():
    # 10 * 0.87 - 5 = 3.7 ; 2 * 0.87 - 5 < 0 -> 0
    assert npv.raw_net_now(10.0, FEE, SHIP) == pytest.approx(3.7)
    assert npv.raw_net_now(2.0, FEE, SHIP) == 0.0


def test_npv_marginal_hand_check():
    # npv=100, raw=50, raw_net = 50*0.87-5 = 38.5 -> 100 + (50-38.5) = 111.5
    raw_net = npv.raw_net_now(50.0, FEE, SHIP)
    assert npv.npv_marginal(100.0, 50.0, raw_net) == pytest.approx(111.5)


def test_resolve_legs_v9_fallback_and_le8_default():
    v9, v_le8, synthetic = npv.resolve_legs(100.0, None, 12.0)
    assert v9 == 50.0 and synthetic is True
    v9, v_le8, synthetic = npv.resolve_legs(100.0, float("nan"), 12.0)
    assert v9 == 50.0 and synthetic is True
    v9, v_le8, synthetic = npv.resolve_legs(100.0, 42.0, 12.0)
    assert v9 == 42.0 and synthetic is False
    assert v_le8 == 12.0  # a sub-9 resells at ~raw


def test_tier_argmax_picks_highest_npv():
    # Two synthetic tiers: an expensive-fast one only wins when the fee
    # difference is beaten by the discounting; make the slow tier terrible.
    tiers = (Tier("slow", 10.0, 24.0), Tier("fast", 11.0, 0.1))
    cfg = EngineConfig(tiers=tiers, discount_rate_annual=0.9)
    results, best = npv.evaluate_tiers(0.5, 0.3, 0.2, 500.0, 200.0, 20.0, 10.0, 1, cfg)
    assert len(results) == 2
    assert best.tier_name == "fast"
    assert best.npv == max(results[0].npv, results[1].npv)


def test_roi_display_cap_applied():
    cfg = DEFAULT_CONFIG
    results, _best = npv.evaluate_tiers(0.9, 0.05, 0.05, 50000.0, 20000.0, 100.0, 10.0, 1, cfg)
    for result in results:
        assert result.roi_display <= cfg.roi_display_cap
