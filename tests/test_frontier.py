"""Frontier correctness on small hand-built sets."""

import pytest

from grace.frontier import pareto_frontier


def test_simple_staircase():
    #  (raw, npv): A(1, 10) B(2, 5) C(3, 20) D(4, 20) E(5, 30)
    #  B dominated by A; D dominated by C (same npv, cheaper); A, C, E on it
    raw = [1.0, 2.0, 3.0, 4.0, 5.0]
    npv = [10.0, 5.0, 20.0, 20.0, 30.0]
    assert pareto_frontier(raw, npv) == [True, False, True, False, True]


def test_single_point():
    assert pareto_frontier([7.0], [1.0]) == [True]


def test_empty():
    assert pareto_frontier([], []) == []


def test_ties_on_raw_keep_only_highest_npv():
    raw = [2.0, 2.0, 2.0]
    npv = [5.0, 9.0, 7.0]
    assert pareto_frontier(raw, npv) == [False, True, False]


def test_all_dominated_by_first():
    raw = [1.0, 2.0, 3.0]
    npv = [50.0, 40.0, 10.0]
    assert pareto_frontier(raw, npv) == [True, False, False]


def test_negative_npvs_allowed():
    # the cheapest card is always non-dominated (nothing cheaper beats it)
    raw = [1.0, 2.0]
    npv = [-5.0, -1.0]
    assert pareto_frontier(raw, npv) == [True, True]


def test_order_independent_of_input_order():
    raw = [5.0, 1.0, 3.0]
    npv = [30.0, 10.0, 20.0]
    assert pareto_frontier(raw, npv) == [True, True, True]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        pareto_frontier([1.0], [1.0, 2.0])
