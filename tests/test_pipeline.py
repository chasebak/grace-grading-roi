"""End-to-end pipeline behavior on the synthetic set."""

import numpy as np

from grace import pipeline, synth
from grace.config import EngineConfig


def run_default():
    cards, truth = synth.generate()
    result = pipeline.run(cards)
    return cards, truth, result


def test_every_card_gets_a_row():
    cards, _truth, result = run_default()
    assert len(result) == len(cards)
    assert set(result["status"].unique()) <= {"kept", "dropped"}


def test_dropped_cards_never_rank():
    _cards, _truth, result = run_default()
    dropped = result[result["status"] == "dropped"]
    assert (~dropped["rankable"]).all()
    assert (~dropped["on_frontier"]).all()


def test_frontier_only_among_rankable():
    _cards, _truth, result = run_default()
    frontier = result[result["on_frontier"]]
    assert len(frontier) > 0
    assert frontier["rankable"].all()
    assert (frontier["status"] == "kept").all()


def test_frontier_is_non_dominated():
    _cards, _truth, result = run_default()
    rankable = result[(result["status"] == "kept") & result["rankable"]]
    frontier = rankable[rankable["on_frontier"]]
    raws = rankable["raw_effective"].to_numpy()
    npvs = rankable["npv"].to_numpy()
    frontier_raws = frontier["raw_effective"].to_numpy()
    frontier_npvs = frontier["npv"].to_numpy()
    for point_index in range(len(frontier_raws)):
        cheaper_or_equal = raws <= frontier_raws[point_index]
        strictly_better = npvs > frontier_npvs[point_index]
        dominated = bool(np.any(cheaper_or_equal & strictly_better))
        assert not dominated


def test_rankable_cards_meet_gates():
    _cards, _truth, result = run_default()
    rankable = result[result["rankable"]]
    assert (rankable["npv"] >= 25.0).all()


def test_prior_source_flagged_for_no_pop_cards():
    cards, _truth, result = run_default()
    no_pop = (cards["n10"] + cards["n9"] + cards["n_le8"]) == 0
    flagged = result.loc[no_pop.to_numpy(), "rate_source"]
    assert (flagged == "prior").all()


def test_lambda_penalty_changes_scores_not_npv():
    cards, _truth = synth.generate(n_cards=100, seed=5)
    base = pipeline.run(cards)
    penalized = pipeline.run(cards, EngineConfig(lam=0.5))
    kept = (base["status"] == "kept") & (base["sigma"] > 0)
    assert np.allclose(base["npv"], penalized["npv"])
    assert (penalized.loc[kept, "grace"] < base.loc[kept, "grace"]).all()


def test_probabilities_sum_to_one():
    _cards, _truth, result = run_default()
    total = result["p10"] + result["p9"] + result["p_le8"]
    assert np.allclose(total, 1.0)
