"""Experiment 2 — is the fixed pooling pseudo-count m=40 reasonable?

Three parts:

1. Beta-binomial method-of-moments estimate of the pseudo-count from the
   overdispersion of the synthetic pop ladders, per era. Model: each
   card's gem count is Binomial(n_i, p_i) with p_i ~ Beta(alpha, beta);
   the implied pooling pseudo-count is M = alpha + beta. Moments:
       rho = (s2 / (mu*(1-mu)) - mean(1/n)) / (1 - mean(1/n))
       M   = 1/rho - 1
   where mu is the mean observed gem rate, s2 the across-card variance
   of observed gem rates, and rho the overdispersion correlation.

2. Calibration: pooled p10 predictions versus realized gem rates on
   held-out synthetic cards (realized = a fresh simulated 20-card
   submission at each card's LATENT true gem rate). The planted
   submission-selection effect should show up as predictions sitting
   above the realized rates, most strongly for low-value cards.

3. Prior sensitivity: shift every era prior by +/-0.10 and measure the
   Spearman rank correlation of the GRACE ranking against the default
   priors (supports the "illustrative era base rates,
   sensitivity-analyzed" label).

Usage: python -m grace.experiments.m_estimation [--figure-dir figures]
"""

import argparse
import dataclasses
import os

import numpy as np

from grace import pipeline, pooling, synth
from grace.config import DEFAULT_CONFIG

MIN_LADDER_FOR_MOMENTS = 5
HOLDOUT_FRACTION = 0.3
HOLDOUT_SEED = 1007
FRESH_SUBMISSION_SIZE = 20
CALIBRATION_BINS = 10
PRIOR_SHIFTS = (-0.10, 0.10)


def m_estimate(gem_counts, ladder_sizes):
    """Beta-binomial method-of-moments pseudo-count from one era's ladders."""
    rates = []
    inverse_sizes = []
    for gems, size in zip(gem_counts, ladder_sizes):
        if size < MIN_LADDER_FOR_MOMENTS:
            continue
        rates.append(gems / size)
        inverse_sizes.append(1.0 / size)
    if len(rates) < 3:
        return float("nan")
    mu = float(np.mean(rates))
    s2 = float(np.var(rates, ddof=1))
    mean_inverse = float(np.mean(inverse_sizes))
    denominator = mu * (1.0 - mu)
    if denominator <= 0:
        return float("nan")
    rho = (s2 / denominator - mean_inverse) / (1.0 - mean_inverse)
    if rho <= 0:
        return float("inf")  # no detectable overdispersion
    return 1.0 / rho - 1.0


def m_estimates_by_era(cards_df):
    """Return {era: m_hat} from the observed pop ladders."""
    estimates = {}
    era_values = cards_df["era"].tolist()
    unique_eras = sorted(set(era_values))
    for era in unique_eras:
        era_rows = cards_df[cards_df["era"] == era]
        ladder_sizes = (era_rows["n10"] + era_rows["n9"] + era_rows["n_le8"]).tolist()
        gem_counts = era_rows["n10"].tolist()
        estimates[era] = m_estimate(gem_counts, ladder_sizes)
    return estimates


def calibration_table(cards_df, truth_df, cfg=DEFAULT_CONFIG):
    """Predicted pooled p10 vs realized gem rate on held-out cards.

    Returns (per-card DataFrame, binned DataFrame).
    """
    import pandas as pd

    merged = cards_df.merge(truth_df[["card_id", "latent_p10"]], on="card_id")
    holdout_rng = np.random.default_rng(HOLDOUT_SEED)
    n_rows = len(merged)
    holdout_mask = holdout_rng.uniform(size=n_rows) < HOLDOUT_FRACTION

    records = merged.to_dict("records")
    rows = []
    for index, card in enumerate(records):
        if not holdout_mask[index]:
            continue
        mu_era = cfg.era_priors[card["era"]]
        rates = pooling.pooled_rates(
            int(card["n10"]), int(card["n9"]), int(card["n_le8"]), mu_era, cfg.m
        )
        fresh_gems = holdout_rng.binomial(FRESH_SUBMISSION_SIZE, float(card["latent_p10"]))
        realized = fresh_gems / FRESH_SUBMISSION_SIZE
        row = {
            "card_id": card["card_id"],
            "era": card["era"],
            "v10": float(card["v10"]),
            "predicted_p10": rates.p10,
            "realized_gem_rate": realized,
        }
        rows.append(row)
    per_card = pd.DataFrame(rows)

    value_median = per_card["v10"].median()
    value_groups = []
    for v10 in per_card["v10"].tolist():
        if v10 <= value_median:
            value_groups.append("low_value")
        else:
            value_groups.append("high_value")
    per_card["value_group"] = value_groups

    quantile_edges = np.quantile(
        per_card["predicted_p10"], np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    )
    quantile_edges[0] = quantile_edges[0] - 1e-9
    bin_ids = np.searchsorted(quantile_edges, per_card["predicted_p10"], side="left") - 1
    bin_ids = np.clip(bin_ids, 0, CALIBRATION_BINS - 1)
    per_card["bin"] = bin_ids

    binned = (
        per_card.groupby("bin")
        .agg(
            mean_predicted=("predicted_p10", "mean"),
            mean_realized=("realized_gem_rate", "mean"),
            count=("card_id", "size"),
        )
        .reset_index()
    )
    return per_card, binned


def prior_sensitivity(cards_df, cfg=DEFAULT_CONFIG):
    """Spearman rank correlation of GRACE under shifted era priors."""
    base_result = pipeline.run(cards_df, cfg)
    base_kept = base_result[(base_result["status"] == "kept") & base_result["rankable"]]
    base_grace = base_kept.set_index("card_id")["grace"]

    correlations = {}
    for shift in PRIOR_SHIFTS:
        shifted_priors = {}
        for era, mu in cfg.era_priors.items():
            shifted_priors[era] = min(max(mu + shift, 0.02), 0.95)
        # dataclasses.replace keeps every other field of cfg; building a
        # fresh EngineConfig here would silently reset them to defaults.
        shifted_cfg = dataclasses.replace(cfg, era_priors=shifted_priors)
        shifted_result = pipeline.run(cards_df, shifted_cfg)
        shifted_kept = shifted_result[
            (shifted_result["status"] == "kept") & shifted_result["rankable"]
        ]
        shifted_grace = shifted_kept.set_index("card_id")["grace"]
        common = base_grace.index.intersection(shifted_grace.index)
        base_ranks = base_grace.loc[common].rank()
        shifted_ranks = shifted_grace.loc[common].rank()
        correlations[shift] = float(base_ranks.corr(shifted_ranks))
    return correlations


def make_figure(per_card, binned, out_path):
    import matplotlib.pyplot as plt

    from grace import plots  # noqa: F401  (forces the Agg backend)

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    low = per_card[per_card["value_group"] == "low_value"]
    high = per_card[per_card["value_group"] == "high_value"]
    ax.scatter(
        low["predicted_p10"],
        low["realized_gem_rate"],
        s=12,
        alpha=0.35,
        color="#e08b2d",
        label="held-out cards, low value",
    )
    ax.scatter(
        high["predicted_p10"],
        high["realized_gem_rate"],
        s=12,
        alpha=0.35,
        color="#5b7fa6",
        label="held-out cards, high value",
    )
    ax.plot(
        binned["mean_predicted"],
        binned["mean_realized"],
        marker="o",
        color="#1f4e79",
        linewidth=1.8,
        label="bin means (predicted deciles)",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linewidth=0.9, linestyle="--")
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(0.0, 0.9)
    ax.set_xlabel("predicted p10 (partial-pooled, m=40)")
    ax.set_ylabel("realized gem rate (fresh simulated submission)")
    ax.set_title("Calibration under the planted submission-selection effect")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--synth-dir", default="synth")
    args = parser.parse_args(argv)

    cards_df, truth_df = synth.load(args.synth_dir)

    estimates = m_estimates_by_era(cards_df)
    print("beta-binomial method-of-moments pseudo-count (vs fixed m=40):")
    for era in sorted(estimates.keys()):
        print(f"  {era:<8} m_hat = {estimates[era]:.1f}")

    per_card, binned = calibration_table(cards_df, truth_df)
    gap = float((per_card["predicted_p10"] - per_card["realized_gem_rate"]).mean())
    low = per_card[per_card["value_group"] == "low_value"]
    high = per_card[per_card["value_group"] == "high_value"]
    low_gap = float((low["predicted_p10"] - low["realized_gem_rate"]).mean())
    high_gap = float((high["predicted_p10"] - high["realized_gem_rate"]).mean())
    print(f"held-out cards: {len(per_card)}")
    print(f"mean (predicted - realized) gem rate: {gap:+.3f}")
    print(f"  low-value cards:  {low_gap:+.3f}")
    print(f"  high-value cards: {high_gap:+.3f}")

    correlations = prior_sensitivity(cards_df)
    print("prior sensitivity (Spearman of GRACE ranking vs default priors):")
    for shift in sorted(correlations.keys()):
        print(f"  era priors {shift:+.2f}: rho = {correlations[shift]:.3f}")

    os.makedirs(args.figure_dir, exist_ok=True)
    figure_path = os.path.join(args.figure_dir, "calibration.png")
    make_figure(per_card, binned, figure_path)
    print(f"\nwrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
