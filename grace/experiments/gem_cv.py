"""Experiment 3: are the gem-rate probabilities any good?

5-fold cross-validation over CARDS (one row per card, so no card's ladder
or outcome crosses folds) of seven gem-rate estimators, scored by proper
scoring rules on a fresh simulated submission.

Estimators (p10 for one test card with pop ladder n10 / n_ladder, era e):
  prior_fixed   the engine's fixed era prior mu_e (config.era_priors)
  era_mean      the TRAINING folds' pooled era gem rate
                    mu_hat_e = sum(n10) / sum(n_ladder) over training cards of era e
  raw_rate      n10 / n_ladder, falling back to mu_hat_e for an empty ladder
  pooled_m40    the shipped engine: (n10 + 40 * mu_e) / (n_ladder + 40),
                    exactly pooling.pooled_rates(...).p10 with the fixed prior
  eb_mom        empirical Bayes: (n10 + m_hat_e * mu_hat_e) / (n_ladder + m_hat_e),
                    m_hat_e fit by beta-binomial moments on the training folds
                    (m_estimation.m_estimate); with no measurable
                    overdispersion (m_hat infinite or undefined) it pools
                    fully to mu_hat_e
Two cross arms split eb_mom's change from pooled_m40 into its two parts,
because eb_mom moves BOTH the shrink strength (40 -> m_hat_e) and the
shrink target (mu_e -> mu_hat_e) and a gap between them alone cannot say
which part did the work:
  m40_era_mean  m = 40 toward the training-fold era mean:
                    (n10 + 40 * mu_hat_e) / (n_ladder + 40)   (target only)
  eb_prior      m_hat_e toward the fixed prior:
                    (n10 + m_hat_e * mu_e) / (n_ladder + m_hat_e)   (strength only)
Every shrinkage estimator goes through shrunk_p10(), so m = 0 is the raw
rate and a huge m is the era mean by construction (both are tested).

Targets (one Binomial(SUBMISSION_SIZE, p) outcome per card, seeded):
  next_pop      the next pop submissions: p = the card's observed
                (selection-boosted) gem rate, the rate its ladder was drawn at
  unscreened    an unscreened raw card: p = the card's latent gem rate,
                which the planted selection effect makes every ladder
                overstate (most at low V10)

Scores per test fold (each card's outcome is k gems in n trials):
  log loss  -(k ln p + (n - k) ln(1 - p)) / n, with p clipped to
            [1e-6, 1 - 1e-6]; the clipped predictions are counted per
            estimator and printed, because the raw rate predicts exactly
            0 or 1 for some ladders and the clip, not the model, then
            sets its loss
  Brier     (k (1 - p)^2 + (n - k) p^2) / n, the per-trial squared error
  ECE       expected calibration error over ECE_BINS equal-width bins of
            predicted p on [0, 1]: sum_b (cards_b / cards) *
            |mean predicted_b - mean realized rate_b|, realized rate = k / n
Each metric is reported as mean +/- sample sd over the 5 folds, and as
the PAIRED per-fold difference against pooled_m40 (estimator - m40;
negative means better than the shipped m = 40), with the number of folds
in which the estimator beats m40.

A reference row, true_rate, predicts the target's own true rate. No real
estimator can know it; it is the floor the others are read against, and
the bound test asserts no estimator beats it on mean log loss.

Failure modes: the ladders and the outcomes come from the same generator,
so this scores the estimators on the generator's own world (it says which
estimator fits THIS beta-binomial world, not real pop reports); 5 folds of
~120 cards leave a fold sd that can swamp a small gap, so a gap smaller
than its fold sd is a null and is printed as one. Fitting era means or
m_hat on all cards would leak the test fold (guard:
test_changing_the_test_fold_leaves_the_fit_unchanged).

Usage: python -m grace.experiments.gem_cv
"""

import argparse
import math

import numpy as np

from grace import synth
from grace.config import DEFAULT_CONFIG
from grace.experiments import m_estimation

K_FOLDS = 5
FOLD_SEED = 3003
OUTCOME_SEED = 3011
SUBMISSION_SIZE = 20
CLIP = 1e-6
ECE_BINS = 10
SHIPPED_M = 40.0
BASELINE = "pooled_m40"
ESTIMATORS = (
    "prior_fixed",
    "era_mean",
    "raw_rate",
    "pooled_m40",
    "m40_era_mean",
    "eb_prior",
    "eb_mom",
)
REFERENCE = "true_rate"
ESTIMATOR_LABELS = {
    "prior_fixed": "Fixed era prior",
    "era_mean": "Era mean (training folds)",
    "raw_rate": "Raw rate n10 / n",
    "pooled_m40": "Shipped pooling, m = 40",
    "m40_era_mean": "Cross arm: m = 40 toward the era mean",
    "eb_prior": "Cross arm: m fit by moments, toward the fixed prior",
    "eb_mom": "Empirical Bayes, m fit by moments",
    "true_rate": "True rate (reference, unknowable)",
}
ESTIMATOR_SHORT = {
    "prior_fixed": "fixed prior",
    "era_mean": "era mean",
    "raw_rate": "raw rate",
    "pooled_m40": "m = 40",
    "m40_era_mean": "m = 40 toward the era mean",
    "eb_prior": "moment m toward the fixed prior",
    "eb_mom": "empirical Bayes",
    "true_rate": "true rate",
}
# target name -> truth.csv column holding the rate its outcome is drawn at
TARGETS = {"next_pop": "observed_p10", "unscreened": "latent_p10"}
TARGET_LABELS = {
    "next_pop": "next pop submissions (observed, selection-boosted rate)",
    "unscreened": "unscreened raw cards (latent rate)",
}
METRICS = ("log_loss", "brier", "ece")


def clip_probability(p):
    """p clipped to [CLIP, 1 - CLIP], and whether the clip moved it."""
    low = CLIP
    high = 1.0 - CLIP
    if p < low:
        return low, True
    if p > high:
        return high, True
    return p, False


def binomial_log_loss(p, k, n):
    """Per-trial binomial log loss -(k ln p + (n - k) ln(1 - p)) / n.

    p is clipped first, so a prediction of exactly 0 or 1 costs
    -ln(1e-6) ~ 13.8 per wrong trial instead of infinity. Failure mode: the
    clip, not the model, sets that loss; clipped_count() reports how often.
    """
    clipped, _moved = clip_probability(p)
    gems = k * math.log(clipped)
    misses = (n - k) * math.log(1.0 - clipped)
    return -(gems + misses) / n


def binomial_brier(p, k, n):
    """Per-trial Brier score (k (1 - p)^2 + (n - k) p^2) / n (no clip)."""
    gem_error = k * (1.0 - p) ** 2
    miss_error = (n - k) * p**2
    return (gem_error + miss_error) / n


def expected_calibration_error(predictions, gem_counts, n, bins=ECE_BINS):
    """Equal-width-bin ECE: sum_b w_b |mean pred_b - mean realized_b|.

    Bins split [0, 1] into `bins` equal widths (p = 1 joins the top bin);
    w_b is the share of cards in bin b; realized = k / n. Failure mode: a
    constant predictor lands in one bin and scores only its overall bias,
    so ECE alone cannot rank estimators (log loss and Brier do that).
    """
    n_cards = len(predictions)
    card_indices = range(n_cards)
    totals = {}
    for index in card_indices:
        p = predictions[index]
        bin_id = min(int(p * bins), bins - 1)
        realized = gem_counts[index] / n
        entry = totals.setdefault(bin_id, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += p
        entry[2] += realized
    error = 0.0
    bin_ids = sorted(totals.keys())
    for bin_id in bin_ids:
        count, predicted_sum, realized_sum = totals[bin_id]
        gap = abs(predicted_sum / count - realized_sum / count)
        error += count / n_cards * gap
    return error


def clipped_count(predictions):
    """How many predictions the log-loss clip moves."""
    count = 0
    for p in predictions:
        _clipped, moved = clip_probability(p)
        if moved:
            count += 1
    return count


def shrunk_p10(n10, n_ladder, mu, m):
    """(n10 + m mu) / (n_ladder + m): pooling.pooled_rates' p10 formula.

    Endpoints: m = 0 is the raw rate n10 / n_ladder; an empty ladder, an
    infinite m or an undefined (NaN) m is mu itself (full pooling). Failure
    mode the endpoints guard: m = 0 with an empty ladder is 0 / 0, which
    the scratch prototype returned as NaN. (pooled_rates itself is not
    called: its 9-share term divides by zero at m = 0 for an all-gem
    ladder; a test pins this function to pooled_rates' p10 at m = 40.)
    """
    if n_ladder == 0 or not math.isfinite(m):
        return mu
    numerator = n10 + m * mu
    denominator = n_ladder + m
    return numerator / denominator


def assign_folds(n_cards, k=K_FOLDS, seed=FOLD_SEED):
    """fold[i] in 0..k-1 for card i: a seeded permutation taken mod k.

    Every card lands in exactly one test fold and fold sizes differ by at
    most one.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_cards)
    folds = permutation % k
    return folds


def draw_outcomes(rates, n=SUBMISSION_SIZE, seed=OUTCOME_SEED):
    """One fresh Binomial(n, rate) gem count per card (seeded)."""
    rng = np.random.default_rng(seed)
    rate_array = np.asarray(rates, dtype=float)
    counts = rng.binomial(n, rate_array)
    return counts


def build_frame(cards_df, truth_df):
    """One row per card: era, ladder, both target rates, both outcomes."""
    columns = ["card_id", "latent_p10", "observed_p10"]
    frame = cards_df.merge(truth_df[columns], on="card_id")
    frame = frame.sort_values("card_id").reset_index(drop=True)
    frame["n_ladder"] = frame["n10"] + frame["n9"] + frame["n_le8"]
    target_names = list(TARGETS.keys())
    numbered_targets = list(enumerate(target_names))
    for offset, target in numbered_targets:
        rate_column = TARGETS[target]
        rates = frame[rate_column].to_numpy()
        frame[f"k_{target}"] = draw_outcomes(rates, seed=OUTCOME_SEED + offset)
    return frame


def fit_training_fold(frame, folds, test_fold):
    """Era means and moment m_hat from the cards NOT in test_fold.

    Returns {era: {"mu_hat": ..., "m_hat": ...}}. Only training ladders
    enter, and no outcome does: changing anything in the test fold leaves
    this unchanged (tested).
    """
    training_mask = folds != test_fold
    training = frame[training_mask]
    fit = {}
    era_values = training["era"].tolist()
    eras = sorted(set(era_values))
    for era in eras:
        rows = training[training["era"] == era]
        ladder_sizes = rows["n_ladder"].tolist()
        gem_counts = rows["n10"].tolist()
        total_ladder = sum(ladder_sizes)
        total_gems = sum(gem_counts)
        mu_hat = total_gems / total_ladder
        m_hat = m_estimation.m_estimate(gem_counts, ladder_sizes)
        fit[era] = {"mu_hat": float(mu_hat), "m_hat": float(m_hat)}
    return fit


def predict(card, fit, target, cfg=DEFAULT_CONFIG):
    """{estimator: p10} for one test card, plus the true_rate reference."""
    era = card["era"]
    n10 = int(card["n10"])
    n_ladder = int(card["n_ladder"])
    prior = cfg.era_priors[era]
    mu_hat = fit[era]["mu_hat"]
    m_hat = fit[era]["m_hat"]
    predictions = {
        "prior_fixed": prior,
        "era_mean": mu_hat,
        "raw_rate": shrunk_p10(n10, n_ladder, mu_hat, 0.0),
        "pooled_m40": shrunk_p10(n10, n_ladder, prior, SHIPPED_M),
        "m40_era_mean": shrunk_p10(n10, n_ladder, mu_hat, SHIPPED_M),
        "eb_prior": shrunk_p10(n10, n_ladder, prior, m_hat),
        "eb_mom": shrunk_p10(n10, n_ladder, mu_hat, m_hat),
        REFERENCE: float(card[TARGETS[target]]),
    }
    return predictions


def score_fold(frame, folds, test_fold, target, cfg=DEFAULT_CONFIG):
    """{estimator: {metric: value, "clipped": n}} on one test fold."""
    fit = fit_training_fold(frame, folds, test_fold)
    test = frame[folds == test_fold]
    records = test.to_dict("records")
    names = [*ESTIMATORS, REFERENCE]
    predictions = {}
    for name in names:
        predictions[name] = []
    gem_counts = []
    for card in records:
        card_predictions = predict(card, fit, target, cfg)
        for name in names:
            predictions[name].append(card_predictions[name])
        gem_counts.append(int(card[f"k_{target}"]))
    scores = {}
    n = SUBMISSION_SIZE
    for name in names:
        plist = predictions[name]
        log_losses = []
        briers = []
        numbered = list(enumerate(plist))
        for index, p in numbered:
            log_losses.append(binomial_log_loss(p, gem_counts[index], n))
            briers.append(binomial_brier(p, gem_counts[index], n))
        scores[name] = {
            "log_loss": float(np.mean(log_losses)),
            "brier": float(np.mean(briers)),
            "ece": expected_calibration_error(plist, gem_counts, n),
            "clipped": clipped_count(plist),
        }
    return scores, fit


def mean_sd(values):
    """{"mean", "sd"} with the sample sd (ddof = 1) over folds."""
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def run_target(frame, folds, target, cfg=DEFAULT_CONFIG):
    """Fold scores aggregated: mean +/- sd and paired diff vs m = 40."""
    per_fold = []
    fits = []
    fold_ids = range(K_FOLDS)
    for test_fold in fold_ids:
        scores, fit = score_fold(frame, folds, test_fold, target, cfg)
        per_fold.append(scores)
        fits.append(fit)
    names = [*ESTIMATORS, REFERENCE]
    summary = {}
    for name in names:
        entry = {"clipped": 0}
        for fold_scores in per_fold:
            entry["clipped"] += fold_scores[name]["clipped"]
        for metric in METRICS:
            values = []
            diffs = []
            better = 0
            for fold_scores in per_fold:
                value = fold_scores[name][metric]
                baseline = fold_scores[BASELINE][metric]
                values.append(value)
                diffs.append(value - baseline)
                if value < baseline:
                    better += 1
            entry[metric] = mean_sd(values)
            paired = mean_sd(diffs)
            paired["folds_better"] = better
            entry[f"{metric}_diff_vs_m40"] = paired
        summary[name] = entry
    result = {"estimators": summary, "per_fold": per_fold}
    return result, fits


def fold_fit_ranges(fits):
    """{era: {"m_hat_min", "m_hat_max", "mu_hat_min", "mu_hat_max"}} over folds."""
    ranges = {}
    eras = sorted(fits[0].keys())
    for era in eras:
        m_values = []
        mu_values = []
        for fit in fits:
            m_values.append(fit[era]["m_hat"])
            mu_values.append(fit[era]["mu_hat"])
        ranges[era] = {
            "m_hat_min": min(m_values),
            "m_hat_max": max(m_values),
            "mu_hat_min": min(mu_values),
            "mu_hat_max": max(mu_values),
        }
    return ranges


def best_estimator(estimators, metric="log_loss"):
    """The real estimator (never the reference) with the lowest mean metric."""
    best = None
    best_value = math.inf
    for name in ESTIMATORS:
        value = estimators[name][metric]["mean"]
        if value < best_value:
            best = name
            best_value = value
    return best


def is_null(paired):
    """A paired gap is a null when its |mean| is under one fold sd."""
    return abs(paired["mean"]) < paired["sd"]


def verdict(paired):
    """'null', 'better' or 'worse' than m = 40 for a paired difference.

    Rule: a gap under one fold sd is a null whatever its sign; otherwise a
    negative (estimator - m40) gap is 'better'. Failure mode: with 5 folds
    the sd is itself noisy, so the folds-better count is printed beside it.
    """
    if is_null(paired):
        return "null"
    if paired["mean"] < 0:
        return "better"
    return "worse"


def compute_summary(cards_df, truth_df, cfg=DEFAULT_CONFIG):
    """Every number main() prints, as a dict (the report copies it)."""
    frame = build_frame(cards_df, truth_df)
    folds = assign_folds(len(frame))
    fold_sizes = []
    fold_ids = range(K_FOLDS)
    for test_fold in fold_ids:
        fold_sizes.append(int((folds == test_fold).sum()))
    empty_ladders = int((frame["n_ladder"] == 0).sum())
    targets = {}
    fit_ranges = None
    for target in TARGETS:
        result, fits = run_target(frame, folds, target, cfg)
        estimators = result["estimators"]
        best = best_estimator(estimators)
        verdicts = {}
        for name in ESTIMATORS:
            if name == BASELINE:
                continue
            verdicts[name] = verdict(estimators[name]["log_loss_diff_vs_m40"])
        targets[target] = {
            "estimators": estimators,
            "best_by_log_loss": best,
            "log_loss_verdict_vs_m40": verdicts,
        }
        fit_ranges = fold_fit_ranges(fits)  # the fit ignores outcomes: same per target
    summary = {
        "settings": {
            "k_folds": K_FOLDS,
            "fold_seed": FOLD_SEED,
            "outcome_seed": OUTCOME_SEED,
            "submission_size": SUBMISSION_SIZE,
            "clip": CLIP,
            "ece_bins": ECE_BINS,
            "estimator_labels": ESTIMATOR_LABELS,
            "estimator_short": ESTIMATOR_SHORT,
            "target_labels": TARGET_LABELS,
        },
        "n_cards": len(frame),
        "empty_ladders": empty_ladders,
        "fold_sizes": fold_sizes,
        "training_fit_ranges": fit_ranges,
        "targets": targets,
    }
    return summary


def metric_text(entry):
    return f"{entry['mean']:.4f} +/- {entry['sd']:.4f}"


def paired_text(entry):
    return f"{entry['mean']:+.4f} +/- {entry['sd']:.4f} ({entry['folds_better']}/{K_FOLDS})"


def summary_lines(summary):
    """The printed Experiment 3 summary (the report renders the same rows)."""
    lines = [
        f"cards: {summary['n_cards']} ({summary['empty_ladders']} empty ladders), "
        f"{K_FOLDS} folds over cards, outcome = {SUBMISSION_SIZE}-card submission",
    ]
    ranges = summary["training_fit_ranges"]
    eras = sorted(ranges.keys())
    for era in eras:
        entry = ranges[era]
        lines.append(
            f"  {era:<8} training-fold m_hat {entry['m_hat_min']:.1f}-{entry['m_hat_max']:.1f}"
        )
    for target in TARGETS:
        section = summary["targets"][target]
        lines.append("")
        lines.append(f"target: {target}")
        lines.append(
            f"  {'estimator':<12} {'log loss':>20} {'diff vs m40 (folds better)':>30} "
            f"{'brier':>20} {'ece':>20} clipped"
        )
        names = [*ESTIMATORS, REFERENCE]
        for name in names:
            entry = section["estimators"][name]
            lines.append(
                f"  {name:<12} {metric_text(entry['log_loss']):>20} "
                f"{paired_text(entry['log_loss_diff_vs_m40']):>30} "
                f"{metric_text(entry['brier']):>20} {metric_text(entry['ece']):>20} "
                f"{entry['clipped']}"
            )
        lines.append(f"  best real estimator by log loss: {section['best_by_log_loss']}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-dir", default="synth")
    args = parser.parse_args(argv)
    cards_df, truth_df = synth.load(args.synth_dir)
    summary = compute_summary(cards_df, truth_df)
    lines = summary_lines(summary)
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
