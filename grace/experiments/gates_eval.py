"""Experiment 1 — the gate ladder scored as a classifier.

The synthetic generator plants pathologies at known rows (wrong-card
matches, fake grails, thin outliers, bad raws) plus genuine grails that
the gates must NOT kill. This script runs the DROP/REPAIR ladder against
that planted truth and reports precision AND recall, with a per-pathology
breakdown. Plants are deliberately drawn across the gate thresholds, so
recall measures sensitivity to borderline corruption, not just the easy
cases.

Usage: python -m grace.experiments.gates_eval [--figure-dir figures]
"""

import argparse
import os

from grace import gates as gates_mod
from grace import pipeline, synth
from grace.config import DEFAULT_CONFIG

PATHOLOGY_LABELS = ("wrong_card", "fake_grail", "thin_outlier", "bad_raw")


def evaluate(cards_df, truth_df):
    """Run the ladder; return a metrics dict (all counts and rates)."""
    labels = truth_df["label"].tolist()
    should_flag = truth_df["should_flag"].tolist()
    details = truth_df["detail"].fillna("").astype(str).tolist()
    records = cards_df.to_dict("records")

    dropped_flags = []
    repaired_flags = []
    drop_reasons = []
    for card in records:
        decision = gates_mod.apply_gates(card)
        dropped_flags.append(decision.dropped)
        repaired_flags.append(decision.raw_confidence == "repaired")
        drop_reasons.append(decision.drop_reason or "")

    per_pathology = {}
    for pathology in PATHOLOGY_LABELS:
        planted = 0
        caught = 0
        for index, label in enumerate(labels):
            if label != pathology:
                continue
            planted += 1
            flagged = dropped_flags[index] or repaired_flags[index]
            if flagged:
                caught += 1
        recall = caught / planted if planted else float("nan")
        per_pathology[pathology] = {"planted": planted, "caught": caught, "recall": recall}

    flagged_total = 0
    flagged_true = 0
    dropped_total = 0
    dropped_true = 0
    clean_total = 0
    clean_flagged = 0
    grail_total = 0
    grail_survived = 0
    for index, label in enumerate(labels):
        flagged = dropped_flags[index] or repaired_flags[index]
        if flagged:
            flagged_total += 1
            if should_flag[index]:
                flagged_true += 1
        if dropped_flags[index]:
            dropped_total += 1
            if should_flag[index]:
                dropped_true += 1
        if label == "clean":
            clean_total += 1
            if flagged:
                clean_flagged += 1
        if label == "genuine_grail":
            grail_total += 1
            if not dropped_flags[index]:
                grail_survived += 1

    exempt_grail_total = 0
    exempt_grail_survived = 0
    for index, detail in enumerate(details):
        if "kind=exempt_grail" not in detail:
            continue
        exempt_grail_total += 1
        if not dropped_flags[index]:
            exempt_grail_survived += 1

    reason_by_label = {}
    for index, label in enumerate(labels):
        reason = drop_reasons[index]
        if reason == "":
            continue
        key = (label, reason)
        reason_by_label[key] = reason_by_label.get(key, 0) + 1

    metrics = {
        "per_pathology": per_pathology,
        "flag_precision": flagged_true / flagged_total if flagged_total else float("nan"),
        "drop_precision": dropped_true / dropped_total if dropped_total else float("nan"),
        "flagged_total": flagged_total,
        "dropped_total": dropped_total,
        "clean_total": clean_total,
        "clean_flagged": clean_flagged,
        "grail_total": grail_total,
        "grail_survived": grail_survived,
        "exempt_grail_total": exempt_grail_total,
        "exempt_grail_survived": exempt_grail_survived,
        "reason_by_label": reason_by_label,
    }
    return metrics


def false_positive_cost(cards_df, truth_df, cfg=DEFAULT_CONFIG):
    """What the false positives cost a buyer: shadow NPV on clean drops.

    Dropped cards still get shadow metrics from the pipeline (computed on
    the post-ladder effective raw price), so the NPV a buyer forgoes by
    trusting a false drop is directly readable.
    """
    result = pipeline.run(cards_df, cfg)
    labels = truth_df["label"].tolist()
    statuses = result["status"].tolist()
    npv_values = result["npv"].tolist()

    clean_dropped = 0
    positive_npv_drops = 0
    forgone_npv_total = 0.0
    for index, label in enumerate(labels):
        if label != "clean":
            continue
        if statuses[index] != "dropped":
            continue
        clean_dropped += 1
        if npv_values[index] > 0:
            positive_npv_drops += 1
            forgone_npv_total += float(npv_values[index])
    cost = {
        "clean_dropped": clean_dropped,
        "positive_npv_drops": positive_npv_drops,
        "forgone_npv_total": forgone_npv_total,
    }
    return cost


def clean_flags_by_kind(cards_df, truth_df):
    """Flagged clean cards counted by borderline kind ("other" if none).

    Why: precision is only meaningful because the borderline-legit clean
    cards can be lost; this says which kind the ladder loses.
    """
    details = truth_df["detail"].fillna("").astype(str).tolist()
    labels = truth_df["label"].tolist()
    records = cards_df.to_dict("records")
    counts = {}
    numbered_records = list(enumerate(records))
    for index, card in numbered_records:
        if labels[index] != "clean":
            continue
        decision = gates_mod.apply_gates(card)
        flagged = decision.dropped or decision.raw_confidence == "repaired"
        if not flagged:
            continue
        kind = "other"
        detail = details[index]
        if detail.startswith("borderline_clean"):
            kind_part = detail.split(";")[1]
            kind = kind_part.split("=")[1]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def miss_details(cards_df, truth_df):
    """For each pathology, the plant sizes of the cards the ladder missed.

    wrong_card: V10 / graded-10 sold median; fake_grail and thin_outlier:
    V10; bad_raw: the planted divisor (from the truth detail). Each list is
    sorted. Why: every miss should be a sub-threshold plant; these values
    show it (README prose quotes them, the drift test pins them).
    """
    labels = truth_df["label"].tolist()
    details = truth_df["detail"].fillna("").astype(str).tolist()
    records = cards_df.to_dict("records")
    misses = {}
    for pathology in PATHOLOGY_LABELS:
        misses[pathology] = []
    numbered_records = list(enumerate(records))
    for index, card in numbered_records:
        label = labels[index]
        if label not in misses:
            continue
        decision = gates_mod.apply_gates(card)
        flagged = decision.dropped or decision.raw_confidence == "repaired"
        if flagged:
            continue
        if label == "wrong_card":
            size = float(card["v10"]) / float(card["grade10_sold_median"])
        elif label == "bad_raw":
            divisor_part = details[index].split(";")[0]
            size = float(divisor_part.split("=")[1])
        else:
            size = float(card["v10"])
        misses[label].append(size)
    for pathology in PATHOLOGY_LABELS:
        misses[pathology].sort()
    return misses


def summary_lines(metrics, cost):
    """The printed classifier summary (the README quotes these lines)."""
    lines = ["pathology      planted caught  recall"]
    for name in PATHOLOGY_LABELS:
        entry = metrics["per_pathology"][name]
        lines.append(
            f"{name:<14} {entry['planted']:>7} {entry['caught']:>6} {entry['recall']:>7.2f}"
        )
    lines.append(
        f"flag precision (flagged cards that were planted): {metrics['flag_precision']:.2f}"
    )
    lines.append(
        f"drop precision (dropped cards that were planted): {metrics['drop_precision']:.2f}"
    )
    lines.append(f"clean cards flagged: {metrics['clean_flagged']}/{metrics['clean_total']}")
    lines.append(f"genuine grails kept: {metrics['grail_survived']}/{metrics['grail_total']}")
    lines.append(
        "borderline exempt grails kept: "
        f"{metrics['exempt_grail_survived']}/{metrics['exempt_grail_total']}"
    )
    lines.append(
        f"false-positive cost: {cost['clean_dropped']} clean cards dropped, "
        f"{cost['positive_npv_drops']} with positive shadow NPV, "
        f"${cost['forgone_npv_total']:,.0f} NPV forgone"
    )
    return lines


def make_figure(metrics, out_path):
    import matplotlib.pyplot as plt

    from grace import plots  # noqa: F401  (forces the Agg backend)

    names = list(PATHOLOGY_LABELS)
    recalls = []
    for name in names:
        recalls.append(metrics["per_pathology"][name]["recall"])

    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=100)
    bars = ax.bar(names, recalls, color="#5b7fa6")
    for bar, name in zip(bars, names):
        caught = metrics["per_pathology"][name]["caught"]
        planted = metrics["per_pathology"][name]["planted"]
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{caught}/{planted}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("recall (flagged / planted)")
    precision = metrics["flag_precision"]
    survived = metrics["grail_survived"]
    grails = metrics["grail_total"]
    ax.set_title(
        "Gate ladder as a classifier (synthetic plants): "
        f"flag precision {precision:.2f}, genuine grails kept {survived}/{grails}"
    )
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
    metrics = evaluate(cards_df, truth_df)

    print("gate ladder as a classifier (against planted truth)")
    cost = false_positive_cost(cards_df, truth_df)
    lines = summary_lines(metrics, cost)
    for line in lines:
        print(line)
    print()
    print("drop reasons by planted label:")
    sorted_keys = sorted(metrics["reason_by_label"].keys())
    for key in sorted_keys:
        label, reason = key
        count = metrics["reason_by_label"][key]
        print(f"  {label:<14} {reason:<28} {count}")

    os.makedirs(args.figure_dir, exist_ok=True)
    figure_path = os.path.join(args.figure_dir, "gates_classifier.png")
    make_figure(metrics, figure_path)
    print(f"\nwrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
