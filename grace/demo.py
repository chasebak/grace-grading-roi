"""Demo: score the committed synthetic cards, write the frontier plot and
a top-N table to out/ (untracked), print a summary.

Usage: python -m grace.demo [--out OUT_DIR] [--synth-dir SYNTH_DIR] [--top N]
"""

import argparse
import os
import time

from grace import pipeline, plots, synth


def summary_counts(result):
    """The card counts the demo prints (the report copies them)."""
    kept = result[result["status"] == "kept"]
    counts = {
        "cards_scored": len(result),
        "kept": len(kept),
        "dropped": int((result["status"] == "dropped").sum()),
        "repaired_raws": int((kept["raw_confidence"] == "repaired").sum()),
        "rankable": int(kept["rankable"].sum()),
        "on_frontier": int(kept["on_frontier"].sum()),
    }
    return counts


def summary_lines(counts):
    """The printed count lines (the README quotes these lines)."""
    lines = [
        f"cards scored:    {counts['cards_scored']}",
        f"kept:            {counts['kept']}",
        f"dropped:         {counts['dropped']}",
        f"repaired raws:   {counts['repaired_raws']}",
        f"rankable:        {counts['rankable']}",
        f"on frontier:     {counts['on_frontier']}",
    ]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out", help="output directory (default: out)")
    parser.add_argument("--synth-dir", default="synth", help="committed synthetic data dir")
    parser.add_argument("--top", type=int, default=15, help="rows in the top table")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cards_df, _truth_df = synth.load(args.synth_dir)
    result = pipeline.run(cards_df)

    os.makedirs(args.out, exist_ok=True)
    plot_path = os.path.join(args.out, "frontier.png")
    plots.frontier_plot(result, plot_path)

    kept = result[result["status"] == "kept"]
    rankable = kept[kept["rankable"]]

    top_columns = [
        "card_id",
        "era",
        "raw_effective",
        "best_tier",
        "npv",
        "npv_marginal",
        "sigma",
        "grace",
        "p10",
        "on_frontier",
    ]
    top = rankable.sort_values("grace", ascending=False).head(args.top)
    top_path = os.path.join(args.out, "top_cards.csv")
    top[top_columns].to_csv(top_path, index=False, float_format="%.4g")

    counts = summary_counts(result)
    lines = summary_lines(counts)
    for line in lines:
        print(line)
    print()
    print(f"top {args.top} by GRACE (lambda=0, a=0.5 -> NPV/sqrt(capital)):")
    print(top[top_columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"wrote {plot_path} and {top_path}")
    elapsed = time.perf_counter() - started
    print(f"demo finished in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
