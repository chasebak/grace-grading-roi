"""Demo: score the committed synthetic cards, write the frontier plot and
a top-N table to out/ (untracked), print a summary.

Usage: python -m grace.demo [--out OUT_DIR] [--synth-dir SYNTH_DIR] [--top N]
"""

import argparse
import os
import time

from grace import pipeline, plots, synth


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
    dropped = result[result["status"] == "dropped"]
    repaired = kept[kept["raw_confidence"] == "repaired"]
    rankable = kept[kept["rankable"]]
    frontier = kept[kept["on_frontier"]]

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

    print(f"cards scored:    {len(result)}")
    print(f"kept:            {len(kept)}")
    print(f"dropped:         {len(dropped)}")
    print(f"repaired raws:   {len(repaired)}")
    print(f"rankable:        {len(rankable)}")
    print(f"on frontier:     {len(frontier)}")
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
