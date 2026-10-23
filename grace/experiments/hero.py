"""Regenerate the committed README hero figure (the Pareto frontier plot)
from the committed synthetic data.

Usage: python -m grace.experiments.hero [--figure-dir figures]
"""

import argparse
import os

from grace import pipeline, plots, synth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--synth-dir", default="synth")
    args = parser.parse_args(argv)

    cards_df, _truth_df = synth.load(args.synth_dir)
    result = pipeline.run(cards_df)
    os.makedirs(args.figure_dir, exist_ok=True)
    figure_path = os.path.join(args.figure_dir, "frontier.png")
    plots.frontier_plot(result, figure_path)
    print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
