"""Shared plotting helpers (headless backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # must come after the backend is set

FIGSIZE = (9.0, 5.5)  # 900x550 px at dpi=100 keeps the README hero mobile-safe
DPI = 100


def frontier_plot(result_df, out_path, title="Grade-or-not Pareto frontier (synthetic data)"):
    """Scatter of raw price vs NPV with dropped/repaired/frontier marked."""
    kept = result_df[result_df["status"] == "kept"]
    dropped = result_df[result_df["status"] == "dropped"]
    repaired = kept[kept["raw_confidence"] == "repaired"]
    unrankable = kept[~kept["rankable"]]
    rankable = kept[kept["rankable"]]
    frontier = kept[kept["on_frontier"]].sort_values("raw_effective")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.scatter(
        unrankable["raw_effective"],
        unrankable["npv"],
        s=12,
        color="#c9c9c9",
        label="kept, not rankable",
        zorder=1,
    )
    ax.scatter(
        rankable["raw_effective"],
        rankable["npv"],
        s=16,
        color="#5b7fa6",
        alpha=0.75,
        label="rankable",
        zorder=2,
    )
    ax.scatter(
        repaired["raw_effective"],
        repaired["npv"],
        s=46,
        marker="^",
        facecolors="none",
        edgecolors="#e08b2d",
        linewidths=1.3,
        label="raw price repaired",
        zorder=3,
    )
    ax.scatter(
        dropped["raw_effective"],
        dropped["npv"],
        s=28,
        marker="x",
        color="#c23b3b",
        label="dropped by gates",
        zorder=3,
    )
    ax.step(
        frontier["raw_effective"],
        frontier["npv"],
        where="post",
        color="#1f4e79",
        linewidth=1.8,
        zorder=4,
    )
    ax.scatter(
        frontier["raw_effective"],
        frontier["npv"],
        s=42,
        color="#1f4e79",
        label="Pareto frontier",
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("raw price, USD (log scale)")
    ax.set_ylabel("NPV at best tier, USD")
    ax.set_title(title)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
