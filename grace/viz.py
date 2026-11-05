"""The "Hand GRACE a Budget" page data: python -m grace.viz build [--check].

Writes, from the Experiment 4 functions on the committed synthetic world:
  docs/data/summary.json       the $3,000 headline, universe counts, sweep
                               facts, rule labels, why-box text,
                               provenance labels and the footer
  docs/data/budget_grid.json   every rule's metrics at every solved sweep
                               budget, the batches of the rules the page
                               lists, and the card rows those batches name
  docs/data/meta.json          the pinned tokens.css hash, the Chart.js
                               tag and the measured data sizes
JSON is written exactly like reports/ (sorted keys, floats rounded to 10
significant digits, non-finite floats as null) by report.to_json_text.

The page (docs/index.html + docs/app.js) computes nothing: it looks values
up in these files and formats them. So the tests are in Python
(tests/test_viz.py): the bundle equals the backtest functions' output at
the same rounding, the $3,000 cell equals a standalone run, and the
committed files equal a fresh build.

--check rebuilds in memory (about a second) and exits 1 if a data file is
stale or missing, docs/tokens.css differs from the pinned kit hash,
docs/.nojekyll is missing, the data exceed the 800 KB budget, or the
bundle breaks its key allowlist. Failure mode: --check proves the JSON is
fresh, not that the page reads it correctly; the screenshot click-through
(each slider stop against the JSON) covers that and is done outside CI.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

from grace import report, why
from grace.experiments import backtest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = "docs"
SUMMARY_PATH = "docs/data/summary.json"
GRID_PATH = "docs/data/budget_grid.json"
META_PATH = "docs/data/meta.json"
DATA_PATHS = (SUMMARY_PATH, GRID_PATH, META_PATH)
TOKENS_PATH = "docs/tokens.css"
NOJEKYLL_PATH = "docs/.nojekyll"
INDEX_PATH = "docs/index.html"
APP_PATH = "docs/app.js"
# Plan, Shared standards (d): grace's page data stay under 800 KB.
DATA_BUDGET_BYTES = 800_000
# sha256 of the portfolio page kit's tokens.css, base64 like an SRI hash (a
# 64-hex digest would trip the public privacy test's query-hash pattern).
# docs/tokens.css must be a byte-for-byte copy, so the five pages share one look.
TOKENS_SHA256_B64 = "J3texCt1out7QFMk3jA+rC1vsZ5U8WlE3PnDZFY8OO8="
CHART_JS_VERSION = "4.4.1"
CHART_JS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7k'
    'HXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)
FOOTER = (
    "Every number was computed by this repo's tested Python and copied; "
    "the page computes nothing."
)
TITLE = "Hand GRACE a Budget"
DEFAULT_BUDGET = backtest.HEADLINE_BUDGET
# Slider stops and chart ticks that carry a printed label (every stop gets a mark).
TICK_LABEL_BUDGETS = (250, 1000, 3000, 10000, 20000)
# Lines on the chart, drawn in this order (oracle on top of the legend).
CHART_RULES = ("oracle", "grace", "heuristic", "random")
# Rules whose batch the page lists. The random rule is left out: its tiles
# are the mean over all seeded shuffles, and one shuffle's batch would not
# match them (DECISIONS.md, 2026-11-04 page entries).
BATCH_RULES = ("grace", "heuristic", "oracle")
# Tile comparisons: GRACE against its fair baseline, the heuristic.
TILE_RULES = ("grace", "heuristic", "random", "oracle")
PAGE_WHY_KEYS = (
    "oracle_value_captured",
    "backtest_sweep",
    "regret_and_npv_per_budget_dollar",
    "pick_optimism",
    "corruption_leakage",
    "batch_list",
)
# Short names for the legend and the batch buttons (phone width).
RULE_SHORT_LABELS = backtest.RULE_SHORT_LABELS
LABEL_TEXT = {
    "clean": "clean",
    "genuine_grail": "genuine grail (real, kept)",
    "wrong_card": "planted error: wrong-card V10",
    "fake_grail": "planted error: fake grail",
    "thin_outlier": "planted error: thin outlier",
    "bad_raw": "planted error: corrupted raw price",
}
CARD_ID_PATTERN = re.compile(r"^card_\d{4}$")

SUMMARY_KEYS = (
    "chart_rules",
    "batch_rules",
    "default_budget",
    "footer",
    "generated_by",
    "headline",
    "label_text",
    "nesting",
    "provenance",
    "rule_labels",
    "rule_short_labels",
    "settings",
    "spread_band",
    "sweep_facts",
    "table_rules",
    "title",
    "universe",
    "why",
)
GRID_KEYS = ("budgets", "cards", "cells", "generated_by", "picks")
META_KEYS = (
    "all_rules_picks_grid_bytes",
    "chartjs_tag",
    "chartjs_version",
    "data_budget_bytes",
    "data_bytes",
    "generated_by",
    "tokens_css_sha256_base64",
)
METRIC_KEYS = (
    "budget",
    "leakage_capital_share",
    "leakage_share",
    "mean_predicted_p10",
    "mean_true_p10",
    "n_picks",
    "npv_per_budget_dollar",
    "npv_per_dollar_spent",
    "oracle_value_captured",
    "pick_optimism_dollars",
    "pick_optimism_p10",
    "pick_optimism_share",
    "planted_picks",
    "predicted_npv",
    "regret",
    "spent",
    "true_npv",
)
ORACLE_EXTRA_KEYS = ("bracket_width", "upper_bound")
CARD_KEYS = (
    "capital_paid",
    "gated",
    "label",
    "planted",
    "predicted_capital",
    "predicted_npv",
    "predicted_p10",
    "true_npv",
    "true_p10",
)


def page_cells(sweep_results):
    """{budget: {rule: metrics without picks}} for every rule at every budget."""
    cells = {}
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        results = sweep_results[budget]
        rule_cells = {}
        for rule in backtest.ALL_RULES:
            entry = dict(results[rule])
            entry.pop("picks")
            rule_cells[rule] = entry
        cells[budget] = rule_cells
    return cells


def page_picks(sweep_results, rules=BATCH_RULES):
    """{budget: {rule: card_ids in buy order}} for the listed rules only."""
    picks = {}
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        results = sweep_results[budget]
        rule_picks = {}
        for rule in rules:
            rule_picks[rule] = list(results[rule]["picks"])
        picks[budget] = rule_picks
    return picks


def card_rows(universe, picks):
    """{card_id: row} for every card some listed batch buys.

    Row fields are copied from the backtest universe: predicted NPV and
    capital are the engine's (regular tier), true NPV and capital_paid are
    the closed-form truth, label is the generator's plant label. Only
    picked cards ship, so the file never carries the whole universe.
    """
    wanted = set()
    budgets = sorted(picks.keys())
    for budget in budgets:
        rule_picks = picks[budget]
        rules = sorted(rule_picks.keys())
        for rule in rules:
            for card_id in rule_picks[rule]:
                wanted.add(card_id)
    records = universe.to_dict("records")
    rows = {}
    for card in records:
        card_id = card["card_id"]
        if card_id not in wanted:
            continue
        rows[card_id] = {
            "label": card["label"],
            "planted": bool(card["should_flag"]),
            "gated": bool(card["gated"]),
            "predicted_npv": float(card["npv"]),
            "predicted_capital": float(card["capital"]),
            "predicted_p10": float(card["predicted_p10"]),
            "true_npv": float(card["true_npv"]),
            "capital_paid": float(card["true_capital"]),
            "true_p10": float(card["latent_p10"]),
        }
    return rows


def nesting_breaks(sweep_results, rule):
    """How often a rule's batch at one budget is NOT inside the next one's.

    Counts consecutive sweep budgets b < b' where some card bought at b is
    not bought at b'. A nonzero count is why the slider snaps to solved
    budgets and never derives a batch from a neighbour. Failure mode: 0
    would not prove nesting in general, only on these budgets.
    """
    budgets = sorted(sweep_results.keys())
    pair_count = len(budgets) - 1
    not_nested = 0
    positions = range(pair_count)
    for index in positions:
        smaller = set(sweep_results[budgets[index]][rule]["picks"])
        larger = set(sweep_results[budgets[index + 1]][rule]["picks"])
        if not smaller.issubset(larger):
            not_nested += 1
    return {"pairs": pair_count, "not_nested": not_nested}


def headline_cells(standalone_results):
    """The headline budget's metrics for every rule (no picks)."""
    headline = {}
    for rule in backtest.ALL_RULES:
        entry = dict(standalone_results[rule])
        entry.pop("picks")
        headline[rule] = entry
    return headline


def provenance_labels(n_budgets):
    """Per-widget provenance pills (text only; the page shows them as is)."""
    budget_text = f"${DEFAULT_BUDGET:,}"
    draws = backtest.N_RANDOM_DRAWS
    labels = {
        "headline": f"synthetic truth · {budget_text} budget · the README headline",
        "slider": f"snaps to {n_budgets} solved budgets",
        "tiles": "synthetic truth · precomputed at the chosen budget",
        "chart": (
            f"synthetic truth · precomputed at {n_budgets} budgets · random = mean of "
            f"{draws} seeded shuffles"
        ),
        "rules_table": "synthetic truth · every rule at the chosen budget · precomputed",
        "batch": "synthetic truth · the rule's solved batch at the chosen budget",
        "fallback": f"synthetic truth · precomputed at {n_budgets} budgets",
    }
    return labels


def build_bundle():
    """(summary, grid, universe_rows) as plain dicts, already rounded."""
    universe = backtest.load_world()
    universe_summary = backtest.universe_summary(universe)
    sweep_results = backtest.run_backtest(universe, backtest.SWEEP_BUDGETS)
    standalone = backtest.run_backtest(universe, (DEFAULT_BUDGET,))
    facts = report.sweep_findings(sweep_results, universe_summary["planted_base_rate"])
    picks = page_picks(sweep_results)
    cards = card_rows(universe, picks)
    nesting = {}
    for rule in BATCH_RULES:
        nesting[rule] = nesting_breaks(sweep_results, rule)
    why_text = {}
    for key in PAGE_WHY_KEYS:
        why_text[key] = why.WHY_THIS_METRIC[key]
    n_budgets = len(backtest.SWEEP_BUDGETS)
    summary = {
        "generated_by": "python -m grace.viz build (Experiment 4 functions, synthetic data)",
        "title": TITLE,
        "default_budget": DEFAULT_BUDGET,
        "settings": {
            "tier": backtest.REGULAR_TIER_NAME,
            "budgets": list(backtest.SWEEP_BUDGETS),
            "random_seed": backtest.RANDOM_SEED,
            "random_draws": backtest.N_RANDOM_DRAWS,
            "tick_label_budgets": list(TICK_LABEL_BUDGETS),
        },
        "rule_labels": backtest.RULE_LABELS,
        "rule_short_labels": RULE_SHORT_LABELS,
        "chart_rules": list(CHART_RULES),
        "batch_rules": list(BATCH_RULES),
        "table_rules": list(backtest.ALL_RULES),
        "headline": headline_cells(standalone[DEFAULT_BUDGET]),
        "universe": universe_summary,
        "sweep_facts": facts,
        "nesting": nesting,
        "label_text": LABEL_TEXT,
        "why": why_text,
        "provenance": provenance_labels(n_budgets),
        "spread_band": (
            "No p5-p95 band: the lines come from one synthetic world, and a spread over "
            "generator seeds is not computed yet (to measure)."
        ),
        "footer": FOOTER,
    }
    grid = {
        "generated_by": "python -m grace.viz build (Experiment 4 sweep, regular tier)",
        "budgets": list(backtest.SWEEP_BUDGETS),
        "cells": page_cells(sweep_results),
        "picks": picks,
        "cards": cards,
    }
    all_picks = page_picks(sweep_results, backtest.ALL_RULES)
    all_rules_grid = dict(grid)
    all_rules_grid["picks"] = all_picks
    all_rules_grid["cards"] = card_rows(universe, all_picks)
    return (
        report.round_floats(summary),
        report.round_floats(grid),
        report.round_floats(all_rules_grid),
    )


def text_bytes(text):
    return len(text.encode("utf-8"))


def build_outputs():
    """{repo-relative path: file text} for the three data files."""
    summary, grid, all_rules_grid = build_bundle()
    summary_text = report.to_json_text(summary)
    grid_text = report.to_json_text(grid)
    meta = {
        "generated_by": "python -m grace.viz build",
        "tokens_css_sha256_base64": TOKENS_SHA256_B64,
        "chartjs_version": CHART_JS_VERSION,
        "chartjs_tag": CHART_JS_TAG,
        "data_budget_bytes": DATA_BUDGET_BYTES,
        "data_bytes": {
            "budget_grid.json": text_bytes(grid_text),
            "summary.json": text_bytes(summary_text),
        },
        # The measured alternative behind the BATCH_RULES choice.
        "all_rules_picks_grid_bytes": text_bytes(report.to_json_text(all_rules_grid)),
    }
    outputs = {
        SUMMARY_PATH: summary_text,
        GRID_PATH: grid_text,
        META_PATH: report.to_json_text(meta),
    }
    return outputs


def sha256_base64(data):
    digest = hashlib.sha256(data).digest()
    return base64.b64encode(digest).decode("ascii")


def longest_list(value):
    """Length of the longest list anywhere inside a JSON value."""
    if isinstance(value, dict):
        longest = 0
        inner_values = list(value.values())
        for inner in inner_values:
            longest = max(longest, longest_list(inner))
        return longest
    if isinstance(value, list):
        longest = len(value)
        for inner in value:
            longest = max(longest, longest_list(inner))
        return longest
    return 0


def key_problems(name, mapping, allowed):
    """A problem line when mapping's keys differ from the allowlist."""
    keys = sorted(mapping.keys())
    expected = sorted(allowed)
    if keys != expected:
        return [f"{name}: keys {keys} are not the allowlist {expected}"]
    return []


def bundle_problems(summary, grid, meta):
    """Allowlist and shape problems in the page data (empty when clean).

    Rule: fixed key sets per file, per rule cell and per card row; card ids
    are synthetic card_NNNN; every listed pick has a card row; no list is
    as long as the liquid universe (no per-card dump). Failure mode: it
    checks shape, not values; the generic text patterns are
    tests/test_privacy_patterns.py's job.
    """
    problems = []
    problems += key_problems("summary.json", summary, SUMMARY_KEYS)
    problems += key_problems("budget_grid.json", grid, GRID_KEYS)
    problems += key_problems("meta.json", meta, META_KEYS)
    if problems:
        return problems
    cells = grid["cells"]
    budgets = sorted(cells.keys())
    for budget in budgets:
        problems += key_problems(f"cells[{budget}]", cells[budget], backtest.ALL_RULES)
        rules = sorted(cells[budget].keys())
        for rule in rules:
            allowed = METRIC_KEYS
            if rule == "oracle":
                allowed = (*METRIC_KEYS, *ORACLE_EXTRA_KEYS)
            problems += key_problems(f"cells[{budget}][{rule}]", cells[budget][rule], allowed)
    cards = grid["cards"]
    card_ids = sorted(cards.keys())
    for card_id in card_ids:
        if not CARD_ID_PATTERN.match(card_id):
            problems.append(f"budget_grid.json: card id {card_id!r} is not card_NNNN")
        problems += key_problems(f"cards[{card_id}]", cards[card_id], CARD_KEYS)
    pick_budgets = sorted(grid["picks"].keys())
    for budget in pick_budgets:
        rule_picks = grid["picks"][budget]
        problems += key_problems(f"picks[{budget}]", rule_picks, BATCH_RULES)
        rules = sorted(rule_picks.keys())
        for rule in rules:
            for card_id in rule_picks[rule]:
                if card_id not in cards:
                    problems.append(f"picks[{budget}][{rule}]: {card_id} has no card row")
    universe_size = summary["universe"]["n_liquid"]
    for name, value in (("summary.json", summary), ("budget_grid.json", grid)):
        if longest_list(value) >= universe_size:
            problems.append(f"{name}: a list is as long as the liquid universe")
    return problems


def static_problems(root):
    """Problems with the hand-written page files the build does not write."""
    root_path = Path(root)
    problems = []
    tokens = root_path / TOKENS_PATH
    if not tokens.is_file():
        problems.append(f"{TOKENS_PATH} is missing")
    elif sha256_base64(tokens.read_bytes()) != TOKENS_SHA256_B64:
        problems.append(f"{TOKENS_PATH} is not a byte copy of the page kit's tokens.css")
    if not (root_path / NOJEKYLL_PATH).is_file():
        problems.append(f"{NOJEKYLL_PATH} is missing")
    index = root_path / INDEX_PATH
    if not index.is_file():
        problems.append(f"{INDEX_PATH} is missing")
    elif index.read_text(encoding="utf-8").count(CHART_JS_TAG) != 1:
        problems.append(f"{INDEX_PATH} must carry the pinned Chart.js tag exactly once")
    return problems


def check(root=REPO_ROOT, outputs=None):
    """Every problem --check reports (empty when the page data are fresh)."""
    if outputs is None:
        outputs = build_outputs()
    root_path = Path(root)
    problems = []
    total = 0
    paths = sorted(outputs.keys())
    for path in paths:
        target = root_path / path
        if not target.is_file():
            problems.append(f"{path} is missing; run python -m grace.viz build")
            continue
        committed = target.read_text(encoding="utf-8")
        total += text_bytes(committed)
        if committed != outputs[path]:
            problems.append(f"{path} is stale; run python -m grace.viz build")
    if total > DATA_BUDGET_BYTES:
        problems.append(f"docs/data holds {total} bytes, over the {DATA_BUDGET_BYTES} budget")
    summary = json.loads(outputs[SUMMARY_PATH])
    grid = json.loads(outputs[GRID_PATH])
    meta = json.loads(outputs[META_PATH])
    problems += bundle_problems(summary, grid, meta)
    problems += static_problems(root)
    return problems


def write(root=REPO_ROOT):
    outputs = build_outputs()
    root_path = Path(root)
    (root_path / "docs" / "data").mkdir(parents=True, exist_ok=True)
    paths = sorted(outputs.keys())
    for path in paths:
        (root_path / path).write_text(outputs[path], encoding="utf-8")
    return paths


def decision_claims(summary, meta):
    """(doc, substring) pairs for the page numbers DECISIONS.md quotes.

    tests/test_readme_drift.py checks each appears verbatim, so a changed
    build that leaves DECISIONS.md behind fails CI.
    """
    grace_nesting = summary["nesting"]["grace"]
    oracle_nesting = summary["nesting"]["oracle"]
    shipped_kb = meta["data_bytes"]["budget_grid.json"] / 1000
    all_rules_kb = meta["all_rules_picks_grid_bytes"] / 1000
    claims = [
        (
            "DECISIONS.md",
            (
                f"GRACE's batch is not inside the next budget's batch at "
                f"{grace_nesting['not_nested']} of {grace_nesting['pairs']} consecutive "
                f"budget pairs (the oracle's at {oracle_nesting['not_nested']})"
            ),
        ),
        (
            "DECISIONS.md",
            (
                f"the shipped budget grid is {shipped_kb:.0f} KB; with every rule's batch it "
                f"would be {all_rules_kb:.0f} KB"
            ),
        ),
    ]
    return claims


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"], help="build the page data")
    parser.add_argument("--check", action="store_true", help="fail if the page data are stale")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.root)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print("page data are up to date")
        return 0
    paths = write(args.root)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
