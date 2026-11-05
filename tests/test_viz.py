"""The "Hand GRACE a Budget" page data equal the Experiment 4 functions.

Why: the page computes nothing, so its correctness is a Python fact. These
tests prove the committed docs/data/*.json equal a fresh build
(python -m grace.viz build --check), that the bundle equals the backtest
functions at the same 10-significant-digit rounding (the summary and the
$3,000 cell against a standalone run), that the page files follow the page
kit (pinned Chart.js tag, tokens.css byte copy, .nojekyll, footer, a why box
and a provenance pill per widget), and that the data stay inside their size
budget and key allowlist. Controls show a stale file, a wrong tokens.css,
a missing tag and a planted non-synthetic field are each caught.
Not covered here: whether the browser renders the JSON correctly; the
screenshot click-through (each slider stop against the JSON) does that.
"""

import base64
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from grace import report, viz, why
from grace.experiments import backtest

REPO_ROOT = Path(__file__).resolve().parent.parent
# The portfolio page kit sits next to the repo locally; CI has no copy.
KIT_TOKENS = REPO_ROOT.parent / "_build" / "pagekit" / "tokens.css"


@pytest.fixture(scope="module")
def fresh_outputs():
    return viz.build_outputs()


@pytest.fixture(scope="module")
def committed():
    bundle = {}
    for path in viz.DATA_PATHS:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        bundle[path] = json.loads(text)
    return bundle


@pytest.fixture(scope="module")
def universe():
    return backtest.load_world()


@pytest.fixture(scope="module")
def sweep_results(universe):
    return backtest.run_backtest(universe, backtest.SWEEP_BUDGETS)


def page_text(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def copy_page_tree(tmp_path, outputs):
    """A throwaway docs/ tree holding the given outputs and the static files."""
    static_paths = (viz.TOKENS_PATH, viz.NOJEKYLL_PATH, viz.INDEX_PATH, viz.APP_PATH)
    for path in static_paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / path).read_bytes())
    paths = sorted(outputs.keys())
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(outputs[path], encoding="utf-8")


def test_committed_page_data_match_a_fresh_build(fresh_outputs):
    assert viz.check(REPO_ROOT, fresh_outputs) == []


def test_build_is_deterministic(fresh_outputs):
    assert viz.build_outputs() == fresh_outputs


def test_control_stale_and_missing_data_are_caught(tmp_path, fresh_outputs):
    copy_page_tree(tmp_path, fresh_outputs)
    assert viz.check(tmp_path, fresh_outputs) == []

    grid_path = tmp_path / viz.GRID_PATH
    text = grid_path.read_text(encoding="utf-8")
    mutated = text.replace('"generated_by": "python', '"generated_by": "PYTHON', 1)
    assert mutated != text
    grid_path.write_text(mutated, encoding="utf-8")
    assert viz.check(tmp_path, fresh_outputs) == [
        f"{viz.GRID_PATH} is stale; run python -m grace.viz build"
    ]

    (tmp_path / viz.SUMMARY_PATH).unlink()
    problems = viz.check(tmp_path, fresh_outputs)
    assert f"{viz.SUMMARY_PATH} is missing; run python -m grace.viz build" in problems


def test_control_broken_static_files_are_caught(tmp_path, fresh_outputs):
    copy_page_tree(tmp_path, fresh_outputs)
    tokens = tmp_path / viz.TOKENS_PATH
    tokens.write_bytes(tokens.read_bytes() + b"\n")
    (tmp_path / viz.NOJEKYLL_PATH).unlink()
    index = tmp_path / viz.INDEX_PATH
    index.write_text(
        index.read_text(encoding="utf-8").replace("integrity=", "data-x=", 1), encoding="utf-8"
    )
    problems = viz.check(tmp_path, fresh_outputs)
    assert problems == [
        f"{viz.TOKENS_PATH} is not a byte copy of the page kit's tokens.css",
        f"{viz.NOJEKYLL_PATH} is missing",
        f"{viz.INDEX_PATH} must carry the pinned Chart.js tag exactly once",
    ]


def test_cli_check_exits_zero_on_the_committed_tree(capsys):
    assert viz.main(["build", "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


# ---- the bundle equals the functions ------------------------------------


def test_headline_cell_equals_a_standalone_run(universe, committed):
    """Plan: the bundle matches the functions (summary and the $3k cell)."""
    standalone = backtest.run_backtest(universe, (backtest.HEADLINE_BUDGET,))
    results = standalone[backtest.HEADLINE_BUDGET]
    summary = committed[viz.SUMMARY_PATH]
    grid = committed[viz.GRID_PATH]
    budget_key = str(backtest.HEADLINE_BUDGET)
    for rule in backtest.ALL_RULES:
        expected = dict(results[rule])
        picks = expected.pop("picks")
        rounded = report.round_floats(expected)
        assert summary["headline"][rule] == rounded, rule
        assert grid["cells"][budget_key][rule] == rounded, rule
        if rule in viz.BATCH_RULES:
            assert grid["picks"][budget_key][rule] == picks, rule


def test_headline_equals_the_readme_numbers(committed):
    """The page's headline tile and the README cite the same numbers."""
    numbers = json.loads(page_text(report.NUMBERS_PATH))
    kpis = numbers["kpis"]
    headline = committed[viz.SUMMARY_PATH]["headline"]
    budget = backtest.HEADLINE_BUDGET
    for rule in backtest.ALL_RULES:
        entry = headline[rule]
        assert entry["oracle_value_captured"] == kpis[f"kpi_oracle_value_captured_{rule}_{budget}"]
        assert entry["regret"] == kpis[f"kpi_regret_{rule}_{budget}"]
        assert entry["planted_picks"] == kpis[f"kpi_planted_picks_{rule}_{budget}"]


def test_every_sweep_cell_equals_the_backtest(sweep_results, committed):
    grid = committed[viz.GRID_PATH]
    assert grid["budgets"] == list(backtest.SWEEP_BUDGETS)
    assert grid["cells"] == report.round_floats(viz.page_cells(sweep_results))


def test_page_batches_equal_the_backtest_picks(sweep_results, committed):
    """Every listed batch is a solved run, never derived from a neighbour."""
    grid = committed[viz.GRID_PATH]
    budgets = sorted(sweep_results.keys())
    for budget in budgets:
        for rule in viz.BATCH_RULES:
            assert grid["picks"][str(budget)][rule] == sweep_results[budget][rule]["picks"]


def test_page_matches_the_report_sweep(committed):
    """The page and reports/backtest_sweep.json are one backtest, not two."""
    sweep = json.loads(page_text(report.SWEEP_PATH))
    grid = committed[viz.GRID_PATH]
    budgets = sorted(sweep["cells"].keys())
    for budget in budgets:
        for rule in backtest.ALL_RULES:
            entry = dict(sweep["cells"][budget][rule])
            picks = entry.pop("picks")
            assert grid["cells"][budget][rule] == entry
            if rule in viz.BATCH_RULES:
                assert grid["picks"][budget][rule] == picks


def test_card_rows_equal_the_universe(universe, committed):
    cards = committed[viz.GRID_PATH]["cards"]
    records = universe.to_dict("records")
    checked = 0
    for card in records:
        card_id = card["card_id"]
        if card_id not in cards:
            continue
        row = cards[card_id]
        assert row["true_npv"] == report.round_floats(float(card["true_npv"]))
        assert row["capital_paid"] == report.round_floats(float(card["true_capital"]))
        assert row["predicted_npv"] == report.round_floats(float(card["npv"]))
        assert row["true_p10"] == report.round_floats(float(card["latent_p10"]))
        assert row["planted"] == bool(card["should_flag"])
        assert row["label"] == card["label"]
        checked += 1
    assert checked == len(cards)


def test_batch_metrics_agree_with_their_card_rows(committed):
    """The batch table's rows add up to its summary line (a cross-check the
    page itself never performs)."""
    grid = committed[viz.GRID_PATH]
    budgets = sorted(grid["picks"].keys())
    for budget in budgets:
        for rule in viz.BATCH_RULES:
            picks = grid["picks"][budget][rule]
            true_total = 0.0
            planted = 0
            for card_id in picks:
                row = grid["cards"][card_id]
                true_total += row["true_npv"]
                if row["planted"]:
                    planted += 1
            cell = grid["cells"][budget][rule]
            assert cell["n_picks"] == len(picks)
            assert cell["planted_picks"] == planted
            assert true_total == pytest.approx(cell["true_npv"], rel=1e-8)


def test_summary_facts_equal_the_report(committed):
    numbers = json.loads(page_text(report.NUMBERS_PATH))
    summary = committed[viz.SUMMARY_PATH]
    assert summary["sweep_facts"] == numbers["backtest"]["sweep_facts"]
    assert summary["universe"] == numbers["backtest"]["universe"]
    assert summary["settings"]["budgets"] == numbers["backtest"]["settings"]["sweep_budgets"]


def test_why_text_is_the_shared_dict(committed):
    summary_why = committed[viz.SUMMARY_PATH]["why"]
    for key in viz.PAGE_WHY_KEYS:
        assert summary_why[key] == why.WHY_THIS_METRIC[key]


def test_nesting_breaks_counts_a_hand_case():
    sweep = {
        1: {"grace": {"picks": ["a"]}},
        2: {"grace": {"picks": ["a", "b"]}},
        3: {"grace": {"picks": ["c"]}},
    }
    assert viz.nesting_breaks(sweep, "grace") == {"pairs": 2, "not_nested": 1}


def test_grace_batches_are_not_nested(committed):
    """Why the slider snaps to solved budgets: at least one GRACE batch is
    not inside the next budget's (a 0 would make interpolation look safe)."""
    nesting = committed[viz.SUMMARY_PATH]["nesting"]["grace"]
    assert nesting["pairs"] == len(backtest.SWEEP_BUDGETS) - 1
    assert nesting["not_nested"] > 0


def test_default_budget_is_a_slider_stop(committed):
    summary = committed[viz.SUMMARY_PATH]
    assert summary["default_budget"] in summary["settings"]["budgets"]
    labelled = summary["settings"]["tick_label_budgets"]
    for budget in labelled:
        assert budget in summary["settings"]["budgets"]


# ---- page kit conventions ------------------------------------------------


def test_tokens_css_is_the_pinned_kit_copy():
    data = (REPO_ROOT / viz.TOKENS_PATH).read_bytes()
    digest = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
    assert digest == viz.TOKENS_SHA256_B64


def test_pinned_tokens_hash_equals_the_kit():
    if not KIT_TOKENS.is_file():
        pytest.skip("page kit not present (CI); the pinned hash test above still runs")
    assert (REPO_ROOT / viz.TOKENS_PATH).read_bytes() == KIT_TOKENS.read_bytes()


def test_meta_records_the_pinned_hash_and_tag(committed):
    meta = committed[viz.META_PATH]
    assert meta["tokens_css_sha256_base64"] == viz.TOKENS_SHA256_B64
    assert meta["chartjs_tag"] == viz.CHART_JS_TAG
    assert meta["chartjs_version"] == "4.4.1"


def test_chart_js_tag_is_exactly_the_kit_string():
    expected = (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
        'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zz'
        'e6Z7kHXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
    )
    assert expected == viz.CHART_JS_TAG
    assert page_text(viz.INDEX_PATH).count(expected) == 1


def test_page_loads_no_other_external_code():
    html = page_text(viz.INDEX_PATH)
    sources = re.findall(r'<script[^>]*src="([^"]+)"', html)
    chart_src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
    assert sources == [chart_src, "app.js"]
    links = re.findall(r'<link[^>]*href="([^"]+)"', html)
    assert links == ["tokens.css"]
    app = page_text(viz.APP_PATH)
    assert "http://" not in app
    assert "https://" not in app


def test_nojekyll_exists():
    assert (REPO_ROOT / viz.NOJEKYLL_PATH).is_file()


def test_footer_text_is_the_kit_footer(committed):
    assert committed[viz.SUMMARY_PATH]["footer"] == (
        "Every number was computed by this repo's tested Python and copied; "
        "the page computes nothing."
    )
    assert 'id="footer-text"' in page_text(viz.INDEX_PATH)


def test_every_widget_has_a_why_box_and_a_provenance_pill(committed):
    html = page_text(viz.INDEX_PATH)
    summary = committed[viz.SUMMARY_PATH]
    sections = html.split('<section class="card"')[1:]
    assert len(sections) >= 5
    for section in sections:
        assert 'class="provenance" data-prov="' in section
        assert 'class="why" data-why="' in section
    why_keys = re.findall(r'data-why="([a-z_]+)"', html)
    assert sorted(set(why_keys)) == sorted(viz.PAGE_WHY_KEYS)
    prov_keys = re.findall(r'data-prov="([a-z_]+)"', html)
    assert sorted(set(prov_keys)) == sorted(summary["provenance"].keys())


def test_slider_stops_are_the_sweep_budgets(committed):
    summary = committed[viz.SUMMARY_PATH]
    n_budgets = len(backtest.SWEEP_BUDGETS)
    assert summary["provenance"]["slider"] == f"snaps to {n_budgets} solved budgets"
    html = page_text(viz.INDEX_PATH)
    assert 'type="range"' in html
    assert 'class="ticks stops"' in html


def test_fallback_message_is_the_kit_text():
    assert "chart unavailable, table shown" in page_text(viz.INDEX_PATH)
    assert "window.Chart" in page_text(viz.APP_PATH)


def test_page_computes_no_sums():
    """A coarse guard on the parity rule: the viewer has no reducers or
    running totals. Formatting (x100 for a percent) is allowed."""
    app = page_text(viz.APP_PATH)
    assert ".reduce(" not in app
    assert "+= c" not in app
    assert "Math.max(" not in app
    assert "Math.min(" not in app


# ---- size and privacy ----------------------------------------------------


def test_page_data_stay_within_the_budget():
    total = 0
    for path in viz.DATA_PATHS:
        total += (REPO_ROOT / path).stat().st_size
    assert total <= viz.DATA_BUDGET_BYTES


def test_measured_sizes_in_meta_are_the_file_sizes(committed):
    meta = committed[viz.META_PATH]
    assert meta["data_bytes"]["budget_grid.json"] == (REPO_ROOT / viz.GRID_PATH).stat().st_size
    assert meta["data_bytes"]["summary.json"] == (REPO_ROOT / viz.SUMMARY_PATH).stat().st_size
    assert meta["all_rules_picks_grid_bytes"] > meta["data_bytes"]["budget_grid.json"]


def test_committed_bundle_passes_the_allowlist(committed):
    problems = viz.bundle_problems(
        committed[viz.SUMMARY_PATH], committed[viz.GRID_PATH], committed[viz.META_PATH]
    )
    assert problems == []


def test_control_planted_fields_fail_the_allowlist(committed):
    summary = copy.deepcopy(committed[viz.SUMMARY_PATH])
    grid = copy.deepcopy(committed[viz.GRID_PATH])
    meta = committed[viz.META_PATH]

    summary_extra = copy.deepcopy(summary)
    summary_extra["store"] = "x"
    assert viz.bundle_problems(summary_extra, grid, meta) != []

    first_id = sorted(grid["cards"].keys())[0]
    renamed = copy.deepcopy(grid)
    renamed["cards"]["real-card"] = renamed["cards"].pop(first_id)
    problems = viz.bundle_problems(summary, renamed, meta)
    assert "budget_grid.json: card id 'real-card' is not card_NNNN" in problems

    extra_field = copy.deepcopy(grid)
    extra_field["cards"][first_id]["cost_basis"] = 1.0
    assert viz.bundle_problems(summary, extra_field, meta) != []

    dumped = copy.deepcopy(grid)
    dumped["budgets"] = list(range(summary["universe"]["n_liquid"]))
    problems = viz.bundle_problems(summary, dumped, meta)
    assert "budget_grid.json: a list is as long as the liquid universe" in problems


def test_random_mean_label_goes_by_rule():
    """picksText marks the random row '(mean)' by rule, never by whether the
    shuffle mean happens to be a whole number."""
    app = page_text(viz.APP_PATH)
    assert 'if (rule === "random") text += " (mean)";' in app
    assert "Number.isInteger(entry.n_picks)" not in app


def test_page_says_a_one_beats_grace_in_words(committed):
    """The every-rule table shows it; the headline card must say it too,
    built from the sweep facts and the headline cells."""
    html = page_text(viz.INDEX_PATH)
    assert 'id="a-one-note"' in html
    app = page_text(viz.APP_PATH)
    assert 'byId("a-one-note")' in app
    assert "f.npv_per_dollar_beats_grace" in app
    assert "h.gated_npv_per_dollar.oracle_value_captured" in app
    facts = committed[viz.SUMMARY_PATH]["sweep_facts"]
    assert facts["npv_per_dollar_beats_grace"] > 0


def test_tick_labels_are_decluttered_at_narrow_widths():
    """At phone width the $250 and $1k labels would touch; the page hides an
    inner label that would overlap its neighbour and keeps every mark."""
    app = page_text(viz.APP_PATH)
    assert "function declutterTicks()" in app
    assert 'window.addEventListener("resize", declutterTicks)' in app
    assert '{ "class": "tick-label" }' in app

