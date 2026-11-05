"""The committed reports equal a fresh run (python -m grace.report --check).

Why: README numbers are copied from reports/numbers.json; this test makes
that file itself trustworthy by recomputing it. The run takes about two
seconds, so it stays in CI. The controls show a stale or missing file is
caught, so the check cannot pass vacuously.
"""

import json
import math
from pathlib import Path

import pytest

from grace import report, why
from grace.experiments import backtest, gem_cv

REPO_ROOT = Path(__file__).resolve().parent.parent
# Plan budget for the grace page data (Shared standards (d)).
DATA_BUDGET_BYTES = 800_000


@pytest.fixture(scope="module")
def fresh_outputs():
    return report.build_outputs(REPO_ROOT)


def test_committed_reports_match_a_fresh_run(fresh_outputs):
    problems = report.check(REPO_ROOT, fresh_outputs)
    assert problems == []


def test_report_is_deterministic(fresh_outputs):
    again = report.build_outputs(REPO_ROOT)
    assert again == fresh_outputs


def test_control_stale_and_missing_files_are_caught(tmp_path, fresh_outputs):
    paths = sorted(fresh_outputs.keys())
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fresh_outputs[path], encoding="utf-8")
    figure = tmp_path / report.FIGURE_PATH
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes((REPO_ROOT / report.FIGURE_PATH).read_bytes())
    assert report.check(tmp_path, fresh_outputs) == []

    numbers_path = tmp_path / report.NUMBERS_PATH
    text = numbers_path.read_text(encoding="utf-8")
    mutated = text.replace('"generated_by": "python', '"generated_by": "PYTHON', 1)
    assert mutated != text
    numbers_path.write_text(mutated, encoding="utf-8")
    stale = report.check(tmp_path, fresh_outputs)
    assert stale == [f"{report.NUMBERS_PATH} is stale; run python -m grace.report"]

    (tmp_path / report.SWEEP_PATH).unlink()
    missing = report.check(tmp_path, fresh_outputs)
    assert f"{report.SWEEP_PATH} is missing; run python -m grace.report" in missing


def test_floats_rounded_to_ten_significant_digits_and_no_nan():
    assert report.round_floats(1.0 / 3.0) == 0.3333333333
    assert report.round_floats(float("nan")) is None
    assert report.round_floats({1: [2.5, math.inf]}) == {"1": [2.5, None]}
    assert report.round_floats(True) is True


def test_json_line_requires_a_unique_key():
    text = '{\n "a": 1,\n "b": {\n  "a": 2\n },\n "c": 3\n}\n'
    assert report.json_line(text, "c") == 6
    with pytest.raises(KeyError):
        report.json_line(text, "a")


def test_kpi_keys_are_unique_lines_in_numbers_json():
    text = (REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8")
    numbers = json.loads(text)
    kpi_keys = sorted(numbers["kpis"].keys())
    assert len(kpi_keys) > 10
    for key in kpi_keys:
        report.json_line(text, key)  # raises unless exactly one line


def test_numbers_json_shares_the_why_dict():
    """README, page and findings show one copy of the why-box text."""
    text = (REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8")
    numbers = json.loads(text)
    assert numbers["why"] == why.WHY_THIS_METRIC
    keys = sorted(why.WHY_THIS_METRIC.keys())
    for key in keys:
        entry = why.WHY_THIS_METRIC[key]
        assert sorted(entry.keys()) == ["kind", "lie", "question"], key


def test_report_files_stay_within_the_page_data_budget():
    total = 0
    for path in (report.NUMBERS_PATH, report.SWEEP_PATH):
        total += (REPO_ROOT / path).stat().st_size
    assert total <= DATA_BUDGET_BYTES


def test_report_json_keys_are_an_allowlist():
    """Top-level keys are fixed, so no stray (possibly private) field ships."""
    numbers = json.loads((REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8"))
    assert sorted(numbers.keys()) == [
        "assumptions",
        "backtest",
        "demo",
        "gates",
        "gem_cv",
        "generated_by",
        "kpis",
        "m_estimation",
        "why",
    ]
    sweep = json.loads((REPO_ROOT / report.SWEEP_PATH).read_text(encoding="utf-8"))
    assert sorted(sweep.keys()) == ["budgets", "cells", "generated_by", "picks_note", "rule_labels"]


def test_sweep_picks_are_synthetic_card_ids_only():
    sweep = json.loads((REPO_ROOT / report.SWEEP_PATH).read_text(encoding="utf-8"))
    cells = sweep["cells"]
    budgets = sorted(cells.keys())
    for budget in budgets:
        rules = sorted(cells[budget].keys())
        for rule in rules:
            picks = cells[budget][rule]["picks"]
            for card_id in picks:
                assert card_id.startswith("card_") and len(card_id) == 9, card_id


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


def test_gem_cv_section_is_aggregates_only():
    """Experiment 3 ships fold aggregates, never a per-card row."""
    numbers = json.loads((REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8"))
    section = numbers["gem_cv"]
    assert sorted(section.keys()) == [
        "empty_ladders",
        "fold_sizes",
        "n_cards",
        "settings",
        "targets",
        "training_fit_ranges",
    ]
    assert longest_list(section) <= gem_cv.K_FOLDS
    assert sorted(section["targets"].keys()) == sorted(gem_cv.TARGETS.keys())


def test_control_a_stale_or_missing_figure_is_caught(tmp_path, fresh_outputs):
    """The sweep chart is checked by the series digest in its PNG text chunk:
    a chart drawn from other numbers, or no chart, must be reported."""
    paths = sorted(fresh_outputs.keys())
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fresh_outputs[path], encoding="utf-8")
    stale_message = f"{report.FIGURE_PATH} is stale; run python -m grace.report"
    missing_message = f"{report.FIGURE_PATH} is missing; run python -m grace.report"
    assert report.check(tmp_path, fresh_outputs) == [missing_message]

    cells = json.loads(fresh_outputs[report.SWEEP_PATH])["cells"]
    budgets = sorted(cells.keys())
    first = budgets[0]
    changed_cell = dict(cells[first])
    grace_entry = dict(changed_cell["grace"])
    grace_entry["npv_per_budget_dollar"] += 0.5
    changed_cell["grace"] = grace_entry
    changed = dict(cells)
    changed[first] = changed_cell
    figure = tmp_path / report.FIGURE_PATH
    figure.parent.mkdir(parents=True, exist_ok=True)
    backtest.make_figure(changed, str(figure), backtest.figure_metadata(changed))
    assert report.check(tmp_path, fresh_outputs) == [stale_message]

    report.write_figure(tmp_path, fresh_outputs)
    assert report.check(tmp_path, fresh_outputs) == []


def test_picks_label_goes_by_rule_not_by_whole_numbers():
    """A shuffle mean that lands on a whole number is still a mean."""
    whole_mean = {"planted_picks": 3.0, "n_picks": 27.0}
    assert report.picks_text(whole_mean, "random") == "3.0/27.0 (mean)"
    assert report.picks_text(whole_mean, "grace") == "3/27"


def test_kpi_regret_row_shows_the_rule_that_beats_grace():
    """The regret row must not imply GRACE has the lowest regret: gated NPV
    per dollar (a = 1) is in its value cell, and the expectation names only
    the heuristic and random."""
    text = (REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8")
    numbers = json.loads(text)
    table = report.render_kpi_table(numbers, text)
    regret_rows = []
    table_lines = table.splitlines()
    for line in table_lines:
        if line.startswith("| Regret and NPV per budget dollar"):
            regret_rows.append(line)
    assert len(regret_rows) == 1
    row = regret_rows[0]
    headline = numbers["backtest"]["by_budget"][str(backtest.HEADLINE_BUDGET)]
    a_one_regret = report.dollars(headline["gated_npv_per_dollar"]["regret"])
    assert f"gated NPV per dollar {a_one_regret}" in row
    assert "every baseline" not in row
    assert "Lower regret than the heuristic and random" in row

