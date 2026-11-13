"""JavaScript parity: docs/play/calc.js equals grace/play.py on the fixtures.

Why: the calculator page computes in the browser, so the formulas exist
twice. This test runs calc.js under node (tests/play_parity_runner.js) on
every fixture input set in docs/play/data/parity.json and compares each
value with the Python output stored beside it. tests/test_play.py proves
that stored output equals a fresh Python run.

Rule: numbers must agree to 1e-9 relative. The Python side is stored at 10
significant digits, so up to 5e-10 of that is rounding. A 1e-9 absolute
floor lets an exact zero compare with a rounding-sized remainder. Text,
booleans, nulls, list lengths and dict keys must match exactly.

Skipped, with the reason printed, when node is not installed. The CI
workflow installs node and sets GRACE_REQUIRE_NODE=1, which turns that
skip into a failure, so CI cannot pass without running it.
Failure mode: agreement on the fixtures is not agreement on every input; a
formula change must add a fixture that exercises it. The controls show a
changed number, a changed flag and a missing key are each reported.
"""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grace import play

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = "tests/play_parity_runner.js"
RELATIVE_TOLERANCE = play.PARITY_RELATIVE_TOLERANCE
NODE_TIMEOUT_SECONDS = 60
REQUIRE_NODE_VARIABLE = "GRACE_REQUIRE_NODE"


def node_binary():
    """The node to run: NODE_BINARY if set, else whatever is on PATH."""
    from_env = os.environ.get("NODE_BINARY")
    if from_env:
        return from_env
    return shutil.which("node")


def differences(expected, actual, where):
    """Every place the Python value and the JavaScript value disagree."""
    return play.json_differences(expected, actual, where, "python", "javascript")


@pytest.fixture(scope="module")
def parity():
    text = (REPO_ROOT / play.PARITY_PATH).read_text(encoding="utf-8")
    return json.loads(text)


@pytest.fixture(scope="module")
def javascript_outputs():
    node = node_binary()
    if node is None:
        reason = (
            "node is not installed, so docs/play/calc.js was NOT checked against "
            "grace/play.py in this run (CI installs node and runs it)"
        )
        print(reason)
        if os.environ.get(REQUIRE_NODE_VARIABLE) == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    command = [node, str(REPO_ROOT / RUNNER_PATH)]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=NODE_TIMEOUT_SECONDS
    )
    return json.loads(completed.stdout)


def test_calc_js_matches_python_on_every_fixture(parity, javascript_outputs):
    fixtures = parity["fixtures"]
    actual_fixtures = javascript_outputs["fixtures"]
    assert len(actual_fixtures) == len(fixtures)
    problems = []
    compared_numbers = 0
    fixture_count = len(fixtures)
    positions = range(fixture_count)
    for position in positions:
        expected = fixtures[position]
        actual = actual_fixtures[position]
        assert actual["name"] == expected["name"]
        name = expected["name"]
        problems += differences(expected["expected"], actual["actual"], name)
        problems += differences(
            expected["expected_rank_at_rate"], actual["actual_rank_at_rate"], name + ".rank_at_rate"
        )
        compared_numbers += count_numbers(expected["expected"])
    assert problems == []
    # a vacuous pass (empty outputs) would compare nothing
    assert compared_numbers > 1000


def test_rate_helpers_match_python(parity, javascript_outputs):
    helpers = parity["helpers"]
    actual = javascript_outputs["helpers"]
    for name in ("weekly_pct_from_annual", "annual_pct_from_weekly"):
        expected_values = []
        for sample in helpers[name]:
            expected_values.append(sample["expected"])
        assert len(expected_values) >= 3
        assert differences(expected_values, actual[name], name) == []


def count_numbers(value):
    """How many numbers a JSON value holds (for the vacuous-pass guard)."""
    if play.is_plain_number(value):
        return 1
    total = 0
    if isinstance(value, dict):
        inner_values = list(value.values())
        for inner in inner_values:
            total += count_numbers(inner)
    if isinstance(value, list):
        for inner in value:
            total += count_numbers(inner)
    return total


def test_fixtures_cover_what_the_spec_lists(parity):
    """At least 12 input sets: mixed companies, both fee timings, rate 0, a
    high rate, a batch below a minimum, a price above a cap, each range mode."""
    fixtures = parity["fixtures"]
    assert len(fixtures) >= play.MIN_FIXTURES
    names = []
    range_modes = set()
    rates = set()
    fee_timings = set()
    companies = set()
    flags = set()
    skipped = set()
    for entry in fixtures:
        names.append(entry["name"])
        range_modes.add(entry["inputs"]["range_mode"])
        rates.add(entry["inputs"]["weekly_rate_pct"])
        skipped.update(entry["expected"]["skipped_companies"])
        for tier in entry["expected"]["tiers"]:
            fee_timings.add((tier["company"], tier["fee_timing"]))
            companies.add(tier["company"])
            flags.update(tier["flags"])
    assert len(set(names)) == len(names)
    assert range_modes == set(play.RANGE_MODES)
    assert 0.0 in rates
    assert max(rates) > 10.0
    for company in ("psa", "tag"):
        for fee_timing in play.FEE_TIMINGS:
            assert (company, fee_timing) in fee_timings
    assert {"psa", "tag", "bgs"} <= companies
    assert flags == {play.FLAG_BELOW_MINIMUM, play.FLAG_OVER_CAP}
    assert skipped == {"tag"}


def test_stored_python_outputs_are_at_ten_significant_digits(parity):
    assert parity["significant_digits"] == 10
    assert parity["relative_tolerance"] == RELATIVE_TOLERANCE
    first = parity["fixtures"][0]
    fresh = play.evaluate(first["inputs"], play.service_companies())
    assert first["name"] == "defaults"
    assert differences(first["expected"], json.loads(json.dumps(fresh)), "defaults") == []


def test_control_differences_are_reported(parity):
    """The comparison can fail: a changed number, flag, key and length."""
    expected = parity["fixtures"][0]["expected"]
    assert differences(expected, copy.deepcopy(expected), "x") == []

    changed_number = copy.deepcopy(expected)
    changed_number["tiers"][0]["npv"] *= 1 + 1e-8
    problems = differences(expected, changed_number, "x")
    assert len(problems) == 1
    assert problems[0].startswith("x.tiers[0].npv: python ")

    within_tolerance = copy.deepcopy(expected)
    within_tolerance["tiers"][0]["npv"] *= 1 + 1e-11
    assert differences(expected, within_tolerance, "x") == []

    changed_flag = copy.deepcopy(expected)
    changed_flag["tiers"][0]["eligible"] = not expected["tiers"][0]["eligible"]
    assert len(differences(expected, changed_flag, "x")) == 1

    missing_key = copy.deepcopy(expected)
    missing_key["tiers"][0].pop("day_sale")
    assert differences(expected, missing_key, "x")[0].startswith("x.tiers[0]: keys differ")

    shorter = copy.deepcopy(expected)
    shorter["ranking"].pop()
    assert differences(expected, shorter, "x")[0].startswith("x.ranking: length")

    # a boolean is never "close to" a number, and null is not zero
    assert differences(1, True, "x") != []
    assert differences(0.0, None, "x") != []
