"""README / MODEL.md / DECISIONS.md drift test: every quoted result number
equals what reports/numbers.json says.

Rule (what is checked):
1. Generated blocks. Each block between
   <!-- grace.report:begin NAME --> and <!-- grace.report:end NAME -->
   must equal report.render_blocks(committed numbers.json)[NAME] exactly,
   and every block the report renders must be present in its doc (a
   dropped block is drift too).
2. Prose claims. Every (doc, substring) in report.prose_claims() must
   appear in that doc, with whitespace runs (including line breaks)
   treated as one space, so prose can be re-wrapped freely while any
   changed digit fails.
3. Page claims. Every (doc, substring) in viz.decision_claims() (the page
   numbers DECISIONS.md quotes, rendered from the committed docs/data/
   JSON) must appear the same way.
tests/test_report.py and tests/test_viz.py separately prove numbers.json
and docs/data/ equal a fresh run, so together they chain the docs to the
code.

Failure modes guarded: a number typed or rounded by hand inside a block, a
stale doc after a code change, a deleted block, and a vacuous pass (a
coverage floor plus controls that mutate one digit and delete one block).
Not guarded: a result number typed into prose that nobody listed in
prose_claims(); keep result numbers inside blocks where possible.
"""

import json
import re
from pathlib import Path

from grace import report, viz

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"\d")
# 14 blocks and 43 prose claims exist today; fewer means the parser or the
# claim list stopped seeing them and the test would pass vacuously.
MIN_BLOCKS = 14
MIN_CLAIMS = 43
MIN_PAGE_CLAIMS = 2


def committed_numbers():
    text = (REPO_ROOT / report.NUMBERS_PATH).read_text(encoding="utf-8")
    numbers = json.loads(text)
    return numbers, text


def doc_text(doc_name):
    return (REPO_ROOT / doc_name).read_text(encoding="utf-8")


def block_content(text, name):
    bounds = report.doc_block_bounds(text, name)
    if bounds is None:
        return None
    start, stop = bounds
    content = text[start:stop]
    return content.rstrip("\n")


def block_drift(doc_texts, blocks):
    """Human-readable mismatches between doc blocks and rendered blocks."""
    problems = []
    names = sorted(report.BLOCK_FILES.keys())
    for name in names:
        doc_name = report.BLOCK_FILES[name]
        content = block_content(doc_texts[doc_name], name)
        if content is None:
            problems.append(f"{doc_name}: block {name} is missing")
        elif content != blocks[name]:
            problems.append(f"{doc_name}: block {name} differs from reports/numbers.json")
    return problems


def squash(text):
    pieces = text.split()
    return " ".join(pieces)


def claim_drift(doc_texts, claims):
    problems = []
    squashed = {}
    doc_names = sorted(doc_texts.keys())
    for doc_name in doc_names:
        squashed[doc_name] = squash(doc_texts[doc_name])
    for doc_name, claim in claims:
        if squash(claim) not in squashed[doc_name]:
            problems.append(f"{doc_name}: claim not found: {claim}")
    return problems


def all_doc_texts():
    texts = {}
    doc_names = sorted({*report.BLOCK_FILES.values(), "DECISIONS.md"})
    for doc_name in doc_names:
        texts[doc_name] = doc_text(doc_name)
    return texts


def test_generated_blocks_match_numbers_json():
    numbers, numbers_text = committed_numbers()
    blocks = report.render_blocks(numbers, numbers_text)
    assert len(blocks) >= MIN_BLOCKS
    assert block_drift(all_doc_texts(), blocks) == []


def test_prose_claims_match_numbers_json():
    numbers, _numbers_text = committed_numbers()
    claims = report.prose_claims(numbers)
    assert len(claims) >= MIN_CLAIMS
    for _doc_name, claim in claims:
        assert NUMBER_PATTERN.search(claim), claim  # every claim carries a number
    assert claim_drift(all_doc_texts(), claims) == []


def test_every_block_carries_numbers_or_why_text():
    numbers, numbers_text = committed_numbers()
    blocks = report.render_blocks(numbers, numbers_text)
    names = sorted(blocks.keys())
    for name in names:
        text = blocks[name]
        has_number = NUMBER_PATTERN.search(text) is not None
        has_why = "Why this metric" in text
        assert has_number or has_why, name


def test_control_a_changed_digit_in_a_block_is_caught():
    numbers, numbers_text = committed_numbers()
    blocks = report.render_blocks(numbers, numbers_text)
    texts = all_doc_texts()
    readme = texts["README.md"]
    headline = block_content(readme, "headline")
    assert "0." in headline
    texts["README.md"] = readme.replace(headline, headline.replace("0.", "9.", 1), 1)
    problems = block_drift(texts, blocks)
    assert problems == ["README.md: block headline differs from reports/numbers.json"]


def test_control_a_deleted_block_is_caught():
    numbers, numbers_text = committed_numbers()
    blocks = report.render_blocks(numbers, numbers_text)
    texts = all_doc_texts()
    begin = report.BLOCK_BEGIN.format(name="exp4_tables")
    texts["README.md"] = texts["README.md"].replace(begin, "", 1)
    problems = block_drift(texts, blocks)
    assert problems == ["README.md: block exp4_tables is missing"]


def test_control_a_changed_prose_number_is_caught():
    numbers, _numbers_text = committed_numbers()
    claims = report.prose_claims(numbers)
    texts = all_doc_texts()
    decisions_claims = []
    for doc_name, claim in claims:
        if doc_name == "DECISIONS.md":
            decisions_claims.append(claim)
    target = decisions_claims[0]
    digit_match = NUMBER_PATTERN.search(target)
    digit = digit_match.group(0)
    replacement = "1" if digit != "1" else "2"
    changed = target[: digit_match.start()] + replacement + target[digit_match.end() :]
    # Work on the squashed text: a claim may wrap across lines in the file.
    squashed = squash(texts["DECISIONS.md"])
    assert squash(target) in squashed
    texts["DECISIONS.md"] = squashed.replace(squash(target), squash(changed), 1)
    problems = claim_drift(texts, claims)
    assert problems == [f"DECISIONS.md: claim not found: {target}"]


def committed_page_data():
    summary = json.loads(doc_text(viz.SUMMARY_PATH))
    meta = json.loads(doc_text(viz.META_PATH))
    return summary, meta


def test_page_claims_match_the_page_data():
    summary, meta = committed_page_data()
    claims = viz.decision_claims(summary, meta)
    assert len(claims) >= MIN_PAGE_CLAIMS
    for _doc_name, claim in claims:
        assert NUMBER_PATTERN.search(claim), claim
    assert claim_drift(all_doc_texts(), claims) == []


def test_control_a_changed_page_number_is_caught():
    summary, meta = committed_page_data()
    changed_summary = json.loads(json.dumps(summary))
    changed_summary["nesting"]["grace"]["not_nested"] += 1
    claims = viz.decision_claims(changed_summary, meta)
    problems = claim_drift(all_doc_texts(), claims)
    assert len(problems) == 1
    assert problems[0].startswith("DECISIONS.md: claim not found: GRACE's batch")
