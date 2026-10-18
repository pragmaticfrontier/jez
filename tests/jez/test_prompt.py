from __future__ import annotations

from jez.engine import FakeBackend
from jez.prompt import SYSTEM_PROMPT, plan, question_block, render_content
from jez.schema import validate_request


def _req(**questions):
    return validate_request({"state": {"ticket": 12, "text": "Charged twice"}, "model": "jez-small-latest", "questions": questions})


def test_two_orders_per_question_and_target_alignment():
    req = _req(dept={"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "sales": None}})
    probes = plan(req, FakeBackend().table)
    assert [p.order for p in probes] == ["given", "reversed"]
    given, rev = probes
    assert given.labels == ("A", "B", "C") and given.targets == ("billing", "tech", "sales")
    assert rev.labels == ("A", "B", "C") and rev.targets == ("sales", "tech", "billing")
    text = given.messages[1]["content"]
    assert "A) billing — money" in text and "B) tech — bugs" in text and "C) sales\n" in text
    assert "(A, B, or C)" in text
    assert given.messages[0]["content"] == SYSTEM_PROMPT


def test_state_rendered_once_as_indented_json():
    req = _req(q={"type": "noul", "instructions": "Is it billing?"})
    (p, _) = plan(req, None)
    assert p.messages[1]["content"].startswith('### STATE\n{\n  "ticket": 12,\n  "text": "Charged twice"\n}\n\n### QUESTION')


def test_noul_swaps_yes_and_no():
    req = _req(q={"type": "noul", "instructions": "Urgent?", "criteria": {"true": "deadline today", "false": "no rush"}})
    given, rev = plan(req, None)
    assert given.labels == ("Yes", "No") and given.targets == ("true", "false")
    assert rev.labels == ("No", "Yes") and rev.targets == ("false", "true")
    assert "Yes — deadline today\nNo — no rush" in given.messages[1]["content"]
    assert "No — no rush\nYes — deadline today" in rev.messages[1]["content"]
    assert "Reply with only No or Yes." in rev.messages[1]["content"]


def test_score_levels_are_digits_and_reverse_lists_highest_first():
    req = _req(q={"type": "score", "instructions": "How angry?", "criteria": ["calm", {"level": "mid"}, "furious"]})
    given, rev = plan(req, None)
    assert given.labels == ("0", "1", "2") and given.targets == ("0", "1", "2")
    assert rev.labels == ("2", "1", "0") and rev.targets == ("2", "1", "0")
    body = given.messages[1]["content"]
    assert "### LEVELS (0 = lowest, 2 = highest)\n0 — calm\n1 — {\"level\": \"mid\"}\n2 — furious" in body
    assert "digit (2 to 0)" in rev.messages[1]["content"]


def test_structured_instructions_and_long_descriptions_become_json_blocks():
    long_desc = {"when": "x" * 200}
    req = _req(q={"type": "choice", "instructions": {"task": "route", "see": "`text`"}, "criteria": {"a": long_desc, "b": ["short"]}})
    (p, _) = plan(req, None)
    body = p.messages[1]["content"]
    assert '### QUESTION\n{\n  "task": "route",\n  "see": "`text`"\n}' in body
    assert "A) a —\n    {\n" in body  # indented block for a long object
    assert 'B) b — ["short"]' in body


def test_255_options_use_two_letter_labels_and_a_range():
    req = _req(q={"type": "choice", "instructions": "Pick", "criteria": {f"o{i}": None for i in range(255)}})
    table = FakeBackend().table
    text, labels, targets = question_block(req.questions[0], table)
    assert labels[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and labels[26] == "AA" and len(labels) == 255
    assert f"(A to {labels[-1]})" in text and targets == [f"o{i}" for i in range(255)]


def test_render_content():
    assert render_content("  hi ") == "hi"
    assert render_content([1, {"a": None}]) == '[\n  1,\n  {\n    "a": null\n  }\n]'
