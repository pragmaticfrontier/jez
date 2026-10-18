import pytest
from pydantic import ValidationError

from sysone.schema import BoolQuestion, ChoiceQuestion, DecideRequest, ScoreQuestion


def test_type_aliases_and_inference():
    req = DecideRequest(state="x", questions={
        "a": {"type": "noul", "instructions": "Is it urgent?"},
        "b": {"type": "Boolean", "instructions": "Is it urgent?"},
        "c": {"instructions": "Pick", "options": ["x", "y"]},
        "d": {"instructions": "Rate", "min": 1, "max": 3},
        "e": {"instructions": "Is it urgent?"},
    })
    assert isinstance(req.questions["a"], BoolQuestion)
    assert isinstance(req.questions["b"], BoolQuestion)
    assert isinstance(req.questions["c"], ChoiceQuestion)
    assert req.questions["c"].criteria == {"x": "", "y": ""}
    assert isinstance(req.questions["d"], ScoreQuestion)
    assert isinstance(req.questions["e"], BoolQuestion)


def test_state_may_be_json():
    req = DecideRequest(state={"amount": 4200, "notes": ["a"]}, questions={"q": {"type": "bool", "instructions": "x"}})
    assert req.state["amount"] == 4200


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "x", "criteria": {"only": "one"}},
    {"type": "score", "instructions": "x", "min": 5, "max": 5},
    {"type": "score", "instructions": "x", "min": 1, "max": 5, "criteria": {"9": "out of range"}},
    {"type": "score", "instructions": "x", "min": 1, "max": 40},
    {"type": "wat", "instructions": "x"},
    {"type": "bool", "instructions": ""},
    {"type": "bool", "instructions": "x", "extra": 1},
])
def test_invalid_questions(q):
    with pytest.raises((ValidationError, ValueError)):
        DecideRequest(state="s", questions={"q": q})


def test_options_validation():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": {"type": "bool", "instructions": "x"}}, options={"temperature": 0})
    req = DecideRequest(state="s", questions={"q": {"type": "bool", "instructions": "x"}}, options={"debias": True})
    assert req.options.debias is True and req.options.temperature is None
