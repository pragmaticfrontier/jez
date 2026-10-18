from sysone.prompt import LETTERS, build_probes, format_state
from sysone.schema import BoolQuestion, ChoiceQuestion, ScoreQuestion


def test_choice_probe_labels_and_targets():
    q = ChoiceQuestion(instructions="Route it", criteria={"billing": "money", "technical": "bugs", "account": "login"})
    (p,) = build_probes(q, "state text")
    assert p.labels == ["A", "B", "C"]
    assert p.targets == ["billing", "technical", "account"]
    user = p.messages[-1]["content"]
    assert "A) billing — money" in user and "C) account — login" in user and "state text" in user
    assert p.messages[0]["role"] == "system"


def test_choice_debias_rotations_cover_every_position():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": "", "c": "", "d": ""})
    probes = build_probes(q, "s", debias=True)
    assert len(probes) == 4
    for pos in range(4):
        assert sorted(p.targets[pos] for p in probes) == ["a", "b", "c", "d"]


def test_bool_probe():
    (p,) = build_probes(BoolQuestion(instructions="Is it urgent?"), "s")
    assert p.labels == ["Yes", "No"] and p.targets == ["true", "false"]


def test_score_digits_vs_letters():
    (p,) = build_probes(ScoreQuestion(instructions="sev", min=1, max=5, criteria={"5": "critical"}), "s")
    assert p.labels == ["1", "2", "3", "4", "5"] and p.targets == p.labels
    assert "5 — critical" in p.messages[-1]["content"]
    (p,) = build_probes(ScoreQuestion(instructions="sev", min=0, max=10), "s")
    assert p.labels == list(LETTERS[:11]) and p.targets == [str(i) for i in range(11)]


def test_format_state_json():
    assert format_state("  hi ") == "hi"
    assert format_state({"a": 1}) == '{\n  "a": 1\n}'
