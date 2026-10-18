import numpy as np
import pytest

from sysone.decider import aggregate
from sysone.prompt import build_probes
from sysone.schema import BoolQuestion, ChoiceQuestion, ScoreQuestion


def test_choice_aggregate_and_temperature():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": "", "c": ""})
    probes = build_probes(q, "s")
    ans = aggregate(q, probes, [np.array([2.0, 1.0, 0.0])], temperature=1.0, return_logits=True)
    assert ans.choice == "a"
    assert abs(sum(ans.probabilities.values()) - 1) < 1e-3
    assert ans.probabilities["a"] == pytest.approx(0.6652, abs=1e-3)
    assert ans.logits == {"a": 2.0, "b": 1.0, "c": 0.0}
    flat = aggregate(q, probes, [np.array([2.0, 1.0, 0.0])], temperature=100.0)
    assert flat.confidence < 0.35  # high T flattens


def test_choice_debias_maps_rotated_labels_back():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": "", "c": ""})
    probes = build_probes(q, "s", debias=True)
    # model always prefers whatever option sits in position A (pure position bias)
    biased = [np.array([5.0, 0.0, 0.0]) for _ in probes]
    ans = aggregate(q, probes, biased, temperature=1.0)
    vals = list(ans.probabilities.values())
    assert max(vals) - min(vals) < 1e-6  # bias cancels out exactly
    # a real preference for "c" survives rotation
    real = []
    for p in probes:
        z = np.zeros(3)
        z[p.targets.index("c")] = 3.0
        real.append(z)
    assert aggregate(q, probes, real, temperature=1.0).choice == "c"


def test_bool_aggregate():
    q = BoolQuestion(instructions="x")
    probes = build_probes(q, "s")
    ans = aggregate(q, probes, [np.array([0.0, 1.0])], temperature=1.0)
    assert ans.value is False and ans.probability == pytest.approx(0.2689, abs=1e-3)
    assert ans.confidence == pytest.approx(1 - 0.2689, abs=1e-3)


def test_score_aggregate_expected_value():
    q = ScoreQuestion(instructions="x", min=1, max=3)
    probes = build_probes(q, "s")
    ans = aggregate(q, probes, [np.array([0.0, 0.0, 0.0])], temperature=1.0)
    assert ans.expected == pytest.approx(2.0, abs=1e-3) and ans.confidence == pytest.approx(1 / 3, abs=1e-3)
    ans = aggregate(q, probes, [np.array([0.0, 0.0, 10.0])], temperature=1.0)
    assert ans.score == 3 and ans.expected > 2.99
