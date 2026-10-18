from __future__ import annotations

import math

import numpy as np

from jez.calibration import Calibration
from jez.decider import Decider, average_orders
from jez.engine import FakeBackend
from jez.prompt import Probe
from jez.schema import validate_request


def _req(**questions):
    return validate_request({"state": "I was charged twice", "model": "jez-small-latest", "questions": questions})


def test_average_orders_maps_targets_back():
    p1 = Probe("q", "choice", "given", (), ("A", "B", "C"), ("x", "y", "z"))
    p2 = Probe("q", "choice", "reversed", (), ("A", "B", "C"), ("z", "y", "x"))
    targets, avg = average_orders([p1, p2], [np.log([0.7, 0.2, 0.1]), np.log([0.5, 0.3, 0.2])])
    assert targets == ["x", "y", "z"]
    assert avg[0] == np.mean([np.log(0.7), np.log(0.2)])  # x: A in p1, C in p2
    assert avg[2] == np.mean([np.log(0.1), np.log(0.5)])


def test_two_order_debiasing_cancels_pure_position_bias():
    """A backend that always favours position A should not favour any particular option after averaging."""
    biased = FakeBackend(scale=0.0, position_bias=4.0)
    dec = Decider(biased, "jez-small-1.0.0")
    req = _req(q={"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}})
    probs = dec.decide(req).answers["q"]["probabilities"]
    assert math.isclose(probs["a"], probs["c"], abs_tol=1e-6)  # the two ends see the bias once each
    assert probs["b"] < probs["a"]  # the middle option never sits in position A
    nd = Decider(biased, "jez-small-1.0.0", debias=False).decide(req).answers["q"]["probabilities"]
    assert nd["a"] > 0.9  # without debiasing the bias wins outright


def test_temperature_never_changes_the_winning_option():
    be = FakeBackend()
    req = _req(
        c={"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(8)}},
        s={"type": "score", "instructions": "How?", "criteria": [f"l{i}" for i in range(6)]},
        n={"type": "noul", "instructions": "Is it?"},
    )
    base = Decider(be, "jez-small-1.0.0").decide(req).answers
    for T in (0.3, 2.0, 9.0):
        calib = Calibration(temperature={b: T for b in Calibration().temperature})
        hot = Decider(be, "jez-small-1.0.0", calibration=calib).decide(req).answers
        assert hot["c"]["choice"] == base["c"]["choice"]
        assert max(hot["s"]["probabilities"], key=hot["s"]["probabilities"].get) == max(base["s"]["probabilities"], key=base["s"]["probabilities"].get)
        assert (hot["n"]["noul"] >= 0.5) == (base["n"]["noul"] >= 0.5)
        if T > 1:
            assert hot["c"]["confidence"] < base["c"]["confidence"]


def test_usage_counts_state_plus_questions_once():
    be = FakeBackend()
    dec = Decider(be, "jez-small-1.0.0")
    req = _req(a={"type": "noul", "instructions": "one two three"}, b={"type": "noul", "instructions": "four five"})
    d = dec.decide(req)
    n_state, n_q = dec.count(req)
    assert d.input_tokens == n_state + sum(n_q)
    assert d.probes == 4 and d.body()["usage"] == {"input_tokens": d.input_tokens, "output_tokens": 0}


def test_probabilities_sum_to_exactly_one_after_rounding():
    dec = Decider(FakeBackend(), "jez-small-1.0.0")
    req = _req(c={"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(255)}})
    probs = dec.decide(req).answers["c"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9
