"""Jev's answer formulas, from https://docs.typesafe.ai/confidence.

choice   confidence = (p_max - 1/n) / (1 - 1/n)
score    score      = sum_i i * p_i
         confidence = 1 - spread / even_spread, where
                      spread      = sum_i p_i * |i - peak|
                      even_spread = sum_i |i - (n - 1) / 2| / n
noul     no confidence field; the probability carries the uncertainty.
         |2p - 1| is the choice formula at n = 2 and is what evaluation uses.

All confidences are clamped to [0, 1].
"""
from __future__ import annotations

from collections.abc import Sequence


def _clamp(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return min(1.0, max(0.0, float(x)))


def choice_confidence(probabilities: Sequence[float]) -> float:
    n = len(probabilities)
    if n < 2:
        return 1.0
    p_max = max(probabilities)
    return _clamp((p_max - 1.0 / n) / (1.0 - 1.0 / n))


def score_value(probabilities: Sequence[float]) -> float:
    return float(sum(i * p for i, p in enumerate(probabilities)))


def score_confidence(probabilities: Sequence[float]) -> float:
    n = len(probabilities)
    if n < 2:
        return 1.0
    peak = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return _clamp(1.0 - spread / even_spread)


def noul_confidence(p_true: float) -> float:
    return _clamp(abs(2.0 * p_true - 1.0))
