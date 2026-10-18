"""Confidence calibration: temperature scaling plus the usual diagnostics.

The product of a decision model is the probability, so it has to be honest.
Temperature scaling fits a single scalar T per question type on labelled
examples so that softmax(logits / T) matches observed accuracy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

QUESTION_TYPES = ("choice", "bool", "score")


@dataclass
class Calibration:
    temperature: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in QUESTION_TYPES})
    model: str | None = None

    def temperature_for(self, qtype: str) -> float:
        return float(self.temperature.get(qtype, 1.0))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path is None or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        temps = {t: float(v) for t, v in data.get("temperature", {}).items()}
        return cls(temperature={**{t: 1.0 for t in QUESTION_TYPES}, **temps}, model=data.get("model"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"model": self.model, "temperature": self.temperature}, indent=2) + "\n")


def softmax(z: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


Logits = "np.ndarray | list[np.ndarray]"  # a 2-D array, or a list of 1-D rows of different lengths


def _rows(logits) -> list[np.ndarray]:
    if isinstance(logits, np.ndarray) and logits.ndim == 2:
        return list(logits.astype(np.float64))
    return [np.asarray(r, dtype=np.float64) for r in logits]


def per_example_stats(logits, labels, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(confidence, correct, nll) per example under softmax(logits / T). Rows may differ in length."""
    conf, correct, nlls = [], [], []
    for row, gold in zip(_rows(logits), np.asarray(labels, dtype=int)):
        p = softmax(row, temperature)
        conf.append(p.max())
        correct.append(float(p.argmax() == gold))
        nlls.append(-np.log(max(p[gold], 1e-12)))
    return np.array(conf), np.array(correct), np.array(nlls)


def nll(logits, labels, temperature: float) -> float:
    """Mean negative log-likelihood of the gold labels under softmax(logits / T)."""
    return float(per_example_stats(logits, labels, temperature)[2].mean())


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search for the T that minimises NLL (convex in log T).

    Accepts a 2-D array or a list of 1-D logit rows, so questions with different
    numbers of options can be pooled.
    """
    rows = _rows(logits)
    labels = np.asarray(labels, dtype=int)
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(rows, labels, math.exp(c)), nll(rows, labels, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(rows, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(rows, labels, math.exp(d))
    return float(math.exp((a + b) / 2))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def reliability_table(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            rows.append({"bin": f"({lo:.1f}, {hi:.1f}]", "n": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    return rows


def selective_table(confidence: np.ndarray, correct: np.ndarray,
                    thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)) -> list[dict]:
    """Coverage and accuracy if we only act when confidence >= threshold."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    rows = []
    for t in thresholds:
        mask = confidence >= t
        rows.append({"threshold": t, "coverage": float(mask.mean()),
                     "accuracy": float(correct[mask].mean()) if mask.any() else float("nan")})
    return rows
