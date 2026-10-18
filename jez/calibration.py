"""Temperature calibration per question type and option-count bucket.

Buckets:
    noul
    choice:2, choice:3-5, choice:6-26, choice:27-255
    score:2, score:3-5, score:6-10

A temperature divides the averaged label log-probabilities before the softmax.
It is monotone, so it never changes the winning option. Temperatures are fitted
after training on a calibration split that is separate from every test split,
and ship inside the model release as ``calibration.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from sysone.calibration import expected_calibration_error, fit_temperature, softmax  # numerics shared with the prototype

BUCKETS = ("noul", "choice:2", "choice:3-5", "choice:6-26", "choice:27-255", "score:2", "score:3-5", "score:6-10")


def bucket(qtype: str, n: int) -> str:
    if qtype == "noul":
        return "noul"
    if qtype == "choice":
        if n <= 2:
            return "choice:2"
        if n <= 5:
            return "choice:3-5"
        if n <= 26:
            return "choice:6-26"
        return "choice:27-255"
    if qtype == "score":
        if n <= 2:
            return "score:2"
        if n <= 5:
            return "score:3-5"
        return "score:6-10"
    raise ValueError(qtype)


@dataclass
class Calibration:
    temperature: dict[str, float] = field(default_factory=lambda: {b: 1.0 for b in BUCKETS})
    model: str | None = None

    def temperature_for(self, qtype: str, n: int) -> float:
        return float(self.temperature.get(bucket(qtype, n), 1.0))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path is None or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        temps = {b: 1.0 for b in BUCKETS}
        temps.update({k: float(v) for k, v in data.get("temperature", {}).items()})
        return cls(temperature=temps, model=data.get("model"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"model": self.model, "temperature": self.temperature}, indent=2) + "\n")


@dataclass
class Record:
    """One scored question: averaged label log-probabilities and the gold index."""
    qtype: str
    logprobs: np.ndarray
    gold: int


def fit(records: Iterable[Record], model: str | None = None, min_examples: int = 50) -> Calibration:
    """Fit one temperature per bucket; buckets with fewer than ``min_examples`` keep T = 1."""
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(bucket(r.qtype, len(r.logprobs)), []).append(r)
    calib = Calibration(model=model)
    for b, rows in groups.items():
        if len(rows) < min_examples:
            continue
        T = fit_temperature([r.logprobs for r in rows], [r.gold for r in rows])
        calib.temperature[b] = round(float(T), 4)
    return calib


def ece(records: Iterable[Record], calib: Calibration | None = None) -> float:
    conf, correct = [], []
    for r in records:
        T = calib.temperature_for(r.qtype, len(r.logprobs)) if calib else 1.0
        p = softmax(r.logprobs, T)
        conf.append(float(p.max()))
        correct.append(float(int(p.argmax()) == r.gold))
    if not conf:
        return float("nan")
    return expected_calibration_error(np.array(conf), np.array(correct))
