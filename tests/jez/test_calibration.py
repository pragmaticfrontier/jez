from __future__ import annotations

import numpy as np
import pytest

from jez.calibration import BUCKETS, Calibration, Record, bucket, ece, fit


@pytest.mark.parametrize("qtype,n,expected", [
    ("noul", 2, "noul"), ("choice", 2, "choice:2"), ("choice", 3, "choice:3-5"), ("choice", 5, "choice:3-5"),
    ("choice", 6, "choice:6-26"), ("choice", 26, "choice:6-26"), ("choice", 27, "choice:27-255"), ("choice", 255, "choice:27-255"),
    ("score", 2, "score:2"), ("score", 5, "score:3-5"), ("score", 6, "score:6-10"), ("score", 10, "score:6-10"),
])
def test_buckets(qtype, n, expected):
    assert bucket(qtype, n) == expected


def test_fit_recovers_scale_per_bucket_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    records = []
    for n, true_t in ((4, 3.0), (2, 0.5)):
        z = rng.normal(size=(1500, n)) * 2
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        golds = [rng.choice(n, p=pi) for pi in p]
        logits = z * true_t
        records += [Record("choice", row - np.logaddexp.reduce(row), g) for row, g in zip(logits, golds)]
    calib = fit(records, model="m")
    assert abs(calib.temperature["choice:3-5"] - 3.0) / 3.0 < 0.15
    assert abs(calib.temperature["choice:2"] - 0.5) / 0.5 < 0.2
    assert calib.temperature["noul"] == 1.0  # no data, unchanged
    assert ece(records, calib) < ece(records, None)
    path = tmp_path / "calibration.json"
    calib.save(path)
    loaded = Calibration.load(path)
    assert loaded.model == "m" and loaded.temperature == calib.temperature and set(loaded.temperature) == set(BUCKETS)
