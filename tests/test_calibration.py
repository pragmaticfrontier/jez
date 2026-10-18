import pytest
import numpy as np

from sysone.calibration import expected_calibration_error, fit_temperature, selective_table, softmax


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    n, c, true_t = 4000, 4, 3.0
    # draw well-calibrated logits z, then present them scaled up by true_t
    z = rng.normal(size=(n, c)) * 2
    p = softmax(z)
    labels = np.array([rng.choice(c, p=pi) for pi in p])
    t_hat = fit_temperature(z * true_t, labels)
    assert abs(t_hat - true_t) / true_t < 0.15


def test_ece_and_selective():
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 1, 0])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.05, abs=1e-6)
    rows = selective_table(conf, correct, thresholds=(0.0, 0.9))
    assert rows[1]["coverage"] == 0.5 and rows[1]["accuracy"] == 1.0
