"""The guard's thermal governor decides from three readings; None means the reading is unavailable."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("gpu_guard", Path(__file__).resolve().parents[2] / "tools" / "gpu_guard.py")
guard = importlib.util.module_from_spec(spec)
sys.modules["gpu_guard"] = guard
spec.loader.exec_module(guard)
LIMITS = dict(max_gpu_temp=84.0, max_soc_temp=90.0, min_tlimit=6.0)


def sample(gpu=70.0, tlimit=20.0, soc=75.0):
    return guard.Sample(1, gpu, tlimit, soc, 50.0)


def test_any_hot_reading_pauses():
    assert not guard.should_pause(sample(), **LIMITS)
    assert guard.should_pause(sample(gpu=84), **LIMITS)
    assert guard.should_pause(sample(tlimit=6), **LIMITS)
    assert guard.should_pause(sample(soc=90), **LIMITS)
    assert guard.should_pause(guard.Sample(1, None, 2.0, None, None), **LIMITS)


def test_resume_needs_every_reading_past_its_margin():
    assert guard.can_resume(sample(gpu=78, tlimit=12, soc=84), margin=6.0, **LIMITS)
    assert not guard.can_resume(sample(gpu=79, tlimit=12, soc=84), margin=6.0, **LIMITS)
    assert not guard.can_resume(sample(gpu=78, tlimit=11, soc=84), margin=6.0, **LIMITS)
    assert not guard.can_resume(sample(gpu=78, tlimit=12, soc=85), margin=6.0, **LIMITS)
    assert guard.can_resume(guard.Sample(1, None, None, None, None), margin=6.0, **LIMITS)


def test_hysteresis_band_is_not_empty():
    # a reading just below the pause line must not immediately satisfy resume, or the governor would flap
    assert not guard.should_pause(sample(gpu=83.9), **LIMITS)
    assert not guard.can_resume(sample(gpu=83.9), margin=6.0, **LIMITS)
