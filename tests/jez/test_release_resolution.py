"""Released weights resolve by model id, and their calibration file is found next to the weights or on the Hub."""
from pathlib import Path

import jez.models as registry
from jez.server import calibration_file


def test_registry_prefers_released_weights_over_the_base(monkeypatch):
    spec = registry.MODELS["jez-large-1.0.0"]
    monkeypatch.delenv("JEZ_CHECKPOINT_LARGE", raising=False)
    assert registry.parse_model_env("jez-large-1.0.0") == [("jez-large-1.0.0", spec.weights or spec.base)]
    released = registry.ModelSpec(**{**spec.__dict__, "weights": "pragmaticfrontier/jez-large-1.0.0"})
    monkeypatch.setitem(registry.MODELS, "jez-large-1.0.0", released)
    assert registry.parse_model_env("jez-large-1.0.0") == [("jez-large-1.0.0", "pragmaticfrontier/jez-large-1.0.0")]
    monkeypatch.setenv("JEZ_CHECKPOINT_LARGE", "/tmp/stand-in")
    assert registry.parse_model_env("jez-large-1.0.0") == [("jez-large-1.0.0", "/tmp/stand-in")]
    assert registry.parse_model_env("jez-large-1.0.0=Qwen/Qwen3-1.7B") == [("jez-large-1.0.0", "Qwen/Qwen3-1.7B")]


def test_calibration_file_resolution(tmp_path: Path):
    pattern = str(tmp_path / "releases" / "{model}" / "calibration.json")
    # 1. the configured release file wins
    local = tmp_path / "releases" / "jez-large-1.0.0" / "calibration.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}")
    assert calibration_file(pattern, "jez-large-1.0.0", "pragmaticfrontier/jez-large-1.0.0") == str(local)
    # 2. a checkpoint directory carrying its own calibration.json
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "calibration.json").write_text("{}")
    assert calibration_file(pattern, "jez-small-1.0.0", str(snap)) == str(snap / "calibration.json")
    # 3. a Hub id is fetched through the downloader; a failure falls back to the pattern path (T = 1)
    calls = []
    assert calibration_file(pattern, "jez-small-1.0.0", "pragmaticfrontier/jez-small-1.0.0",
                            download=lambda repo, name: calls.append((repo, name)) or "/cache/calibration.json") == "/cache/calibration.json"
    assert calls == [("pragmaticfrontier/jez-small-1.0.0", "calibration.json")]

    def failing(repo, name):
        raise RuntimeError("offline")
    assert calibration_file(pattern, "jez-small-1.0.0", "pragmaticfrontier/jez-small-1.0.0", download=failing) == pattern.format(model="jez-small-1.0.0")
    # 4. a plain base id without a calibration.json on the Hub also degrades to T = 1
    assert calibration_file(pattern, "jez-small-1.0.0", "Qwen/Qwen3.5-0.8B", download=failing).endswith("jez-small-1.0.0/calibration.json")
