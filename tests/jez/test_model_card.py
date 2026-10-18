"""The model card describes the run that was actually made, from training.json, never from a fixed template line."""
from pathlib import Path

from tools.publish_hf import front_matter, plan
from train.model_card import describe_objective, describe_schedule, describe_selection


def test_objective_names_only_the_terms_that_were_used():
    gold_only = describe_objective({"kl_weight": 0.5, "base_kl_weight": 0.0}, "none (gold labels only)")
    assert gold_only.startswith("cross-entropy on the gold label, over label logits")
    assert "KL" not in gold_only
    anchored = describe_objective({"kl_weight": 0.5, "base_kl_weight": 0.5}, "none (gold labels only)")
    assert "0.5 × KL(base ‖ student)" in anchored and "teacher" not in anchored
    taught = describe_objective({"kl_weight": 0.5}, "Qwen/Qwen3.8-27B")
    assert "0.5 × KL(teacher ‖ student)" in taught


def test_schedule_reports_a_capped_run_as_a_fraction_of_an_epoch():
    capped = describe_schedule({"steps": 1000, "epoch_steps": 4359, "batch": 64, "lr": 5e-05}, "chosen on Tier 1 pilots")
    assert "1,000 steps of 64 sequences (23% of an epoch)" in capped and capped.endswith("5e-05 chosen on Tier 1 pilots")
    full = describe_schedule({"steps": 4359, "epoch_steps": 4359, "batch": 64, "lr": 1e-05}, "by sweep")
    assert "1 epoch (4,359 steps of 64 sequences)" in full
    legacy = describe_schedule({"lr": 2e-4}, "by sweep")  # records written before the run facts were saved
    assert "1 epoch, 64 sequences per step" in legacy


def test_selection_reports_step_and_validation_or_the_kept_base():
    assert describe_selection({}, True, "").startswith("no checkpoint beat the base model")
    text = describe_selection({"step": 1000, "val": {"loss": 0.40602, "accuracy": 0.837}}, False, "snapshot picked on Tier 1 at n = 300")
    assert "step 1000 (loss 0.4060, accuracy 83.7%)" in text and text.endswith("snapshot picked on Tier 1 at n = 300")


def test_publish_plan_puts_weights_at_the_root_and_refuses_without_a_card(tmp_path: Path):
    rel = tmp_path / "rel"
    (rel / "weights").mkdir(parents=True)
    (rel / "weights" / "config.json").write_text("{}")
    (rel / "weights" / "model.safetensors").write_bytes(b"0")
    (rel / "adapter").mkdir()
    (rel / "adapter" / "adapter_config.json").write_text("{}")
    (rel / "calibration.json").write_text("{}")
    import pytest
    with pytest.raises(SystemExit):
        plan(rel)
    (rel / "MODEL_CARD.md").write_text("# card")
    dests = [d for _, d in plan(rel)]
    assert dests == ["config.json", "model.safetensors", "adapter/adapter_config.json", "calibration.json", "MODEL_CARD.md"]


def test_front_matter_is_a_yaml_header_with_the_base_and_licence():
    fm = front_matter("google/gemma-4-E4B-it", "apache-2.0", "https://example/licence", ["jez", "jev"])
    assert fm.startswith("---\nlicense: apache-2.0\nlicense_link: https://example/licence\nbase_model: google/gemma-4-E4B-it\n")
    assert fm.rstrip().endswith("---") and "  - jez\n  - jev" in fm and "  - uk" in fm


def test_notes_become_known_limit_bullets(tmp_path: Path, monkeypatch):
    import json, subprocess, sys
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "calibration.json").write_text(json.dumps({"model": "jez-large-1.0.0", "temperature": {"noul": 1.0}}))
    (rel / "training.json").write_text(json.dumps({"recipe": {"method": "lora", "rank": 64, "alpha": 128}, "step": 1000, "lr": 5e-5,
                                                   "val": {"loss": 0.4, "accuracy": 0.83}, "base_kl_weight": 0.5}))
    cmd = [sys.executable, "-m", "train.model_card", "--release", str(rel), "--model-id", "jez-large-1.0.0",
           "--note", "Prompt-injection detection sits below the base.", "--note", "Second note."]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], capture_output=True)
    card = (rel / "MODEL_CARD.md").read_text()
    assert "- Prompt-injection detection sits below the base.\n- Second note." in card
    assert "0.5 × KL(base ‖ student)" in card and "{" not in card.split("## Known limits")[0].split("| Method |")[0]
