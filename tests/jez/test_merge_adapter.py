"""Merging a LoRA adapter yields a plain checkpoint that answers like the adapter-wrapped model."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from train.finetune import RECIPES
from train.merge_adapter import merge


def test_merged_weights_reproduce_the_adapter_logits(tmp_path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    base_dir = tmp_path / "base"
    LlamaForCausalLM(cfg).save_pretrained(base_dir)
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B").save_pretrained(base_dir)  # any tokenizer; it is copied through

    from peft import LoraConfig, get_peft_model

    recipe = RECIPES["large"]
    lora = get_peft_model(LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32),
                          LoraConfig(r=recipe["rank"], lora_alpha=recipe["alpha"], target_modules=recipe["targets"], lora_dropout=0.0))
    for name, p in lora.named_parameters():
        if "lora_B" in name:  # peft starts B at zero, which would make the merge trivially equal to the base
            torch.nn.init.normal_(p, std=0.05)
    lora.save_pretrained(tmp_path / "release" / "adapter")
    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        want = lora(input_ids=ids).logits

    out = merge(str(base_dir), tmp_path / "release" / "adapter", tmp_path / "release" / "weights", dtype=torch.float32, device="cpu")
    assert (out / "config.json").exists() and (out / "tokenizer_config.json").exists()
    merged = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32)
    with torch.no_grad():
        got = merged(input_ids=ids).logits
    assert torch.allclose(want, got, atol=1e-4)
    assert not any("lora" in k for k in merged.state_dict())


def test_engine_loads_an_adapter_directory_by_merging_it_in_memory(tmp_path):
    """eval.run_eval and the server can point straight at <release>/snapshots/step-N/adapter."""
    from jez.engine.torch_backend import adapter_base, load_text_model

    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    base_dir = tmp_path / "base"
    LlamaForCausalLM(cfg).save_pretrained(base_dir)
    from peft import LoraConfig, get_peft_model

    recipe = RECIPES["large"]
    lora = get_peft_model(LlamaForCausalLM.from_pretrained(base_dir, dtype=torch.float32),
                          LoraConfig(r=recipe["rank"], lora_alpha=recipe["alpha"], target_modules=recipe["targets"], lora_dropout=0.0))
    for name, p in lora.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.05)
    adapter = tmp_path / "snapshots" / "step-250" / "adapter"
    lora.save_pretrained(adapter)
    assert adapter_base(str(adapter)) == str(base_dir)
    assert adapter_base(str(base_dir)) is None

    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        want = lora(input_ids=ids).logits
    model, family, _ = load_text_model(str(adapter), torch.float32, "cpu")
    with torch.no_grad():
        got = model(input_ids=ids).logits
    assert family == "auto"
    assert torch.allclose(want, got, atol=1e-4)
    assert not any("lora" in k for k in model.state_dict())
