"""The trainer resumes a release run from its saved state and lands on the same weights as an uninterrupted run."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jez.labels import Label
from train.finetune import Item, load_state, save_state, train_run

VOCAB, HIDDEN = 40, 8


class FakeInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN)
        self.fail_after: int | None = None
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated power-off")
        return SimpleNamespace(last_hidden_state=torch.tanh(self.proj(self.embed(input_ids))))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInner()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight = self.model.embed.weight  # tied, as in both Jez bases

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        assert kwargs["state_dict"]["model.embed.weight"].dtype == torch.bfloat16


class FakeBackend:
    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.model = FakeModel()
        self.inner = self.model.model
        self.head_weight = self.model.lm_head.weight
        self.softcap = None
        self.pad_id = 0
        self.device = "cpu"
        self.tok = SimpleNamespace(save_pretrained=lambda path: None)


def items(n: int, seed: int) -> list[Item]:
    g = torch.Generator().manual_seed(seed)
    labels = [Label("A", (10,)), Label("B", (11,)), Label("C", (12,))]
    out = []
    for i in range(n):
        length = int(torch.randint(3, 9, (1,), generator=g))
        ids = torch.randint(13, VOCAB, (length,), generator=g).tolist()
        out.append(Item(ids=ids, labels=labels, gold=i % 3, teacher=None))
    return out


ARGS = SimpleNamespace(batch=4, micro=4, micro_tokens=64, kl_weight=0.5, log_every=1, eval_every=2, gradient_checkpointing=False,
                       seed=0, state_every=2, resume=True)
RECIPE = {"method": "full", "lr": 1e-2}
TRAIN, VAL = items(24, 1), items(6, 2)  # 6 optimizer steps


def weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_resume_reaches_the_weights_of_an_uninterrupted_run(tmp_path):
    whole = FakeBackend()
    train_run(whole, RECIPE, TRAIN, VAL, ARGS, 1e-2, tmp_path / "whole")
    assert not (tmp_path / "whole" / "state.pt").exists()  # a finished run leaves no state behind

    import train.finetune as ft

    crashed = FakeBackend()
    run = tmp_path / "run"
    original = ft.save_state
    seen = []

    def spy(model, opt, path, payload):
        original(model, opt, path, payload)
        seen.append(payload["step"])
        if payload["step"] == 4:
            crashed.inner.fail_after = crashed.inner.calls  # the next forward pass "loses power"

    ft.save_state = spy
    try:
        with pytest.raises(RuntimeError, match="power-off"):
            train_run(crashed, RECIPE, TRAIN, VAL, ARGS, 1e-2, run)
    finally:
        ft.save_state = original
    assert seen == [2, 4]
    assert (run / "state.pt").exists()

    resumed = FakeBackend()
    result = train_run(resumed, RECIPE, TRAIN, VAL, ARGS, 1e-2, run)
    assert not (run / "state.pt").exists()
    assert [h["step"] for h in result["history"]] == [0, 2, 4, 6]  # validation history spans both halves
    for name, w in weights(whole.model).items():
        assert torch.allclose(w, weights(resumed.model)[name], atol=1e-6), name


def test_state_from_another_run_is_refused(tmp_path):
    be = FakeBackend()
    opt = torch.optim.AdamW(be.model.parameters(), lr=1e-2)
    config = {"lr": 1e-2, "steps": 6, "batch": 4, "seed": 0, "n_train": 24}
    save_state(be.model, opt, tmp_path / "state.pt", {"step": 2, "best": {}, "history": [], "base_val": {}, "config": config})
    with pytest.raises(SystemExit, match="different run"):
        load_state(be.model, opt, tmp_path / "state.pt", {**config, "lr": 2e-2})
    fresh = FakeBackend(seed=7)
    opt2 = torch.optim.AdamW(fresh.model.parameters(), lr=1e-2)
    payload = load_state(fresh.model, opt2, tmp_path / "state.pt", config)
    assert payload["step"] == 2
    for name, w in weights(be.model).items():
        assert torch.equal(w, weights(fresh.model)[name]), name
