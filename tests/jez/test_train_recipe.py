"""The recipe options added after mix-v2: a step cap, adapter snapshots and the KL anchor to the frozen base."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from jez.labels import Label
from train.finetune import Item, batch_loss, label_logits, train_run

VOCAB, HIDDEN = 40, 8


class LoraLikeInner(torch.nn.Module):
    """A decoder with a zero-initialised delta branch that stands in for LoRA layers."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN)
        self.delta = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        torch.nn.init.zeros_(self.delta.weight)
        self.adapter_on = True

    def forward(self, input_ids, attention_mask, use_cache=False):
        h = self.proj(self.embed(input_ids))
        if self.adapter_on:
            h = h + self.delta(h)
        return SimpleNamespace(last_hidden_state=torch.tanh(h))


class LoraLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LoraLikeInner()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight = self.model.embed.weight

    @contextlib.contextmanager
    def disable_adapter(self):
        self.model.adapter_on = False
        try:
            yield
        finally:
            self.model.adapter_on = True

    def save_pretrained(self, path, **kwargs):  # the adapter save of the LoRA recipe
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")


class PlainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LoraLikeInner()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight = self.model.embed.weight

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeBackend:
    def __init__(self, model_cls, seed: int = 0):
        torch.manual_seed(seed)
        self.model = model_cls()
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
        out.append(Item(ids=torch.randint(13, VOCAB, (length,), generator=g).tolist(), labels=labels, gold=i % 3, teacher=None))
    return out


def args(**over):
    base = dict(batch=4, micro=4, micro_tokens=64, kl_weight=0.5, log_every=1, eval_every=2, gradient_checkpointing=False,
                seed=0, state_every=0, resume=False, steps=None, snapshot_every=0, base_kl_weight=0.0)
    base.update(over)
    return SimpleNamespace(**base)


# the "lora" recipe path in setup_model needs peft and a real transformer; these tests drive train_run with a model that
# already carries its adapter, through the "full" recipe (every parameter trainable), which exercises the same loop
RECIPE = {"method": "full", "lr": 1e-2}
TRAIN, VAL = items(40, 1), items(6, 2)  # 10 optimizer steps at batch 4


def test_step_cap_shortens_the_run_and_snapshots_land_in_the_release(tmp_path):
    be = FakeBackend(PlainModel)
    result = train_run(be, RECIPE, TRAIN, VAL, args(steps=6, snapshot_every=2), 1e-2, tmp_path)
    assert result["steps"] == 6
    assert [int(h["step"]) for h in result["history"]][-1] == 6
    snaps = sorted(p.name for p in (tmp_path / "snapshots").iterdir())
    assert snaps == ["step-2", "step-4", "step-6"]
    meta = json.loads((tmp_path / "snapshots" / "step-4" / "training.json").read_text())
    assert meta["step"] == 4 and meta["snapshot"] is True and meta["recipe"] == RECIPE


def test_step_cap_never_exceeds_one_epoch(tmp_path):
    result = train_run(FakeBackend(PlainModel), RECIPE, TRAIN, VAL, args(steps=500), 1e-2, None)
    assert result["steps"] == 10


def test_base_kl_is_zero_while_the_adapter_is_zero_and_grows_with_it():
    be = FakeBackend(LoraLikeModel)
    chunk = TRAIN[:4]
    plain, _, _ = batch_loss(be, be.model, chunk, 0.0)
    anchored, _, _ = batch_loss(be, be.model, chunk, 0.0, base_kl_weight=1.0)
    assert torch.allclose(plain, anchored, atol=1e-6)  # delta is zero: student equals base, KL vanishes

    torch.nn.init.normal_(be.inner.delta.weight, std=0.5)
    plain, _, _ = batch_loss(be, be.model, chunk, 0.0)
    anchored, _, _ = batch_loss(be, be.model, chunk, 0.0, base_kl_weight=2.0)
    # the expected anchor: 2 x mean over items of KL(base || student) over the label logits
    L = max(len(it.ids) for it in chunk)
    ids = torch.full((4, L), 0, dtype=torch.long)
    mask = torch.zeros((4, L), dtype=torch.long)
    for b, it in enumerate(chunk):
        ids[b, L - len(it.ids):] = torch.tensor(it.ids)
        mask[b, L - len(it.ids):] = 1
    with torch.no_grad():
        student = label_logits(be, be.inner(ids, mask).last_hidden_state[:, -1, :], chunk)
        with be.model.disable_adapter():
            base = label_logits(be, be.inner(ids, mask).last_hidden_state[:, -1, :], chunk)
    kls = []
    for zs, zb in zip(student, base):
        pb, logps = F.softmax(zb, 0), F.log_softmax(zs, 0)
        kls.append((pb * (torch.log(pb) - logps)).sum())
    want = plain + 2.0 * torch.stack(kls).mean()
    assert torch.allclose(anchored, want, atol=1e-5)
    assert anchored > plain
    assert be.inner.adapter_on  # the context manager restored the adapter


def test_base_kl_refuses_a_model_without_an_adapter(tmp_path):
    with pytest.raises(SystemExit, match="LoRA"):
        train_run(FakeBackend(PlainModel), RECIPE, TRAIN, VAL, args(base_kl_weight=0.5), 1e-2, None)


def test_training_with_the_anchor_moves_the_adapter_less(tmp_path):
    def final_delta_norm(weight):
        be = FakeBackend(LoraLikeModel)
        train_run(be, RECIPE, TRAIN, VAL, args(steps=10, base_kl_weight=weight), 5e-2, None)
        return float(be.inner.delta.weight.detach().norm())

    assert final_delta_norm(5.0) < final_delta_norm(0.0)
