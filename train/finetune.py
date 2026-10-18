"""Fine-tune a Jez base on label logits through the serving planner.

    # learning-rate sweep on 5% of the data, then the full run
    tools/gpu_guard.py --timeout 86400 -- .venv/bin/python -m train.finetune --model jez-small-1.0.0 \\
        --data train/data/mix-v1 --sweep 1e-5,2e-5,4e-5 --sweep-fraction 0.05
    tools/gpu_guard.py --timeout 86400 -- .venv/bin/python -m train.finetune --model jez-small-1.0.0 \\
        --data train/data/mix-v1 --lr 2e-5 --out releases/jez-small-1.0.0

Objective: cross-entropy on the gold label plus 0.5 times KL(teacher || student) over the label
distribution, both read at the answer position through the label-only head. Option order and label
letters are reshuffled for every example, so position carries no signal. Jez-small is a full
fine-tune of the decoder; Jez-large trains LoRA (rank 64, alpha 128) on the attention and MLP
projections of all layers with the token and per-layer embeddings frozen.

Schedule: cosine with 3% warm-up, one epoch, 64 sequences per optimizer step, taken in forward passes of at
most 64 sequences and 32,768 padded tokens (so a batch of long states uses several passes and a batch of
short ones a single wide pass) with gradient checkpointing, sequences up to 4,096 tokens. A 5% validation split picks the checkpoint; if no
checkpoint beats the base model on it, the release keeps the base model.

Recipe options added after the mix-v2 round (Gemma lost 2.4 Tier 1 points to a full-epoch LoRA at 2e-4 chosen on in-mix
validation): ``--steps`` caps the run (the cosine schedule then spans the cap), ``--snapshot-every`` saves the adapter
to <out>/snapshots/step-N for evaluation on Tier 1 samples (the engine loads an adapter directory directly), and
``--base-kl-weight`` adds that weight times KL(base || student) over the label distribution, where the base answer is
read from the same model with its LoRA layers disabled; it anchors the update to the base model's behaviour instead of
hard gold labels alone. LoRA recipes only.

A release run writes its training state (trainable weights, optimizer, step, best-so-far) to <out>/state.pt every
--state-every steps and resumes from it when restarted with the same data, seed and rate, so a power-off costs at most
that many steps. The file is removed when the run completes.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jez import models as registry
from jez.engine.torch_backend import TorchBackend, apply_memory_cap
from jez.labels import Label
from jez.prompt import choice_block, messages_for, noul_block, score_block, state_block
from jez.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, validate_request

log = logging.getLogger("train")

RECIPES = {
    "small": {"method": "full", "lr": 2e-5},
    "large": {"method": "lora", "lr": 1e-4, "rank": 64, "alpha": 128,
              "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
}


@dataclass
class Item:
    ids: list[int]
    labels: list[Label]
    gold: int
    teacher: np.ndarray | None


# ----------------------------------------------------------------------------- examples -> probes

def build_item(backend: TorchBackend, example: dict, rng: random.Random, max_tokens: int) -> Item | None:
    req = validate_request({"state": example["state"], "model": "jez-small-latest", "questions": {"q": example["question"]}})
    q = req.questions[0]
    table = backend.table
    if isinstance(q, ChoiceQuestion):
        perm = list(range(q.n))
        rng.shuffle(perm)  # option order
        q.options = [q.options[i] for i in perm]
        letters = [lab.text for lab in table.choice_labels(q.n)]
        rng.shuffle(letters)  # label letters
        text, label_texts, targets = choice_block(q, letters, reverse=False)
        by_text = {lab.text: lab for lab in table.choice}
    elif isinstance(q, NoulQuestion):
        text, label_texts, targets = noul_block(q, reverse=rng.random() < 0.5)
        by_text = {lab.text: lab for lab in table.noul}
    elif isinstance(q, ScoreQuestion):
        text, label_texts, targets = score_block(q, reverse=rng.random() < 0.5)
        by_text = {lab.text: lab for lab in table.score}
    else:
        return None
    gold = str(example["gold"])
    if gold not in targets:
        return None
    labels = [by_text[t] for t in label_texts]
    if any(lab.multi_token for lab in labels):
        return None
    ids = backend.encode(backend.render(messages_for(state_block(req.state), text)))
    if len(ids) > max_tokens:
        return None
    teacher = None
    if example.get("teacher"):
        t = example["teacher"]
        teacher = np.array([float(t.get(tg, 0.0)) for tg in targets], dtype=np.float64)
        teacher = teacher / teacher.sum() if teacher.sum() > 0 else None
    return Item(ids=ids, labels=labels, gold=targets.index(gold), teacher=teacher)


def label_logits(backend: TorchBackend, hidden: torch.Tensor, items: list[Item]) -> list[torch.Tensor]:
    """Per item: logits over its labels (logsumexp over surface forms), differentiable."""
    uniq = sorted({i for it in items for lab in it.labels for i in lab.ids})
    index = {tid: j for j, tid in enumerate(uniq)}
    W = backend.head_weight[uniq]
    logits = hidden.to(W.dtype) @ W.T if W.dtype != torch.float32 else hidden.float() @ W.T
    if backend.softcap:
        logits = torch.tanh(logits / backend.softcap) * backend.softcap
    logits = logits.float()
    out = []
    for b, it in enumerate(items):
        rows = [torch.logsumexp(logits[b, [index[i] for i in lab.ids]], 0) for lab in it.labels]
        out.append(torch.stack(rows))
    return out


def batch_loss(backend: TorchBackend, model, items: list[Item], kl_weight: float,
               base_kl_weight: float = 0.0) -> tuple[torch.Tensor, float, int]:
    L = max(len(it.ids) for it in items)
    pad = backend.pad_id
    input_ids = torch.full((len(items), L), pad, dtype=torch.long)
    mask = torch.zeros((len(items), L), dtype=torch.long)
    for b, it in enumerate(items):
        input_ids[b, L - len(it.ids):] = torch.tensor(it.ids)
        mask[b, L - len(it.ids):] = 1
    # backend.inner is the decoder; peft injects LoRA layers into the same module tree, so it serves both recipes.
    # Weights stay in fp32 (full fine-tune) and compute runs under bf16 autocast, so 2e-5 updates are not rounded away.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=backend.device.startswith("cuda"))
    with autocast:
        out = backend.inner(input_ids=input_ids.to(backend.device), attention_mask=mask.to(backend.device), use_cache=False)
        hidden = out.last_hidden_state[:, -1, :]
    base_logits = None
    if base_kl_weight > 0:
        # the base model's answer distribution, from the same weights with the LoRA layers switched off
        with torch.no_grad(), model.disable_adapter(), autocast:
            base_out = backend.inner(input_ids=input_ids.to(backend.device), attention_mask=mask.to(backend.device), use_cache=False)
            base_logits = label_logits(backend, base_out.last_hidden_state[:, -1, :].float(), items)
    losses, correct = [], 0
    for b, (it, z) in enumerate(zip(items, label_logits(backend, hidden.float(), items))):
        logp = F.log_softmax(z, dim=0)
        loss = -logp[it.gold]
        if it.teacher is not None and kl_weight > 0:
            t = torch.tensor(it.teacher, dtype=torch.float32, device=logp.device)
            loss = loss + kl_weight * (t * (torch.log(t.clamp_min(1e-9)) - logp)).sum()
        if base_logits is not None:
            pb = F.softmax(base_logits[b], dim=0)
            loss = loss + base_kl_weight * (pb * (torch.log(pb.clamp_min(1e-9)) - logp)).sum()
        losses.append(loss)
        correct += int(z.argmax().item() == it.gold)
    return torch.stack(losses).mean(), correct, len(items)


def micro_batches(items: list[Item], micro: int, micro_tokens: int) -> list[list[Item]]:
    """Split length-sorted items into forward passes of at most ``micro`` sequences and ``micro_tokens`` padded tokens.

    Activation memory grows with sequences times the longest sequence, so a fixed sequence count lets a batch of
    4,096-token states need ten times the memory of a batch of short ones. The token budget bounds that peak;
    short sequences still fill the whole ``micro`` width.
    """
    chunks: list[list[Item]] = []
    current: list[Item] = []
    longest = 0
    for it in items:
        n = len(it.ids)
        if current and (len(current) >= micro or (len(current) + 1) * max(longest, n) > micro_tokens):
            chunks.append(current)
            current, longest = [], 0
        current.append(it)
        longest = max(longest, n)
    if current:
        chunks.append(current)
    return chunks


@torch.no_grad()
def evaluate(backend, model, items: list[Item], micro: int, kl_weight: float, micro_tokens: int = 16384) -> dict:
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for chunk in micro_batches(sorted(items, key=lambda it: len(it.ids)), micro, micro_tokens):
        loss, correct, k = batch_loss(backend, model, chunk, kl_weight)
        tot_loss += float(loss) * k
        tot_correct += correct
        n += k
    model.train()
    return {"loss": tot_loss / max(1, n), "accuracy": tot_correct / max(1, n), "n": n}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
    return rows[:limit] if limit else rows


def prepare(backend, rows, rng, max_tokens, label) -> list[Item]:
    items = [build_item(backend, e, rng, max_tokens) for e in rows]
    kept = [it for it in items if it is not None]
    log.info("%s: %d of %d examples usable", label, len(kept), len(rows))
    return kept


def setup_model(backend: TorchBackend, recipe: dict):
    model = backend.model
    if recipe["method"] == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model, [p for p in model.parameters() if p.requires_grad]
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=recipe["rank"], lora_alpha=recipe["alpha"], target_modules=recipe["targets"], lora_dropout=0.0, bias="none")
    peft_model = get_peft_model(model, cfg)
    for name, p in peft_model.named_parameters():
        if "embed" in name:
            p.requires_grad_(False)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    log.info("LoRA trainable parameters: %.1fM", sum(p.numel() for p in params) / 1e6)
    return peft_model, params


def train_run(backend, recipe, train_items, val_items, args, lr: float, out: Path | None) -> dict:
    model, params = setup_model(backend, recipe)
    model.train()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # activations cost about 2.9 MB per token on Qwen3.5-0.8B (51 GiB for a 16k-token pass); recomputing them in
        # the backward pass brings that to 7 GiB for about 14% more compute, measured on 2026-10-16
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = max(1, len(train_items) // args.batch)
    if getattr(args, "steps", None):
        steps = min(steps, args.steps)  # a capped run: the schedule spans the cap, the data order is unchanged
    warm = max(1, int(0.03 * steps))
    base_kl = float(getattr(args, "base_kl_weight", 0.0) or 0.0)
    snapshot_every = int(getattr(args, "snapshot_every", 0) or 0)
    if base_kl > 0 and not hasattr(model, "disable_adapter"):
        raise SystemExit("--base-kl-weight needs a LoRA recipe: the base answer is read with the adapter disabled")

    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / warm
        progress = (step - warm) / max(1, steps - warm)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    state_path = out / "state.pt" if out is not None else None
    config = {"lr": lr, "steps": steps, "batch": args.batch, "seed": args.seed, "n_train": len(train_items)}
    # what every saved checkpoint records next to the recipe, so the model card can describe the run truthfully
    record = {"lr": lr, "steps": steps, "epoch_steps": max(1, len(train_items) // args.batch), "batch": args.batch,
              "n_train": len(train_items), "data": str(getattr(args, "data", "") or ""),
              "kl_weight": float(getattr(args, "kl_weight", 0.0) or 0.0), "base_kl_weight": base_kl}
    saved = load_state(model, opt, state_path, config) if state_path is not None and args.resume and state_path.exists() else None
    if saved is not None:
        start, best, history, base_val = saved["step"], saved["best"], saved["history"], saved["base_val"]
        log.info("resumed %s at step %d/%d (best validation %.4f at step %d)", state_path, start, steps, best["val"]["loss"], best["step"])
    else:
        start = 0
        base_val = evaluate(backend, model, val_items, args.micro, args.kl_weight, args.micro_tokens)
        log.info("base model on validation: loss %.4f acc %.4f", base_val["loss"], base_val["accuracy"])
        best = {"step": 0, "val": base_val, "is_base": True}
        history = [{"step": 0, "val": base_val}]
    order = list(range(len(train_items)))
    random.Random(args.seed).shuffle(order)  # the same order on every start, so a resumed run sees each example once
    t0 = time.perf_counter()
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = sorted((train_items[i] for i in order[step * args.batch:(step + 1) * args.batch]), key=lambda it: len(it.ids))
        running = 0.0
        for chunk in micro_batches(batch, args.micro, args.micro_tokens):
            loss, _, _ = batch_loss(backend, model, chunk, args.kl_weight, base_kl)
            share = len(chunk) / len(batch)  # the step's loss is the mean over its sequences, whatever the chunking
            (loss * share).backward()
            running += float(loss.detach()) * share
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0 or step == steps - 1:
            log.info("step %d/%d loss %.4f lr %.2e %.0fs", step + 1, steps, running, lr_at(step), time.perf_counter() - t0)
        if (step + 1) % args.eval_every == 0 or step == steps - 1:
            val = evaluate(backend, model, val_items, args.micro, args.kl_weight, args.micro_tokens)
            history.append({"step": step + 1, "train_loss": running, "val": val})
            log.info("  validation: loss %.4f acc %.4f (best %.4f at step %d)", val["loss"], val["accuracy"], best["val"]["loss"], best["step"])
            if val["loss"] < best["val"]["loss"]:
                best = {"step": step + 1, "val": val, "is_base": False}
                if out is not None:
                    save(model, backend, recipe, out, extra={**record, "step": step + 1, "val": val})
        if out is not None and snapshot_every and (step + 1) % snapshot_every == 0:
            snap = out / "snapshots" / f"step-{step + 1}"
            save(model, backend, recipe, snap, extra={**record, "step": step + 1, "snapshot": True})
            log.info("  snapshot saved to %s", snap)
        if state_path is not None and args.state_every and (step + 1) % args.state_every == 0 and step < steps - 1:
            save_state(model, opt, state_path, {"step": step + 1, "best": best, "history": history, "base_val": base_val, "config": config})
            log.info("  training state saved at step %d", step + 1)
    result = {"lr": lr, "steps": steps, "best": best, "history": history, "base_val": base_val}
    if state_path is not None and state_path.exists():
        state_path.unlink()  # the run is complete; the release directory holds the chosen weights
    if out is not None and best["is_base"]:
        log.warning("no checkpoint beat the base model on validation; the release keeps the base model")
        (out / "KEEP_BASE_MODEL").write_text(json.dumps(result, indent=1))
    return result


def save(model, backend, recipe, out: Path, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if recipe["method"] == "full":
        # a bf16 copy for serving; the fp32 master weights the optimizer keeps updating stay untouched
        serving = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(out / "weights", state_dict=serving, safe_serialization=True)
        del serving
        backend.tok.save_pretrained(out / "weights")
    else:
        model.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({"recipe": recipe, **extra}, indent=1))


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def save_state(model, opt, path: Path, payload: dict) -> None:
    """Write the trainable weights, the optimizer and ``payload`` to ``path`` atomically."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": trainable_state(model), "opt": opt.state_dict(), **payload}, tmp)
    os.replace(tmp, path)


def load_state(model, opt, path: Path, config: dict) -> dict:
    """Restore what save_state wrote; refuses a state from a run with a different ``config``."""
    st = torch.load(path, map_location="cpu", mmap=True)
    if st["config"] != config:
        raise SystemExit(f"{path} belongs to a different run ({st['config']} vs {config}); remove it to start over")
    if set(st["model"]) != set(trainable_state(model)):
        raise SystemExit(f"{path} holds other parameters than this run trains; remove it to start over")
    loaded = model.load_state_dict(st["model"], strict=False)  # frozen weights and buffers come from the base checkpoint
    if loaded.unexpected_keys:
        raise SystemExit(f"{path} has parameters this model lacks: {loaded.unexpected_keys[:5]}")
    opt.load_state_dict(st["opt"])
    payload = {k: st[k] for k in ("step", "best", "history", "base_val")}
    del st
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="jez-small-1.0.0", help="registry id; picks the base checkpoint and the recipe")
    ap.add_argument("--checkpoint", help="override the base checkpoint")
    ap.add_argument("--data", required=True, help="directory with train.jsonl and val.jsonl")
    ap.add_argument("--out", help="release directory; omit for sweeps")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--sweep", help="comma list of learning rates to try on --sweep-fraction of the data")
    ap.add_argument("--sweep-fraction", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64, help="sequences per optimizer step")
    ap.add_argument("--micro", type=int, default=64, help="sequences per forward pass, at most")
    ap.add_argument("--micro-tokens", type=int, default=32768, help="padded tokens per forward pass, at most")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--kl-weight", type=float, default=0.5, help="weight of KL(teacher || student) when examples carry teacher labels")
    ap.add_argument("--base-kl-weight", type=float, default=0.0,
                    help="weight of KL(base || student) read from the same model with its LoRA disabled (LoRA recipes only)")
    ap.add_argument("--steps", type=int, help="cap the number of optimizer steps (default: one epoch)")
    ap.add_argument("--snapshot-every", type=int, default=0, help="save the adapter to <out>/snapshots/step-N every N steps")
    ap.add_argument("--limit", type=int, help="use only the first N training examples (smoke tests)")
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute activations in the backward pass (default on; --no-gradient-checkpointing to disable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state-every", type=int, default=200, help="save resumable training state every N steps (0 disables)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="continue from <out>/state.pt when it exists (default on; --no-resume starts over)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    apply_memory_cap()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model_id = registry.resolve(args.model) or args.model
    spec = registry.MODELS[model_id]
    recipe = dict(RECIPES[spec.family])
    if args.base_kl_weight and recipe["method"] != "lora":
        raise SystemExit("--base-kl-weight needs a LoRA recipe: the base answer is read with the adapter disabled")
    checkpoint = args.checkpoint or spec.base
    backend = TorchBackend(checkpoint, dtype=torch.float32 if recipe["method"] == "full" else torch.bfloat16)
    data = Path(args.data)
    train_rows = read_jsonl(data / "train.jsonl", args.limit)
    val_rows = read_jsonl(data / "val.jsonl", args.val_limit)
    train_items = prepare(backend, train_rows, rng, args.max_tokens, "train")
    val_items = prepare(backend, val_rows, rng, args.max_tokens, "val")

    if args.sweep:
        results = []
        n = max(args.batch, int(len(train_items) * args.sweep_fraction))
        for lr in [float(x) for x in args.sweep.split(",")]:
            log.info("=== sweep lr %.2e on %d examples", lr, n)
            be = TorchBackend(checkpoint, dtype=backend.dtype) if results else backend  # every rate starts from fresh weights
            r = train_run(be, recipe, train_items[:n], val_items, args, lr, None)
            results.append(r)
            if be is not backend:
                be.release()
        best = min(results, key=lambda r: r["best"]["val"]["loss"])
        print(json.dumps({"sweep": [{"lr": r["lr"], "best_val": r["best"]["val"], "base_val": r["base_val"]} for r in results],
                          "chosen_lr": best["lr"]}, indent=1))
        return

    lr = args.lr or recipe["lr"]
    out = Path(args.out) if args.out else None
    result = train_run(backend, recipe, train_items, val_items, args, lr, out)
    print(json.dumps({"lr": lr, "steps": result["steps"], "best": result["best"], "base_val": result["base_val"]}, indent=1))
    backend.release()


if __name__ == "__main__":
    main()
