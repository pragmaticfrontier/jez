"""Torch backend: text decoder only, label-only output head, shared state cache.

Loading
    Qwen3.5 and Gemma 4 checkpoints are multimodal. The loader builds the text
    decoder class (``Qwen3_5ForCausalLM``, ``Gemma4ForCausalLM``) and maps the
    ``model.language_model.*`` weights onto it; the vision and audio encoders
    are never materialised.

Scoring
    All probes of a request share one token prefix (system prompt + state).
    The prefix runs once with a KV cache; the cache is replicated across the
    batch and every question suffix runs against it. Per-row logits are read at
    the last real token of each suffix. The output head multiplies the final
    hidden state by the label rows of the (tied) embedding only, with Gemma's
    final soft-cap applied to those rows.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..labels import Label, LabelTable, build_label_table

log = logging.getLogger("jez.engine")

_GEMMA_KEY_MAPPING = {r"^model\.language_model\.": "model."}
_SKIP_PREFIXES = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio.", "model.visual.")
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

DEFAULT_CUDA_FRACTION = 0.6  # share of device memory this process may take; the rest stays for the OS


def apply_memory_cap(fraction: float | None = None) -> float | None:
    """Cap PyTorch's CUDA allocator so over-allocation raises OutOfMemoryError instead of exhausting unified memory.

    On a DGX Spark the GPU shares system memory and GPU allocations escape cgroup limits, so this cap is the
    only process-level guard. ``JEZ_CUDA_MEMORY_FRACTION=1`` disables it.
    """
    if not torch.cuda.is_available():
        return None
    frac = fraction if fraction is not None else float(os.environ.get("JEZ_CUDA_MEMORY_FRACTION", DEFAULT_CUDA_FRACTION))
    if 0 < frac < 1:
        torch.cuda.set_per_process_memory_fraction(frac)
        return frac
    return None


def adapter_base(checkpoint: str) -> str | None:
    """The base checkpoint a LoRA adapter directory was trained on (peft's adapter_config.json), else None.

    An adapter directory is accepted wherever a checkpoint is: the base is loaded and the adapter merged in memory,
    so a training snapshot can be evaluated without writing merged weights to disk."""
    cfg = Path(checkpoint) / "adapter_config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text())["base_model_name_or_path"]
    return None


def _safetensors_files(checkpoint: str) -> list[Path]:
    from transformers.utils.hub import cached_file

    checkpoint = adapter_base(checkpoint) or checkpoint
    path = Path(checkpoint)
    if path.is_dir():
        return sorted(path.glob("*.safetensors"))
    index = None
    try:
        index = cached_file(checkpoint, "model.safetensors.index.json")
    except Exception:
        pass
    if index:
        names = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    return [Path(cached_file(checkpoint, n)) for n in names]


def estimate_text_weight_bytes(checkpoint: str, dtype: torch.dtype) -> int:
    """Bytes the text decoder's weights take in ``dtype``, from the safetensors headers (no tensor is read)."""
    per_param = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}.get(dtype, 4)
    total = 0
    for f in _safetensors_files(checkpoint):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__" or name.startswith(_SKIP_PREFIXES):
                continue
            count = 1
            for d in meta["shape"]:
                count *= d
            total += count * per_param
    return total


def check_fits(checkpoint: str, dtype: torch.dtype, headroom: float = 1.3) -> None:
    """Refuse to load a model whose weights (times headroom for activations) would not fit under the allocator cap."""
    if not torch.cuda.is_available():
        return
    need = estimate_text_weight_bytes(checkpoint, dtype) * headroom
    free, total = torch.cuda.mem_get_info()
    frac = float(os.environ.get("JEZ_CUDA_MEMORY_FRACTION", DEFAULT_CUDA_FRACTION))
    budget = (total * frac if 0 < frac < 1 else total) - torch.cuda.memory_reserved()
    budget = min(budget, free)
    if need > budget:
        raise MemoryError(
            f"{checkpoint} in {str(dtype).replace('torch.', '')} needs about {need / 1e9:.1f} GB (weights x {headroom}) "
            f"but only {budget / 1e9:.1f} GB fit under the allocator cap; release another model or raise "
            f"JEZ_CUDA_MEMORY_FRACTION")
    log.info("memory preflight: %s needs ~%.1f GB, budget %.1f GB", checkpoint, need / 1e9, budget / 1e9)


def _resolve_config(checkpoint: str) -> dict:
    from transformers.utils.hub import cached_file

    checkpoint = adapter_base(checkpoint) or checkpoint
    path = Path(checkpoint)
    if path.is_dir():
        return json.loads((path / "config.json").read_text())
    return json.loads(Path(cached_file(checkpoint, "config.json")).read_text())


def load_text_model(checkpoint: str, dtype: torch.dtype, device: str, attn_implementation: str | None = None):
    """Load only the text decoder of a checkpoint (or of a LoRA adapter directory, merged into its base in memory).

    Returns (model, architecture, softcap)."""
    from transformers import AutoModelForCausalLM

    adapter = checkpoint if adapter_base(checkpoint) else None
    checkpoint = adapter_base(checkpoint) or checkpoint
    cfg = _resolve_config(checkpoint)
    arch = (cfg.get("architectures") or ["?"])[0]
    if device.startswith("cuda"):
        apply_memory_cap()
        check_fits(checkpoint, dtype)
    kwargs = {"dtype": dtype, "device_map": device}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    text_cfg = cfg.get("text_config", cfg)
    softcap = text_cfg.get("final_logit_softcapping")
    if arch.startswith("Gemma4"):
        from transformers import Gemma4ForCausalLM

        model = Gemma4ForCausalLM.from_pretrained(checkpoint, key_mapping=_GEMMA_KEY_MAPPING, **kwargs)
        family = "gemma4"
    elif arch.startswith("Qwen3_5"):
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(checkpoint, **kwargs)
        family = "qwen3_5"
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
        family = "auto"
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        log.info("merged adapter %s into %s in memory", adapter, checkpoint)
    return model.eval(), family, softcap


def _expand_batch(obj, n: int) -> None:
    """Repeat every batch-first tensor held by a cache layer (directly or inside a dict) n times."""
    for key, value in list(vars(obj).items()):
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and value.shape[0] == 1:
                setattr(obj, key, value.repeat_interleave(n, dim=0))
        elif isinstance(value, dict):
            for k, v in list(value.items()):
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
                    value[k] = v.repeat_interleave(n, dim=0)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
                    value[i] = v.repeat_interleave(n, dim=0)


def expand_cache(cache, n: int):
    """A deep copy of a batch-1 cache replicated to batch n (the original is left untouched)."""
    new = copy.deepcopy(cache)
    layers = getattr(new, "layers", None)
    if layers is None:
        raise TypeError(f"unsupported cache type {type(cache).__name__}")
    for layer in layers:
        _expand_batch(layer, n)
    return new


def common_prefix_len(rows: Sequence[Sequence[int]]) -> int:
    if not rows:
        return 0
    shortest = min(len(r) for r in rows)
    first = rows[0]
    k = 0
    while k < shortest and all(r[k] == first[k] for r in rows):
        k += 1
    return k


class TorchBackend:
    name = "torch"

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_batch: int = 16,
        share_state: bool = True,
        min_shared_prefix: int = 16,
        attn_implementation: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.dtype = dtype
        self.max_batch = max_batch
        self.share_state = share_state
        self.min_shared_prefix = min_shared_prefix

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(adapter_base(checkpoint) or checkpoint)  # an adapter directory has no tokenizer
        self.model, self.family, self.softcap = load_text_model(checkpoint, dtype, self.device, attn_implementation)
        self.inner = self.model.model  # the decoder without the output head
        self.head_weight = self.model.lm_head.weight  # tied to the embedding for both bases
        self.table: LabelTable = build_label_table(self.tok, name=checkpoint)
        template = self.tok.chat_template or ""
        self.template_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        log.info("loaded %s (%s) on %s %s in %.1fs; %d single-token choice labels",
                 checkpoint, self.family, self.device, dtype, time.perf_counter() - t0, self.table.n_single_token)

    # ------------------------------------------------------------------ text
    def render(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True,
                                            **self.template_kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    # ------------------------------------------------------------------ head
    def _label_logits(self, hidden: torch.Tensor, labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        """hidden: [B, H] final hidden states. Returns per-row restricted log-probabilities over the row's labels."""
        uniq: list[int] = sorted({i for row in labels for lab in row for i in lab.ids})
        index = {tid: j for j, tid in enumerate(uniq)}
        W = self.head_weight[uniq]  # [U, H]
        logits = (hidden.to(W.dtype) @ W.T).float()  # [B, U]
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        logits = logits.cpu().numpy().astype(np.float64)
        out = []
        for row, labs in zip(logits, labels):
            z = np.array([np.logaddexp.reduce([row[index[i]] for i in lab.ids]) for lab in labs])
            z = z - z.max()
            out.append(z - np.log(np.exp(z).sum()))
        return out

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def score(self, prompts: Sequence[str], labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        assert len(prompts) == len(labels)
        if not prompts:
            return []
        ids = [self.encode(p) for p in prompts]
        multi = [i for i, labs in enumerate(labels) if any(lab.multi_token for lab in labs)]
        results: list[np.ndarray | None] = [None] * len(prompts)
        plain = [i for i in range(len(prompts)) if i not in set(multi)]
        if plain:
            rows = [ids[i] for i in plain]
            labs = [labels[i] for i in plain]
            shared = self.share_state and len(rows) > 1
            prefix = common_prefix_len(rows) if shared else 0
            prefix = min(prefix, min(len(r) for r in rows) - 1)
            if shared and prefix >= self.min_shared_prefix:
                scored = self._score_shared(rows, labs, prefix)
            else:
                scored = self._score_unshared(rows, labs)
            for i, z in zip(plain, scored):
                results[i] = z
        for i in multi:
            results[i] = self._score_teacher_forced(ids[i], labels[i])
        return [r for r in results if r is not None]

    def _hidden_unshared(self, rows: Sequence[Sequence[int]]) -> torch.Tensor:
        """Final hidden state at the last token of each (left-padded) row."""
        L = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), L), dtype=torch.long)
        for b, r in enumerate(rows):
            input_ids[b, L - len(r):] = torch.tensor(r)
            mask[b, L - len(r):] = 1
        out = self.inner(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device), use_cache=False)
        return out.last_hidden_state[:, -1, :]

    def _score_unshared(self, rows: Sequence[Sequence[int]], labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for start in range(0, len(rows), self.max_batch):
            chunk = rows[start:start + self.max_batch]
            hidden = self._hidden_unshared(chunk)
            out.extend(self._label_logits(hidden, labels[start:start + len(chunk)]))
        return out

    def _score_shared(self, rows: Sequence[Sequence[int]], labels: Sequence[Sequence[Label]], prefix: int) -> list[np.ndarray]:
        prefix_ids = torch.tensor([rows[0][:prefix]], dtype=torch.long, device=self.device)
        pre = self.inner(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True)
        cache = pre.past_key_values
        out: list[np.ndarray] = []
        for start in range(0, len(rows), self.max_batch):
            chunk = rows[start:start + self.max_batch]
            B = len(chunk)
            suffixes = [r[prefix:] for r in chunk]
            Ls = max(len(s) for s in suffixes)
            input_ids = torch.full((B, Ls), self.pad_id, dtype=torch.long)
            mask = torch.ones((B, prefix + Ls), dtype=torch.long)
            last = torch.zeros(B, dtype=torch.long)
            for b, s in enumerate(suffixes):
                input_ids[b, :len(s)] = torch.tensor(s)
                mask[b, prefix + len(s):] = 0
                last[b] = len(s) - 1
            batch_cache = expand_cache(cache, B)
            out_b = self.inner(
                input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device),
                past_key_values=batch_cache, use_cache=True,
                cache_position=torch.arange(prefix, prefix + Ls, device=self.device),
            )
            hs = out_b.last_hidden_state  # [B, Ls, H]
            hidden = hs[torch.arange(B, device=hs.device), last.to(hs.device)]
            out.extend(self._label_logits(hidden, labels[start:start + B]))
            del batch_cache
        return out

    def _score_teacher_forced(self, prompt_ids: Sequence[int], labels: Sequence[Label]) -> np.ndarray:
        """Labels that are not single tokens: sum log-probabilities over their tokens.

        The first token of every label is normalised over the first tokens of all labels (as in the
        single-token case); later tokens are normalised over the full vocabulary.
        """
        first_ids = [lab.ids[0] for lab in labels]
        hidden = self._hidden_unshared([prompt_ids])
        first_rows = [[Label(text=lab.text, ids=(lab.ids[0],)) for lab in labels]]
        first_lp = self._label_logits(hidden, first_rows)[0]
        total = np.array(first_lp, dtype=np.float64)
        for j, lab in enumerate(labels):
            if len(lab.ids) <= 1:
                continue
            ids = list(prompt_ids) + list(lab.ids[:-1])
            out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
            logp = torch.log_softmax(out.logits[0, len(prompt_ids) - 1:, :].float(), dim=-1)
            for k, tid in enumerate(lab.ids[1:], start=1):
                total[j] += float(logp[k, tid])
        total = total - total.max()
        return total - np.log(np.exp(total).sum())

    def warmup(self) -> None:
        prompt = self.render([{"role": "user", "content": "Reply with A or B."}])
        self.score([prompt, prompt + " "], [self.table.choice_labels(2)] * 2)

    def release(self) -> None:
        """Drop the weights and return their memory. Call this before loading another model in the same process."""
        for name in ("head_weight", "inner", "model"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
