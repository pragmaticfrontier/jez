"""MLX backend for Apple Silicon, through mlx-lm.

The scoring contract is the same as the torch backend's: shared prefix cache,
batched suffixes, label-only output head, restricted log-softmax over labels.

This module is written against mlx-lm's public model API (``mlx_lm.load``,
``model.model(inputs, cache=...)``, ``make_prompt_cache``) and the cache
classes in ``mlx_lm.models.cache`` (KVCache, RotatingKVCache for sliding
windows, ArraysCache / MambaCache for linear attention). It cannot run on the
Linux machine Jez was developed on, so it is exercised by
``eval/backend_agreement.py`` on a Mac: that script dumps label log-probabilities
for the conformance requests from each backend and checks they agree within
0.02.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Sequence

import numpy as np

from ..labels import Label, LabelTable, build_label_table

log = logging.getLogger("jez.engine")


def _expand_arrays(obj, n: int) -> None:
    import mlx.core as mx

    for key, value in list(vars(obj).items()):
        if isinstance(value, mx.array):
            if value.ndim >= 1 and value.shape[0] == 1:
                setattr(obj, key, mx.repeat(value, n, axis=0))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, mx.array) and v.ndim >= 1 and v.shape[0] == 1:
                    value[i] = mx.repeat(v, n, axis=0)
        elif isinstance(value, dict):
            for k, v in list(value.items()):
                if isinstance(v, mx.array) and v.ndim >= 1 and v.shape[0] == 1:
                    value[k] = mx.repeat(v, n, axis=0)


def expand_cache(cache: list, n: int) -> list:
    new = copy.deepcopy(cache)
    for layer in new:
        _expand_arrays(layer, n)
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


class MlxBackend:
    name = "mlx"

    def __init__(self, checkpoint: str, max_batch: int = 16, share_state: bool = True, min_shared_prefix: int = 16,
                 **_: object) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.checkpoint = checkpoint
        self.device = str(mx.default_device())
        self.max_batch = max_batch
        self.share_state = share_state
        self.min_shared_prefix = min_shared_prefix
        t0 = time.perf_counter()
        self.model, self.tok = load(checkpoint)
        self.inner = self.model.model
        args = getattr(self.model, "args", None)
        self.softcap = getattr(args, "final_logit_softcapping", None)
        tied = getattr(args, "tie_word_embeddings", True) or not hasattr(self.model, "lm_head")
        self.head_weight = self.inner.embed_tokens.weight if tied else self.model.lm_head.weight
        self.table: LabelTable = build_label_table(self.tok, name=checkpoint)
        template = getattr(self.tok, "chat_template", "") or ""
        self.template_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
        self.pad_id = getattr(self.tok, "pad_token_id", None) or getattr(self.tok, "eos_token_id", 0) or 0
        log.info("loaded %s on mlx in %.1fs", checkpoint, time.perf_counter() - t0)

    def render(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True,
                                            **self.template_kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def _label_logits(self, hidden, labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        import mlx.core as mx

        uniq = sorted({i for row in labels for lab in row for i in lab.ids})
        index = {tid: j for j, tid in enumerate(uniq)}
        W = self.head_weight[mx.array(uniq)]
        logits = (hidden.astype(W.dtype) @ W.T).astype(mx.float32)
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        logits = np.array(logits, dtype=np.float64)
        out = []
        for row, labs in zip(logits, labels):
            z = np.array([np.logaddexp.reduce([row[index[i]] for i in lab.ids]) for lab in labs])
            z = z - z.max()
            out.append(z - np.log(np.exp(z).sum()))
        return out

    def _hidden_unshared(self, rows: Sequence[Sequence[int]]):
        import mlx.core as mx

        L = max(len(r) for r in rows)
        ids = np.full((len(rows), L), self.pad_id, dtype=np.int32)
        for b, r in enumerate(rows):
            ids[b, L - len(r):] = r
        hs = self.inner(mx.array(ids))  # left padding: last position is the answer position for every row
        return hs[:, -1, :]

    def score(self, prompts: Sequence[str], labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = [self.encode(p) for p in prompts]
        if not ids:
            return []
        shared = self.share_state and len(ids) > 1
        prefix = min(common_prefix_len(ids), min(len(r) for r in ids) - 1) if shared else 0
        out: list[np.ndarray] = []
        if shared and prefix >= self.min_shared_prefix:
            cache = make_prompt_cache(self.model)
            self.inner(mx.array([ids[0][:prefix]]), cache=cache)
            mx.eval([getattr(c, "state", None) for c in cache])
            for start in range(0, len(ids), self.max_batch):
                chunk = ids[start:start + self.max_batch]
                B = len(chunk)
                suffixes = [r[prefix:] for r in chunk]
                Ls = max(len(s) for s in suffixes)
                arr = np.full((B, Ls), self.pad_id, dtype=np.int32)
                last = np.zeros(B, dtype=np.int32)
                for b, s in enumerate(suffixes):
                    arr[b, :len(s)] = s
                    last[b] = len(s) - 1
                hs = self.inner(mx.array(arr), cache=expand_cache(cache, B))
                hidden = hs[mx.arange(B), mx.array(last)]
                out.extend(self._label_logits(hidden, labels[start:start + B]))
        else:
            for start in range(0, len(ids), self.max_batch):
                chunk = ids[start:start + self.max_batch]
                out.extend(self._label_logits(self._hidden_unshared(chunk), labels[start:start + len(chunk)]))
        return out

    def warmup(self) -> None:
        prompt = self.render([{"role": "user", "content": "Reply with A or B."}])
        self.score([prompt], [self.table.choice_labels(2)])
