"""Scorer: a causal LM used as a parallel option scorer.

One batched forward pass evaluates every probe at once. We only read the
logits at the final position and only at the token ids of the allowed answer
labels, so the model can never produce anything outside the declared answers.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("sysone")

DEFAULT_MODEL = os.environ.get("SYSONE_MODEL", "Qwen/Qwen3-1.7B")


def _variants(label: str) -> set[str]:
    """Surface forms that mean the same answer (with/without leading space, casing for words)."""
    forms = {label, " " + label}
    if label.isalpha() and len(label) > 1:
        for v in (label.lower(), label.upper(), label.capitalize()):
            forms |= {v, " " + v}
    return forms


class Scorer:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_batch: int = 32,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.dtype = dtype
        self.max_batch = max_batch

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"dtype": dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(self.device).eval()
        log.info("loaded %s on %s (%s) in %.1fs", model_id, self.device, dtype, time.perf_counter() - t0)

        template = self.tok.chat_template or ""
        # Qwen3-style templates: switch thinking off so the first generated token is the answer.
        self.template_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
        self._label_cache: dict[str, list[int]] = {}
        self._logits_to_keep_ok = True

    # ------------------------------------------------------------------ prompts
    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tok.decode(ids[:max_tokens]) + " …[truncated]"

    # ------------------------------------------------------------------ labels
    def label_ids(self, label: str) -> list[int]:
        """Token ids that represent `label` as a single first token."""
        if label in self._label_cache:
            return self._label_cache[label]
        ids: set[int] = set()
        for form in _variants(label):
            toks = self.tok.encode(form, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"label {label!r} is not a single token for {self.model_id}")
        out = sorted(ids)
        self._label_cache[label] = out
        return out

    # ------------------------------------------------------------------ scoring
    def _last_logits(self, enc: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._logits_to_keep_ok:
            try:
                out = self.model(**enc, use_cache=False, logits_to_keep=1)
                return out.logits[:, -1, :].float()
            except TypeError:
                self._logits_to_keep_ok = False
        out = self.model(**enc, use_cache=False)
        return out.logits[:, -1, :].float()

    @torch.inference_mode()
    def score(self, prompts: list[str], label_sets: list[list[str]]) -> tuple[list[np.ndarray], int, int]:
        """Score every prompt's allowed labels.

        Returns (per-prompt label logits, total input tokens, number of forward passes).
        Each label logit is logsumexp over that label's surface-form token ids.
        """
        assert len(prompts) == len(label_sets)
        results: list[np.ndarray] = []
        total_tokens = 0
        passes = 0
        for start in range(0, len(prompts), self.max_batch):
            chunk = prompts[start : start + self.max_batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            total_tokens += int(enc["attention_mask"].sum().item())
            last = self._last_logits(enc).cpu().numpy()  # [B, V], one device sync
            passes += 1
            for row, labels in zip(last, label_sets[start : start + len(chunk)]):
                results.append(np.array([np.logaddexp.reduce(row[self.label_ids(l)]) for l in labels]))
        return results, total_tokens, passes

    def warmup(self) -> None:
        prompt = self.render([{"role": "user", "content": "Reply with A or B."}])
        self.score([prompt], [["A", "B"]])
