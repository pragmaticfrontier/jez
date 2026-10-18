"""Backends: a backend renders chat messages, counts tokens and scores labels.

``score(prompts, labels)`` returns, per prompt, the log-probabilities of its
labels, normalised over those labels only (a restricted log-softmax). Nothing
is ever sampled.

* :class:`TorchBackend` (``jez.engine.torch_backend``): CUDA or CPU through transformers
* :class:`MlxBackend` (``jez.engine.mlx_backend``): Apple Silicon through mlx-lm
* :class:`FakeBackend`: deterministic hash-based logits for tests and the conformance suite
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from ..labels import Label, LabelTable, build_label_table


class Backend(Protocol):
    name: str
    checkpoint: str
    device: str
    table: LabelTable

    def render(self, messages: Sequence[dict[str, str]]) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def score(self, prompts: Sequence[str], labels: Sequence[Sequence[Label]]) -> list[np.ndarray]: ...


class _WordTokenizer:
    """A stand-in tokenizer: every whitespace-separated word is one token, every label is one token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "big") for w in text.split()]


class FakeBackend:
    """Deterministic logits derived from a hash of (prompt, label): same bytes in, same bytes out."""

    name = "fake"

    def __init__(self, checkpoint: str = "fake", scale: float = 3.0, position_bias: float = 0.0) -> None:
        self.checkpoint = checkpoint
        self.device = "cpu"
        self.scale = scale
        self.position_bias = position_bias
        self.tok = _WordTokenizer()
        self.table = build_label_table(self.tok, name="fake")

    def render(self, messages: Sequence[dict[str, str]]) -> str:
        return "\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages) + "\n<assistant>\n"

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))

    def _logit(self, prompt: str, label: str, index: int) -> float:
        # Hash the question part only (after the state) so that both orders of one question see the
        # same per-option logit, which is what a position-unbiased model would do.
        key = (prompt.split("### QUESTION", 1)[-1] + "\0" + label).encode()
        h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return self.scale * (h / 2**64 - 0.5) * 2 + self.position_bias * (1.0 if index == 0 else 0.0)

    def score(self, prompts: Sequence[str], labels: Sequence[Sequence[Label]]) -> list[np.ndarray]:
        out = []
        for prompt, labs in zip(prompts, labels):
            z = np.array([self._logit(prompt, lab.text, i) for i, lab in enumerate(labs)], dtype=np.float64)
            z = z - z.max()
            out.append(z - np.log(np.exp(z).sum()))
        return out


def load_backend(checkpoint: str, backend: str = "auto", **kwargs) -> Backend:
    """Build a backend. ``auto`` picks MLX on Apple Silicon when mlx-lm imports, otherwise torch."""
    if backend == "fake":
        return FakeBackend(checkpoint)
    if backend == "auto":
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
            backend = "mlx"
        except ImportError:
            backend = "torch"
    if backend == "mlx":
        from .mlx_backend import MlxBackend
        return MlxBackend(checkpoint, **kwargs)
    if backend == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(checkpoint, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")
