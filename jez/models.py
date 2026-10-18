"""The Jez model registry: versioned ids, aliases, base checkpoints.

A server serves one or more entries of this registry. Requests name a model
by its versioned id or an alias; the response echoes the versioned id that
answered. Jev's own names (``jev-latest``, ``jev-1.13.0``, ...) are rejected
with 422 so that a misconfigured client fails loudly instead of believing it
got an answer from Jev.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str  # "small" | "large"
    base: str  # Hugging Face id of the base checkpoint
    arch: str  # "qwen3_5" | "gemma4" | "auto"
    description: str
    release_date: str
    native_context: int
    cost_per_mtok_usd: float  # reported in the Jez-Cost-Usd header; informational
    weights: str | None = None  # Hugging Face id of the released fine-tuned weights; None until a release is published


MODELS: dict[str, ModelSpec] = {
    "jez-small-1.0.0": ModelSpec(
        id="jez-small-1.0.0",
        family="small",
        base="Qwen/Qwen3.5-0.8B",
        arch="qwen3_5",
        description="Jez small: Qwen3.5-0.8B text decoder fine-tuned on label logits. Fast and small.",
        release_date="2026-10-16",
        native_context=262_144,
        cost_per_mtok_usd=0.004,
    ),
    "jez-large-1.0.0": ModelSpec(
        id="jez-large-1.0.0",
        family="large",
        base="google/gemma-4-E4B-it",
        arch="gemma4",
        description="Jez large: Gemma 4 E4B text decoder fine-tuned on label logits. Slower and more accurate.",
        release_date="2026-10-16",
        native_context=131_072,
        cost_per_mtok_usd=0.02,
    ),
}

ALIASES: dict[str, str] = {
    "jez-small-latest": "jez-small-1.0.0",
    "jez-large-latest": "jez-large-1.0.0",
    "jez-latest": "jez-large-1.0.0",
}

# Request limits shared by every Jez model (TypeSafe's defaults for Jev).
PER_REQUEST_CONTEXT = 64_000
PER_QUESTION_CONTEXT = 32_000

_JEV_NAME = re.compile(r"^jev(-|$)", re.IGNORECASE)


def is_jev_name(name: str) -> bool:
    return bool(_JEV_NAME.match(name.strip()))


def resolve(name: str) -> str | None:
    """Return the versioned id for a model id or alias, or None when unknown."""
    key = name.strip()
    if key in MODELS:
        return key
    return ALIASES.get(key)


def aliases_for(model_id: str) -> list[str]:
    return sorted(a for a, target in ALIASES.items() if target == model_id)


def family_of(model_id: str) -> str:
    return MODELS[model_id].family


def parse_model_env(value: str | None) -> list[tuple[str, str]]:
    """Parse ``JEZ_MODELS``: a comma-separated list of ``id`` or ``id=checkpoint`` entries.

    ``jez-small-1.0.0`` serves the released weights (``ModelSpec.weights``) when the
    registry names them, else the base checkpoint; ``JEZ_CHECKPOINT_SMALL`` /
    ``JEZ_CHECKPOINT_LARGE`` override that for a family, and ``jez-small-1.0.0=<path or
    Hub id>`` puts a stand-in behind one id, which is how milestone 0 ran before the fine-tune.
    """
    entries: list[tuple[str, str]] = []
    for raw in (value or "jez-small-1.0.0").split(","):
        raw = raw.strip()
        if not raw:
            continue
        model_id, _, checkpoint = raw.partition("=")
        resolved = resolve(model_id)
        if resolved is None:
            raise ValueError(f"JEZ_MODELS: unknown model id {model_id!r}")
        spec = MODELS[resolved]
        ckpt = checkpoint.strip() or os.environ.get(f"JEZ_CHECKPOINT_{spec.family.upper()}", "").strip() or spec.weights or spec.base
        entries.append((resolved, ckpt))
    return entries
