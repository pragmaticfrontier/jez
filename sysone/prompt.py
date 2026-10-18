"""Turn a (state, question) pair into *probes*.

A probe is one chat prompt plus the list of answer labels whose next-token
logits we read out. The model is never asked to write anything: the prompt
ends where the answer label would begin, and we only look at the logits of
the allowed labels at that position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import BoolQuestion, ChoiceQuestion, ScoreQuestion

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYSTEM_PROMPT = (
    "You are a decision engine, not a chat assistant. You are given a STATE (the situation) "
    "and one QUESTION with a fixed set of allowed answers. Read the state carefully, then answer "
    "the question by outputting only the label of the single best answer. Never explain, never "
    "add any other words."
)


@dataclass(frozen=True)
class Probe:
    messages: list[dict[str, str]]
    labels: list[str]    # surface strings whose first token is scored, e.g. ["A", "B", "C"]
    targets: list[str]   # answer key each label stands for, same order as `labels`


def format_state(state: Any) -> str:
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, ensure_ascii=False, indent=2)


def _messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def choice_probes(q: ChoiceQuestion, state_text: str, debias: bool = False) -> list[Probe]:
    keys = list(q.criteria.keys())
    n = len(keys)
    labels = list(LETTERS[:n])
    rotations = range(n) if debias else range(1)
    probes: list[Probe] = []
    for r in rotations:
        order = keys[r:] + keys[:r]
        lines = []
        for label, key in zip(labels, order):
            desc = q.criteria[key].strip()
            lines.append(f"{label}) {key}" + (f" — {desc}" if desc else ""))
        user = (
            f"### STATE\n{state_text}\n\n"
            f"### QUESTION\n{q.instructions.strip()}\n\n"
            f"### OPTIONS\n" + "\n".join(lines) + "\n\n"
            f"Reply with only the letter ({_join_labels(labels)}) of the single best option."
        )
        probes.append(Probe(messages=_messages(user), labels=labels, targets=order))
    return probes


def bool_probes(q: BoolQuestion, state_text: str) -> list[Probe]:
    user = (
        f"### STATE\n{state_text}\n\n"
        f"### QUESTION\n{q.instructions.strip()}\n\n"
        "Judge this question (or statement) using only the state above. "
        "Reply with only Yes or No."
    )
    return [Probe(messages=_messages(user), labels=["Yes", "No"], targets=["true", "false"])]


def score_probes(q: ScoreQuestion, state_text: str) -> list[Probe]:
    levels = q.levels
    use_digits = all(0 <= lv <= 9 for lv in levels)
    labels = [str(lv) for lv in levels] if use_digits else list(LETTERS[: len(levels)])
    rubric = q.criteria or {}
    lines = []
    for label, lv in zip(labels, levels):
        anchor = rubric.get(str(lv), "").strip()
        head = f"{lv}" if use_digits else f"{label}) {lv}"
        lines.append(head + (f" — {anchor}" if anchor else ""))
    what = "number" if use_digits else "letter"
    user = (
        f"### STATE\n{state_text}\n\n"
        f"### QUESTION\n{q.instructions.strip()}\n\n"
        f"### SCALE ({q.min} = lowest, {q.max} = highest)\n" + "\n".join(lines) + "\n\n"
        f"Reply with only the single {what} ({_join_labels(labels)}) that fits best."
    )
    return [Probe(messages=_messages(user), labels=labels, targets=[str(lv) for lv in levels])]


def build_probes(q: ChoiceQuestion | BoolQuestion | ScoreQuestion, state_text: str, debias: bool = False) -> list[Probe]:
    if isinstance(q, ChoiceQuestion):
        return choice_probes(q, state_text, debias=debias)
    if isinstance(q, BoolQuestion):
        return bool_probes(q, state_text)
    if isinstance(q, ScoreQuestion):
        return score_probes(q, state_text)
    raise TypeError(f"unsupported question type: {type(q).__name__}")
