"""The prompt planner: one state block, one suffix per question and option order.

Every probe is a chat conversation (system + user) that ends exactly where the
answer label would start. The engine renders it with the model's chat template,
adds the assistant turn, and reads the logits of the allowed labels at that
position.

Each question is planned twice: in the caller's option order and in reverse.
The engine averages the two sets of label log-probabilities, which removes most
position bias. Noul questions swap Yes and No the same way.

Instructions and criteria that are objects or arrays reach the model as
indented JSON. Backticked references such as `items[3]` pass through unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .labels import DIGITS, NOUL_LABELS, LabelTable, candidate_labels
from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneRequest

SYSTEM_PROMPT = (
    "You are a decision engine, not a chat assistant. You are given a STATE (the situation to evaluate) "
    "followed by one QUESTION with a fixed set of allowed answers. Read the state carefully, then answer the "
    "question by outputting only the label of the single best answer. Never explain and never add any other "
    "words. Backticked references such as `items[3]` point at parts of the state."
)

_COMPACT_LIMIT = 100  # longer non-string descriptions are rendered as an indented block


@dataclass(frozen=True)
class Probe:
    question_id: str
    qtype: str  # noul | choice | score
    order: str  # given | reversed
    messages: tuple[dict[str, str], ...]
    labels: tuple[str, ...]  # label texts in presentation order, e.g. ("A", "B", "C")
    targets: tuple[str, ...]  # what each label stands for: option name, level index, "true"/"false"


def render_content(value: Any, indent: int = 2) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _describe(desc: Any, prefix_len: int) -> str:
    """Render an option or level description after its label; non-strings become JSON."""
    if desc is None:
        return ""
    if isinstance(desc, str):
        text = desc.strip()
        return f" — {text}" if text else ""
    compact = json.dumps(desc, ensure_ascii=False, separators=(", ", ": "))
    if len(compact) <= _COMPACT_LIMIT:
        return f" — {compact}"
    block = json.dumps(desc, ensure_ascii=False, indent=2)
    pad = " " * 4
    return " —\n" + "\n".join(pad + line for line in block.splitlines())


def state_block(state: Any) -> str:
    return f"### STATE\n{render_content(state)}"


def _instructions_text(q: Question) -> str:
    if q.instructions is None:
        if isinstance(q, NoulQuestion):
            return "Does the state match the description of Yes rather than the description of No?"
        if isinstance(q, ChoiceQuestion):
            return "Which option describes the state best?"
        return "Which level describes the state best?"
    return render_content(q.instructions)


def _label_range(labels: list[str]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    if len(labels) <= 6:
        return ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"{labels[0]} to {labels[-1]}"


def choice_block(q: ChoiceQuestion, labels: list[str], reverse: bool) -> tuple[str, list[str], list[str]]:
    options = list(q.options)
    if reverse:
        options.reverse()
    lines = []
    for label, (name, desc) in zip(labels, options):
        lines.append(f"{label}) {name}{_describe(desc, len(label) + 2)}")
    text = (
        f"### QUESTION\n{_instructions_text(q)}\n\n"
        f"### OPTIONS\n" + "\n".join(lines) + "\n\n"
        f"Reply with only the label ({_label_range(labels)}) of the single best option."
    )
    return text, labels, [name for name, _ in options]


def noul_block(q: NoulQuestion, reverse: bool) -> tuple[str, list[str], list[str]]:
    yes, no = NOUL_LABELS
    pairs = [(yes, "true", q.true), (no, "false", q.false)]
    if reverse:
        pairs.reverse()
    crit_lines = [f"{label}{_describe(desc, len(label))}" for label, _, desc in pairs]
    labels = [label for label, _, _ in pairs]
    text = (
        f"### QUESTION\n{_instructions_text(q)}\n\n"
        f"### ANSWERS\n" + "\n".join(crit_lines) + "\n\n"
        f"Judge the question using only the state above. Reply with only {labels[0]} or {labels[1]}."
    )
    return text, labels, [target for _, target, _ in pairs]


def score_block(q: ScoreQuestion, reverse: bool) -> tuple[str, list[str], list[str]]:
    n = len(q.levels)
    idx = list(range(n))
    if reverse:
        idx.reverse()
    lines = [f"{i}{_describe(q.levels[i], 1)}" for i in idx]
    labels = [DIGITS[i] for i in idx]
    text = (
        f"### QUESTION\n{_instructions_text(q)}\n\n"
        f"### LEVELS (0 = lowest, {n - 1} = highest)\n" + "\n".join(lines) + "\n\n"
        f"Reply with only the single digit ({labels[0]} to {labels[-1]}) of the level that fits best."
    )
    return text, labels, [str(i) for i in idx]


def question_block(q: Question, table: LabelTable | None = None, reverse: bool = False) -> tuple[str, list[str], list[str]]:
    if isinstance(q, ChoiceQuestion):
        labels = [lab.text for lab in table.choice_labels(q.n)] if table else candidate_labels(q.n)
        return choice_block(q, labels, reverse)
    if isinstance(q, NoulQuestion):
        return noul_block(q, reverse)
    if isinstance(q, ScoreQuestion):
        return score_block(q, reverse)
    raise TypeError(type(q).__name__)


def messages_for(state_text: str, question_text: str) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{state_text}\n\n{question_text}"},
    )


def plan(req: SystemOneRequest, table: LabelTable | None = None, debias: bool = True) -> list[Probe]:
    """Probes for a request: per question, the given order and (with debias) the reversed order."""
    state_text = state_block(req.state)
    probes: list[Probe] = []
    for q in req.questions:
        for order in (("given", "reversed") if debias else ("given",)):
            text, labels, targets = question_block(q, table, reverse=order == "reversed")
            probes.append(Probe(
                question_id=q.id, qtype=q.type, order=order,
                messages=messages_for(state_text, text), labels=tuple(labels), targets=tuple(targets),
            ))
    return probes


def plan_single(q: Question, state: Any, table: LabelTable | None, reverse: bool = False) -> Probe:
    """One probe for training and evaluation code that controls the order itself."""
    text, labels, targets = question_block(q, table, reverse=reverse)
    return Probe(question_id=q.id, qtype=q.type, order="reversed" if reverse else "given",
                 messages=messages_for(state_block(state), text), labels=tuple(labels), targets=tuple(targets))
