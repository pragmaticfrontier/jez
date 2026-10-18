"""Answer labels and their token ids, per tokenizer.

Choice options are labelled A..Z, then AA, AB, ... up to 255 labels. Score
levels are labelled with the digits 0..9. Noul uses Yes and No. A label is
*single-token* when the tokenizer encodes it as exactly one token; the engine
then reads that token's logit at the answer position. The table also accepts
the leading-space surface form when it is a single token, and the two ids are
combined by log-sum-exp.

If a tokenizer cannot supply 255 single-token labels, the remaining labels are
*multi-token* and are scored by teacher forcing: the engine feeds the label's
tokens and sums their log-probabilities. Both Qwen3.5 and Gemma 4 tokenizers
supply more than 255 single-token labels, so that path is a fallback.
"""
from __future__ import annotations

import itertools
import string
from dataclasses import dataclass, field
from typing import Protocol

MAX_CHOICE = 255
MAX_SCORE = 10
LETTERS = string.ascii_uppercase
DIGITS = "0123456789"
NOUL_LABELS = ("Yes", "No")


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def candidate_labels(n: int = MAX_CHOICE) -> list[str]:
    """A..Z, then AA, AB, ..., in a fixed order; the first ``n`` are the labels used."""
    out = list(LETTERS)
    for a, b in itertools.product(LETTERS, repeat=2):
        out.append(a + b)
        if len(out) >= n:
            break
    return out[:n]


@dataclass(frozen=True)
class Label:
    text: str
    ids: tuple[int, ...]  # single-token: one id per surface form; multi-token: the full sequence
    multi_token: bool = False


@dataclass
class LabelTable:
    choice: list[Label] = field(default_factory=list)  # 255 entries
    score: list[Label] = field(default_factory=list)  # 10 entries, digits 0..9
    noul: list[Label] = field(default_factory=list)  # Yes, No
    tokenizer_name: str = ""

    @property
    def n_single_token(self) -> int:
        return sum(not lab.multi_token for lab in self.choice)

    def choice_labels(self, n: int) -> list[Label]:
        if not 1 <= n <= MAX_CHOICE:
            raise ValueError(f"choice questions take 1 to {MAX_CHOICE} options, got {n}")
        return self.choice[:n]

    def score_labels(self, n: int) -> list[Label]:
        if not 1 <= n <= MAX_SCORE:
            raise ValueError(f"score questions take 1 to {MAX_SCORE} levels, got {n}")
        return self.score[:n]


def single_token_ids(tok: Tokenizer, text: str, with_space: bool = True) -> tuple[int, ...]:
    """Token ids for the surface forms of ``text`` that are single tokens (plain first)."""
    ids: list[int] = []
    forms = [text, " " + text] if with_space else [text]
    for form in forms:
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    return tuple(ids)


def make_label(tok: Tokenizer, text: str, with_space: bool = True) -> Label:
    plain = tok.encode(text, add_special_tokens=False)
    if len(plain) == 1:
        return Label(text=text, ids=single_token_ids(tok, text, with_space=with_space))
    return Label(text=text, ids=tuple(plain), multi_token=True)


def build_label_table(tok: Tokenizer, name: str = "") -> LabelTable:
    """Build the table for one tokenizer. Multi-token labels are kept, in order, after the single-token ones run out."""
    singles: list[Label] = []
    for text in candidate_labels(26 + 26 * 26):
        lab = make_label(tok, text)
        if not lab.multi_token:
            singles.append(lab)
        if len(singles) >= MAX_CHOICE:
            break
    choice = singles[:MAX_CHOICE]
    if len(choice) < MAX_CHOICE:
        # Teacher-forced fallback: the next candidates in order, as token sequences.
        used = {lab.text for lab in choice}
        for text in candidate_labels(26 + 26 * 26):
            if text in used:
                continue
            choice.append(Label(text=text, ids=tuple(tok.encode(text, add_special_tokens=False)), multi_token=True))
            if len(choice) >= MAX_CHOICE:
                break
    score = [make_label(tok, d) for d in DIGITS]
    noul = [make_label(tok, w) for w in NOUL_LABELS]
    return LabelTable(choice=choice, score=score, noul=noul, tokenizer_name=name)


def check_label_table(table: LabelTable) -> list[str]:
    """Problems that would make scoring inexact; empty when the table is fully single-token."""
    problems = []
    if len(table.choice) < MAX_CHOICE:
        problems.append(f"only {len(table.choice)} choice labels")
    multi = [lab.text for lab in table.choice if lab.multi_token]
    if multi:
        problems.append(f"{len(multi)} choice labels need teacher forcing (first: {multi[:3]})")
    for lab in table.score:
        if lab.multi_token:
            problems.append(f"digit {lab.text!r} is not a single token")
    for lab in table.noul:
        if lab.multi_token:
            problems.append(f"noul label {lab.text!r} is not a single token")
    texts = [lab.text for lab in table.choice]
    if len(set(texts)) != len(texts):
        problems.append("duplicate choice labels")
    return problems
