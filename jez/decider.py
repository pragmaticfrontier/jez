"""Decider: a validated request -> probes -> label log-probabilities -> Jev answers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import models as registry
from .calibration import Calibration
from .confidence import choice_confidence, score_confidence, score_value
from .engine import Backend
from .prompt import Probe, plan, question_block, state_block
from .schema import (
    ChoiceQuestion,
    JezError,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    choice_answer,
    noul_answer,
    score_answer,
)

ROUND = 6


def _rounded_distribution(p: np.ndarray) -> list[float]:
    """Round to ROUND decimals and put the residual on the largest entry so the values sum to 1."""
    q = [round(float(x), ROUND) for x in p]
    residual = round(1.0 - sum(q), ROUND)
    if residual:
        i = int(np.argmax(p))
        q[i] = round(q[i] + residual, ROUND)
    return q


def average_orders(probes: list[Probe], logprobs: list[np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Average the restricted log-probabilities of one question across its probes, keyed by target."""
    targets = list(probes[0].targets)
    acc = {t: [] for t in targets}
    for probe, lp in zip(probes, logprobs):
        for t, v in zip(probe.targets, lp):
            acc[t].append(float(v))
    return targets, np.array([np.mean(acc[t]) for t in targets], dtype=np.float64)


def _softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    z = z / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Decision:
    model_id: str
    answers: dict[str, dict[str, Any]]
    input_tokens: int
    latency_ms: float
    probes: int

    def body(self) -> dict[str, Any]:
        return {"model": self.model_id, "answers": self.answers,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}}


class Decider:
    def __init__(self, backend: Backend, model_id: str, calibration: Calibration | None = None,
                 debias: bool = True) -> None:
        self.backend = backend
        self.model_id = model_id
        self.calibration = calibration or Calibration()
        self.debias = debias

    # ------------------------------------------------------------------ limits
    def count(self, req: SystemOneRequest) -> tuple[int, list[int]]:
        n_state = self.backend.count_tokens(state_block(req.state))
        n_q = [self.backend.count_tokens(question_block(q, self.backend.table)[0]) for q in req.questions]
        return n_state, n_q

    def check_limits(self, req: SystemOneRequest) -> int:
        n_state, n_q = self.count(req)
        for q, n in zip(req.questions, n_q):
            if n_state + n > registry.PER_QUESTION_CONTEXT:
                raise JezError(400, "max_tokens_exceeded",
                               f"state plus question {q.id!r} is {n_state + n} tokens; the per-question limit is "
                               f"{registry.PER_QUESTION_CONTEXT}", field=f"questions.{q.id}")
        total = n_state + sum(n_q)
        if total > registry.PER_REQUEST_CONTEXT:
            raise JezError(400, "max_tokens_exceeded",
                           f"state plus all questions is {total} tokens; the per-request limit is "
                           f"{registry.PER_REQUEST_CONTEXT}", field="questions")
        return total

    # ------------------------------------------------------------------ answers
    def score_requests(self, reqs: list[SystemOneRequest]) -> list[dict[str, tuple[list[str], np.ndarray]]]:
        """Averaged (over orders) restricted log-probabilities per question, before temperature, for many requests.

        Every probe of every request goes to the backend in one call, so evaluation runs at full batch width.
        """
        all_probes: list[tuple[int, Probe]] = []
        for i, req in enumerate(reqs):
            all_probes.extend((i, p) for p in plan(req, self.backend.table, debias=self.debias))
        prompts = [self.backend.render(p.messages) for _, p in all_probes]
        labels = [self._labels_for(p) for _, p in all_probes]
        logprobs = self.backend.score(prompts, labels)
        out: list[dict[str, tuple[list[str], np.ndarray]]] = [{} for _ in reqs]
        by_key: dict[tuple[int, str], tuple[list[Probe], list[np.ndarray]]] = {}
        for (i, p), lp in zip(all_probes, logprobs):
            by_key.setdefault((i, p.question_id), ([], []))
            by_key[(i, p.question_id)][0].append(p)
            by_key[(i, p.question_id)][1].append(lp)
        for (i, qid), (ps, lps) in by_key.items():
            out[i][qid] = average_orders(ps, lps)
        return out

    def decide(self, req: SystemOneRequest) -> Decision:
        t0 = time.perf_counter()
        input_tokens = self.check_limits(req)
        scored = self.score_requests([req])[0]
        answers: dict[str, dict[str, Any]] = {}
        for q in req.questions:
            targets, avg = scored[q.id]
            T = self.calibration.temperature_for(q.type, q.n)
            answers[q.id] = self._answer(q, targets, _softmax(avg, T))
        return Decision(model_id=self.model_id, answers=answers, input_tokens=input_tokens,
                        latency_ms=(time.perf_counter() - t0) * 1000, probes=2 * len(req.questions) if self.debias else len(req.questions))

    def _labels_for(self, probe: Probe):
        table = self.backend.table
        if probe.qtype == "choice":
            by_text = {lab.text: lab for lab in table.choice}
        elif probe.qtype == "score":
            by_text = {lab.text: lab for lab in table.score}
        else:
            by_text = {lab.text: lab for lab in table.noul}
        return [by_text[t] for t in probe.labels]

    @staticmethod
    def _answer(q, targets: list[str], p: np.ndarray) -> dict[str, Any]:
        if isinstance(q, ChoiceQuestion):
            order = {name: i for i, name in enumerate(targets)}
            probs = np.array([p[order[name]] for name in q.names])
            return choice_answer(q.names, _rounded_distribution(probs), round(choice_confidence(probs.tolist()), 4))
        if isinstance(q, NoulQuestion):
            p_true = float(p[targets.index("true")])
            return noul_answer(round(p_true, 4))
        if isinstance(q, ScoreQuestion):
            order = {t: i for i, t in enumerate(targets)}
            probs = np.array([p[order[str(i)]] for i in range(q.n)])
            return score_answer(q.levels, _rounded_distribution(probs), round(score_value(probs.tolist()), 4),
                                round(score_confidence(probs.tolist()), 4))
        raise TypeError(type(q).__name__)
