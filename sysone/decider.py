"""Decider: request -> probes -> one batched scoring pass -> typed answers."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .calibration import Calibration, softmax
from .engine import Scorer
from .prompt import Probe, build_probes, format_state
from .schema import (
    BoolAnswer,
    BoolQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

Answer = ChoiceAnswer | BoolAnswer | ScoreAnswer


def _r(x: float) -> float:
    return round(float(x), 4)


def aggregate(question, probes: list[Probe], logits: list[np.ndarray], temperature: float,
              return_logits: bool = False) -> Answer:
    """Combine the probes of one question into an answer.

    Probabilities are softmax(logits / T) per probe, then averaged per answer
    key across probes (several probes only exist when `debias` is on).
    """
    prob_acc: dict[str, float] = defaultdict(float)
    logit_acc: dict[str, float] = defaultdict(float)
    for probe, z in zip(probes, logits):
        p = softmax(np.asarray(z, dtype=np.float64), temperature)
        for key, pi, zi in zip(probe.targets, p, z):
            prob_acc[key] += float(pi) / len(probes)
            logit_acc[key] += float(zi) / len(probes)

    if isinstance(question, ChoiceQuestion):
        keys = list(question.criteria.keys())
        probs = {k: prob_acc[k] for k in keys}
        best = max(keys, key=probs.__getitem__)
        return ChoiceAnswer(
            choice=best,
            probabilities={k: _r(v) for k, v in probs.items()},
            confidence=_r(probs[best]),
            logits={k: _r(logit_acc[k]) for k in keys} if return_logits else None,
        )

    if isinstance(question, BoolQuestion):
        p_true = prob_acc["true"]
        return BoolAnswer(
            value=bool(p_true >= 0.5),
            probability=_r(p_true),
            confidence=_r(max(p_true, 1.0 - p_true)),
            logits={"true": _r(logit_acc["true"]), "false": _r(logit_acc["false"])} if return_logits else None,
        )

    if isinstance(question, ScoreQuestion):
        levels = question.levels
        probs = {str(lv): prob_acc[str(lv)] for lv in levels}
        best = max(levels, key=lambda lv: probs[str(lv)])
        expected = sum(lv * probs[str(lv)] for lv in levels)
        return ScoreAnswer(
            score=int(best),
            expected=_r(expected),
            probabilities={k: _r(v) for k, v in probs.items()},
            confidence=_r(probs[str(best)]),
            logits={str(lv): _r(logit_acc[str(lv)]) for lv in levels} if return_logits else None,
        )

    raise TypeError(f"unsupported question type: {type(question).__name__}")


class Decider:
    def __init__(self, scorer: Scorer, calibration: Calibration | None = None, max_state_tokens: int = 3000) -> None:
        self.scorer = scorer
        self.calibration = calibration or Calibration()
        self.max_state_tokens = max_state_tokens

    def decide(self, req: DecideRequest) -> DecideResponse:
        t0 = time.perf_counter()
        opts = req.options
        state_text = self.scorer.truncate_text(format_state(req.state), opts.max_state_tokens or self.max_state_tokens)

        plan: list[tuple[str, object, list[Probe]]] = []
        prompts: list[str] = []
        label_sets: list[list[str]] = []
        for name, q in req.questions.items():
            probes = build_probes(q, state_text, debias=opts.debias)
            plan.append((name, q, probes))
            for p in probes:
                prompts.append(self.scorer.render(p.messages))
                label_sets.append(p.labels)

        logits, n_tokens, passes = self.scorer.score(prompts, label_sets)

        answers: dict[str, Answer] = {}
        i = 0
        for name, q, probes in plan:
            q_logits = logits[i : i + len(probes)]
            i += len(probes)
            temperature = opts.temperature or self.calibration.temperature_for(q.type)
            answers[name] = aggregate(q, probes, q_logits, temperature, return_logits=opts.return_logits)

        return DecideResponse(
            answers=answers,
            model=self.scorer.model_id,
            usage=Usage(input_tokens=n_tokens, output_tokens=0, probes=len(prompts), forward_passes=passes),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
