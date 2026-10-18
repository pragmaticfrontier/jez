"""Jev's wire format: request parsing, validation and the error shape.

Every rule here comes from TypeSafe's API reference (https://docs.typesafe.ai/api)
and the shapes its SDKs send. Validation is written by hand rather than through
a pydantic model so that every error can name the exact field, as Jev does.

Status codes:

* 400  the body breaks a request rule: not UTF-8, a byte order mark, non-finite
       numbers, more than 64 nested levels, or too many tokens (``max_tokens_exceeded``)
* 422  a missing or malformed field, an unknown model, an image asset, a ``box`` question
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Union

from . import models as registry

Content = Union[str, dict, list]

MAX_DEPTH = 64
MIN_CHOICE, MAX_CHOICE = 2, 255
MIN_SCORE, MAX_SCORE = 2, 10
QUESTION_TYPES = ("noul", "choice", "score")


class JezError(Exception):
    """An HTTP error with Jev's body shape: {"error": {"type", "message", "field"}}."""

    def __init__(self, status: int, type: str, message: str, field: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.type = type
        self.message = message
        self.field = field
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        err: dict[str, Any] = {"type": self.type, "message": self.message}
        if self.field is not None:
            err["field"] = self.field
        return {"error": err}


def invalid(message: str, field: str | None = None, type: str = "invalid_request") -> JezError:
    return JezError(422, type, message, field)


def bad_request(message: str, field: str | None = None, type: str = "invalid_request") -> JezError:
    return JezError(400, type, message, field)


# ----------------------------------------------------------------------------- parsed request

@dataclass
class NoulQuestion:
    id: str
    instructions: Content | None
    true: Content | None = None
    false: Content | None = None
    type: str = field(default="noul", init=False)

    @property
    def n(self) -> int:
        return 2


@dataclass
class ChoiceQuestion:
    id: str
    instructions: Content | None
    options: list[tuple[str, Content | None]]  # (name, description) in the caller's order
    type: str = field(default="choice", init=False)

    @property
    def n(self) -> int:
        return len(self.options)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.options]


@dataclass
class ScoreQuestion:
    id: str
    instructions: Content | None
    levels: list[Content]  # lowest first
    type: str = field(default="score", init=False)

    @property
    def n(self) -> int:
        return len(self.levels)


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


@dataclass
class SystemOneRequest:
    model: str  # as given by the caller (id or alias)
    model_id: str  # resolved versioned id
    state: Content
    questions: list[Question]

    def question(self, qid: str) -> Question:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)


# ----------------------------------------------------------------------------- JSON parsing

def _reject_constant(name: str) -> Any:
    raise bad_request(f"JSON number {name} is not finite", type="invalid_json")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise bad_request(f"JSON number {text} is not finite", type="invalid_json")
    return value


def parse_json_body(raw: bytes) -> Any:
    """Decode a request body: UTF-8 without a byte order mark, finite numbers only."""
    if raw[:3] == b"\xef\xbb\xbf":
        raise bad_request("the body starts with a UTF-8 byte order mark", type="invalid_json")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        raise bad_request("the body must be UTF-8 JSON", type="invalid_json")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise bad_request(f"the body is not valid UTF-8 at byte {e.start}", type="invalid_json")
    try:
        return json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float)
    except RecursionError:
        raise bad_request(f"JSON nesting exceeds {MAX_DEPTH} levels", type="nesting_too_deep")
    except json.JSONDecodeError as e:
        raise bad_request(f"the body is not valid JSON: {e.msg} at line {e.lineno} column {e.colno}", type="invalid_json")


def depth(value: Any) -> int:
    """Nesting depth of a JSON value: scalars are 0, an empty container is 1."""
    best = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, dict):
            best = max(best, d + 1)
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, list):
            best = max(best, d + 1)
            stack.extend((x, d + 1) for x in v)
    return best


def _check_depth(value: Any, path: str) -> None:
    if depth(value) > MAX_DEPTH:
        raise bad_request(f"{path} is nested more than {MAX_DEPTH} levels deep", field=path, type="nesting_too_deep")


# ----------------------------------------------------------------------------- validation

def _is_content(v: Any) -> bool:
    return isinstance(v, (str, dict, list))


def _content_or_none(v: Any, path: str) -> Content | None:
    if v is None:
        return None
    if not _is_content(v):
        raise invalid(f"{path} must be a string, object or array", field=path)
    _check_depth(v, path)
    return v


def _instructions(q: dict, path: str, required: bool) -> Content | None:
    v = q.get("instructions")
    if v is None or v == "":
        if required:
            raise invalid(f"{path}.instructions is required", field=f"{path}.instructions", type="missing_field")
        return None
    return _content_or_none(v, f"{path}.instructions")


def _validate_noul(qid: str, q: dict, path: str) -> NoulQuestion:
    crit = q.get("criteria")
    true = false = None
    if crit is not None:
        if not isinstance(crit, dict):
            raise invalid(f"{path}.criteria must be an object with 'true' and 'false'", field=f"{path}.criteria")
        _check_depth(crit, f"{path}.criteria")
        extra = set(crit) - {"true", "false"}
        if extra:
            raise invalid(f"{path}.criteria has unknown keys {sorted(extra)}", field=f"{path}.criteria")
        true = _content_or_none(crit.get("true"), f"{path}.criteria.true")
        false = _content_or_none(crit.get("false"), f"{path}.criteria.false")
    described = true is not None or false is not None
    instructions = _instructions(q, path, required=not described)
    return NoulQuestion(id=qid, instructions=instructions, true=true, false=false)


def _validate_choice(qid: str, q: dict, path: str) -> ChoiceQuestion:
    if "criteria" not in q:
        raise invalid(f"{path}.criteria is required", field=f"{path}.criteria", type="missing_field")
    crit = q["criteria"]
    if not isinstance(crit, dict):
        raise invalid(f"{path}.criteria must be an object mapping option names to descriptions", field=f"{path}.criteria")
    _check_depth(crit, f"{path}.criteria")
    if not MIN_CHOICE <= len(crit) <= MAX_CHOICE:
        raise invalid(f"{path}.criteria must have {MIN_CHOICE} to {MAX_CHOICE} options, got {len(crit)}", field=f"{path}.criteria")
    options: list[tuple[str, Content | None]] = []
    for name, desc in crit.items():
        if not isinstance(name, str) or not name.strip():
            raise invalid(f"{path}.criteria has an empty option name", field=f"{path}.criteria")
        options.append((name, _content_or_none(desc, f"{path}.criteria.{name}")))
    instructions = _instructions(q, path, required=False)
    return ChoiceQuestion(id=qid, instructions=instructions, options=options)


def _validate_score(qid: str, q: dict, path: str) -> ScoreQuestion:
    if "criteria" not in q:
        raise invalid(f"{path}.criteria is required", field=f"{path}.criteria", type="missing_field")
    crit = q["criteria"]
    if not isinstance(crit, list):
        raise invalid(f"{path}.criteria must be an ordered array of level descriptions, lowest first", field=f"{path}.criteria")
    _check_depth(crit, f"{path}.criteria")
    if not MIN_SCORE <= len(crit) <= MAX_SCORE:
        raise invalid(f"{path}.criteria must have {MIN_SCORE} to {MAX_SCORE} levels, got {len(crit)}", field=f"{path}.criteria")
    levels: list[Content] = []
    for i, lv in enumerate(crit):
        if lv is None or not _is_content(lv):
            raise invalid(f"{path}.criteria[{i}] must be a string, object or array", field=f"{path}.criteria[{i}]")
        _check_depth(lv, f"{path}.criteria[{i}]")
        levels.append(lv)
    instructions = _instructions(q, path, required=False)
    return ScoreQuestion(id=qid, instructions=instructions, levels=levels)


def validate_question(qid: str, q: Any) -> Question:
    path = f"questions.{qid}"
    if not isinstance(q, dict):
        raise invalid(f"{path} must be an object", field=path)
    qtype = q.get("type")
    if qtype is None:
        raise invalid(f"{path}.type is required", field=f"{path}.type", type="missing_field")
    if qtype == "box":
        raise invalid(f"{path}.type 'box' is not supported: Jez answers text-only questions", field=f"{path}.type",
                      type="unsupported_question")
    if qtype not in QUESTION_TYPES:
        raise invalid(f"{path}.type must be one of {list(QUESTION_TYPES)}, got {qtype!r}", field=f"{path}.type")
    if qtype == "noul":
        return _validate_noul(qid, q, path)
    if qtype == "choice":
        return _validate_choice(qid, q, path)
    return _validate_score(qid, q, path)


def validate_model(name: Any) -> tuple[str, str]:
    if name is None:
        raise invalid("model is required", field="model", type="missing_field")
    if not isinstance(name, str) or not name.strip():
        raise invalid("model must be a non-empty string", field="model")
    resolved = registry.resolve(name)
    if resolved is None:
        if registry.is_jev_name(name):
            raise invalid(
                f"model {name!r} is a Jev model name; Jez serves {sorted(registry.MODELS)} and the aliases "
                f"{sorted(registry.ALIASES)}", field="model", type="unknown_model")
        raise invalid(f"unknown model {name!r}", field="model", type="unknown_model")
    return name, resolved


def validate_request(data: Any) -> SystemOneRequest:
    """Validate a decoded JSON body against Jev's request schema."""
    if not isinstance(data, dict):
        raise invalid("the request body must be a JSON object", field="")
    if "assets" in data and data["assets"] is not None:
        raise invalid("image assets are not supported: Jez answers text-only requests", field="assets",
                      type="unsupported_asset")
    if "state" not in data or data["state"] is None:
        raise invalid("state is required", field="state", type="missing_field")
    state = data["state"]
    if not _is_content(state):
        raise invalid("state must be a string, object or array", field="state")
    _check_depth(state, "state")
    model_name, model_id = validate_model(data.get("model"))
    questions = data.get("questions")
    if questions is None:
        raise invalid("questions is required", field="questions", type="missing_field")
    if not isinstance(questions, dict):
        raise invalid("questions must be an object mapping question ids to questions", field="questions")
    if not questions:
        raise invalid("questions must contain at least one question", field="questions")
    parsed = [validate_question(str(qid), q) for qid, q in questions.items()]
    return SystemOneRequest(model=model_name, model_id=model_id, state=state, questions=parsed)


# ----------------------------------------------------------------------------- answers

def choice_answer(names: list[str], probs: list[float], confidence: float) -> dict[str, Any]:
    return {"type": "choice", "choice": names[max(range(len(names)), key=lambda i: probs[i])],
            "probabilities": dict(zip(names, probs)), "confidence": confidence}


def noul_answer(p_true: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p_true}


def score_answer(levels: list[Content], probs: list[float], score: float, confidence: float) -> dict[str, Any]:
    legend = {str(i): (lv if isinstance(lv, str) else json.dumps(lv, ensure_ascii=False, separators=(",", ":")))
              for i, lv in enumerate(levels)}
    return {"type": "score", "score": score, "legend": legend,
            "probabilities": {str(i): p for i, p in enumerate(probs)}, "confidence": confidence}
