"""Request / response schema for the decision API.

Three question primitives (same shape as the Jev API):

  choice  pick one option from a fixed set         -> distribution over the options
  bool    is a statement true for this state?       -> P(true)
  score   position on an integer rubric (min..max)  -> distribution over levels + expected value
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_OPTIONS = 26  # one letter label per option

# Accepted spellings for the question type field.
_TYPE_ALIASES = {
    "choice": "choice", "select": "choice", "enum": "choice", "classify": "choice",
    "bool": "bool", "boolean": "bool", "noul": "bool", "yesno": "bool", "check": "bool",
    "score": "score", "rating": "score", "scale": "score", "level": "score",
}


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(..., min_length=1, description="What is being decided.")
    criteria: dict[str, str] = Field(
        ..., description="option -> short description of when that option applies ('' allowed)."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_options_shorthand(cls, data: Any) -> Any:
        """Allow {"options": ["a", "b"]} as a shorthand for criteria without descriptions."""
        if isinstance(data, dict) and "criteria" not in data and "options" in data:
            data = dict(data)
            opts = data.pop("options")
            if isinstance(opts, dict):
                data["criteria"] = {str(k): str(v) for k, v in opts.items()}
            else:
                data["criteria"] = {str(o): "" for o in opts}
        return data

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) < 2:
            raise ValueError("a choice question needs at least 2 options")
        if len(v) > MAX_OPTIONS:
            raise ValueError(f"a choice question supports at most {MAX_OPTIONS} options")
        if any(not str(k).strip() for k in v):
            raise ValueError("option names must be non-empty")
        return {str(k): ("" if d is None else str(d)) for k, d in v.items()}


class BoolQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["bool"] = "bool"
    instructions: str = Field(..., min_length=1, description="A yes/no question or a statement to verify.")


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str = Field(..., min_length=1, description="What is being rated.")
    min: int = 1
    max: int = 5
    criteria: dict[str, str] | None = Field(
        None, description="Optional rubric anchors: level (as string) -> description."
    )

    @model_validator(mode="after")
    def _check_range(self) -> "ScoreQuestion":
        if self.max <= self.min:
            raise ValueError("score: max must be greater than min")
        if self.max - self.min + 1 > MAX_OPTIONS:
            raise ValueError(f"score: at most {MAX_OPTIONS} levels are supported")
        if self.criteria:
            clean: dict[str, str] = {}
            for k, d in self.criteria.items():
                try:
                    level = int(k)
                except (TypeError, ValueError):
                    raise ValueError(f"score: rubric key {k!r} is not an integer level")
                if not (self.min <= level <= self.max):
                    raise ValueError(f"score: rubric level {level} is outside [{self.min}, {self.max}]")
                clean[str(level)] = "" if d is None else str(d)
            self.criteria = clean
        return self

    @property
    def levels(self) -> list[int]:
        return list(range(self.min, self.max + 1))


Question = Annotated[Union[ChoiceQuestion, BoolQuestion, ScoreQuestion], Field(discriminator="type")]


class DecideOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debias: bool = Field(
        False,
        description="For choice questions, score every cyclic rotation of the option order and "
        "average, removing position bias. Costs N probes instead of 1 (still one batched pass).",
    )
    temperature: float | None = Field(
        None, gt=0, description="Override the calibrated softmax temperature."
    )
    max_state_tokens: int | None = Field(None, ge=16, description="Truncate the state to this many tokens.")
    return_logits: bool = Field(False, description="Include raw per-answer logits in each answer.")


def _normalize_question(name: str, q: Any) -> Any:
    if not isinstance(q, dict):
        return q
    q = dict(q)
    raw_type = q.get("type")
    if raw_type is None:
        if "criteria" in q or "options" in q:
            # score rubric also uses `criteria`, but only with min/max present
            q["type"] = "score" if ("min" in q or "max" in q) else "choice"
        elif "min" in q or "max" in q:
            q["type"] = "score"
        else:
            q["type"] = "bool"
    else:
        key = str(raw_type).strip().lower()
        if key not in _TYPE_ALIASES:
            raise ValueError(f"question {name!r}: unknown type {raw_type!r}")
        q["type"] = _TYPE_ALIASES[key]
    return q


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Union[str, dict[str, Any], list[Any]] = Field(..., description="The situation to evaluate.")
    questions: dict[str, Question] = Field(..., min_length=1)
    options: DecideOptions = Field(default_factory=DecideOptions)

    @model_validator(mode="before")
    @classmethod
    def _normalize_types(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("questions"), dict):
            data = dict(data)
            data["questions"] = {
                str(name): _normalize_question(str(name), q) for name, q in data["questions"].items()
            }
        return data


# ----------------------------------------------------------------------------- answers

class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(..., description="Probability of the chosen option.")
    logits: dict[str, float] | None = None


class BoolAnswer(BaseModel):
    type: Literal["bool"] = "bool"
    value: bool
    probability: float = Field(..., description="P(true).")
    confidence: float = Field(..., description="max(P(true), P(false)).")
    logits: dict[str, float] | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: int = Field(..., description="Most probable level.")
    expected: float = Field(..., description="Probability-weighted mean level.")
    probabilities: dict[str, float]
    confidence: float = Field(..., description="Probability of the most probable level.")
    logits: dict[str, float] | None = None


Answer = Union[ChoiceAnswer, BoolAnswer, ScoreAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0  # always 0: nothing is generated
    probes: int = Field(..., description="Number of scored prompts (questions x permutations).")
    forward_passes: int


class DecideResponse(BaseModel):
    answers: dict[str, Answer]
    model: str
    usage: Usage
    latency_ms: float
