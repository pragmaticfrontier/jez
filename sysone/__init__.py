"""sysone — a System-One style decision model.

Give it a *state* (text or JSON) and typed *questions*; get back probability
distributions over fixed answers. It never generates text: every question is
answered by one forward pass of a small language model whose next-token logits
are read out only at the allowed answer labels.
"""
from .schema import (
    BoolAnswer,
    BoolQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideOptions,
    DecideRequest,
    DecideResponse,
    ScoreAnswer,
    ScoreQuestion,
)

__version__ = "0.1.0"
__all__ = [
    "BoolAnswer", "BoolQuestion", "ChoiceAnswer", "ChoiceQuestion", "DecideOptions",
    "DecideRequest", "DecideResponse", "ScoreAnswer", "ScoreQuestion", "__version__",
]
