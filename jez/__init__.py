"""Jez — decision models that answer Jev's HTTP API field for field.

A request carries a *state* and typed *questions* (``noul``, ``choice``,
``score``). Each question is answered by reading the model's log-probabilities
of the declared answer labels at the answer position. Nothing is generated, so
an answer outside the declared set cannot occur.

Packages:

* :mod:`jez.schema`      Jev's wire format, validation and error mapping
* :mod:`jez.prompt`      the planner: state once, one suffix per question and order
* :mod:`jez.labels`      per-tokenizer tables of single-token answer labels
* :mod:`jez.confidence`  Jev's confidence and score formulas
* :mod:`jez.calibration` temperatures per question type and option-count bucket
* :mod:`jez.engine`      backends (torch on CUDA, MLX on Apple Silicon, fake for tests)
* :mod:`jez.decider`     request -> plan -> label log-probabilities -> Jev answers
* :mod:`jez.server`      FastAPI service: POST /v1/systemone, GET /v1/models, GET /health
"""

__version__ = "1.0.0"
