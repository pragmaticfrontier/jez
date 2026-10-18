"""Shared helpers for the conformance suite: a fake-backend service, example loading, invariants."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

from jez import models as registry
from jez.calibration import Calibration
from jez.confidence import choice_confidence, score_confidence, score_value
from jez.decider import Decider
from jez.engine import FakeBackend
from jez.server import Service, Settings, create_app

EXAMPLES = Path(__file__).parent / "examples"
API_KEY = "test-key-123"


def example_requests() -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(EXAMPLES.glob("*.json"))}


def with_model(req: dict[str, Any], model: str) -> dict[str, Any]:
    out = dict(req)
    out["model"] = model
    return out


def make_service(models: tuple[str, ...] = ("jez-small-1.0.0",), api_keys: tuple[str, ...] = (API_KEY,),
                 no_auth: bool = False, max_queue: int = 32, backend_factory=None) -> Service:
    settings = Settings()
    settings.api_keys = list(api_keys)
    settings.no_auth = no_auth
    settings.max_queue = max_queue
    settings.backend = "fake"
    svc = Service(settings)
    for model_id in models:
        backend = backend_factory(model_id) if backend_factory else FakeBackend(registry.MODELS[model_id].base)
        svc.deciders[model_id] = Decider(backend, model_id, calibration=Calibration())
    return svc


def make_app(**kwargs):
    svc = make_service(**kwargs)
    return create_app(svc.settings, svc)


def check_answer_invariants(req: dict[str, Any], body: dict[str, Any]) -> None:
    assert set(body) == {"model", "answers", "usage"}, body.keys()
    assert body["model"] in registry.MODELS
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    assert set(body["answers"]) == set(req["questions"])
    for qid, q in req["questions"].items():
        a = body["answers"][qid]
        assert a["type"] == q["type"], (qid, a)
        if q["type"] == "noul":
            assert set(a) == {"type", "noul"}
            assert 0.0 <= a["noul"] <= 1.0
        elif q["type"] == "choice":
            assert set(a) == {"type", "choice", "probabilities", "confidence"}
            assert list(a["probabilities"]) == list(q["criteria"])
            probs = list(a["probabilities"].values())
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
            assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
            assert math.isclose(a["confidence"], choice_confidence(probs), abs_tol=2e-4)
        elif q["type"] == "score":
            assert set(a) == {"type", "score", "legend", "probabilities", "confidence"}
            n = len(q["criteria"])
            assert list(a["probabilities"]) == [str(i) for i in range(n)]
            assert list(a["legend"]) == [str(i) for i in range(n)]
            for i, lv in enumerate(q["criteria"]):
                assert a["legend"][str(i)] == (lv if isinstance(lv, str) else json.dumps(lv, separators=(",", ":")))
            probs = list(a["probabilities"].values())
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
            assert math.isclose(a["score"], score_value(probs), abs_tol=2e-4)
            assert math.isclose(a["confidence"], score_confidence(probs), abs_tol=2e-4)
            assert 0.0 <= a["score"] <= n - 1


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """A real uvicorn server in a thread, for SDK tests that need HTTP (compression, retries)."""

    def __init__(self, app) -> None:
        import uvicorn

        self.port = free_port()
        self.config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="on")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LiveServer":
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
