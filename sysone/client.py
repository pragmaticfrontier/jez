"""Tiny dependency-free HTTP client for a running sysone server."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class SysOneError(RuntimeError):
    pass


class SysOneClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8008", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise SysOneError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from None

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as r:
            return json.load(r)

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], **options: Any) -> dict[str, Any]:
        """Full request: several typed questions about one state, answered in one pass."""
        return self._post("/v1/decide", {"state": state, "questions": questions, "options": options})

    # convenience wrappers for single questions -------------------------------------------------
    def choose(self, state: Any, instructions: str, criteria: dict[str, str] | list[str], **options: Any) -> dict[str, Any]:
        q = {"type": "choice", "instructions": instructions}
        q["criteria" if isinstance(criteria, dict) else "options"] = criteria
        return self.decide(state, {"q": q}, **options)["answers"]["q"]

    def check(self, state: Any, statement: str, **options: Any) -> dict[str, Any]:
        return self.decide(state, {"q": {"type": "bool", "instructions": statement}}, **options)["answers"]["q"]

    def score(self, state: Any, instructions: str, min: int = 1, max: int = 5,
              criteria: dict[str, str] | None = None, **options: Any) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "score", "instructions": instructions, "min": min, "max": max}
        if criteria:
            q["criteria"] = criteria
        return self.decide(state, {"q": q}, **options)["answers"]["q"]
