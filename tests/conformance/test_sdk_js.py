"""TypeSafe's JavaScript SDK (@typesafe-ai/sdk 0.7.0) against a live Jez server, driven through node.

Skipped when node or npm is missing, or when the SDK cannot be installed (no network).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .helpers import API_KEY, EXAMPLES, LiveServer, make_app

JS_DIR = Path(__file__).resolve().parent.parent / "sdk_js"


@pytest.fixture(scope="module")
def node_sdk():
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        pytest.skip("node and npm are required for the JavaScript SDK test")
    if not (JS_DIR / "node_modules" / "@typesafe-ai" / "sdk").exists():
        r = subprocess.run([npm, "install", "--no-audit", "--no-fund", "--silent"], cwd=JS_DIR, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            pytest.skip(f"npm install failed: {r.stderr[-500:]}")
    return node


def test_javascript_sdk(node_sdk):
    with LiveServer(make_app(models=("jez-small-1.0.0",))) as server, LiveServer(make_app(max_queue=0)) as limited:
        env = {**os.environ, "TYPESAFE_API_KEY": API_KEY, "TYPESAFE_BASE_URL": server.url,
               "TYPESAFE_DEFAULT_MODEL": "jez-small-latest", "JEZ_EXPECT_MODEL": "jez-small-1.0.0",
               "JEZ_RATE_LIMITED_URL": limited.url}
        r = subprocess.run([node_sdk, str(JS_DIR / "run.mjs"), str(EXAMPLES)], cwd=JS_DIR, env=env,
                           capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    report = json.loads(r.stdout.strip().splitlines()[-1])
    for name, result in report["examples"].items():
        assert result["ok"], (name, result)
        assert result["usage"]["output_tokens"] == 0
    assert report["typed"]["ok"], report["typed"]
    assert report["models"] == ["jez-small-1.0.0", "jez-small-latest"]
    expected = {"400": "BadRequestError", "401": "AuthenticationError", "413": "APIError", "415": "APIError",
                "422": "UnprocessableEntityError", "422-jev": "UnprocessableEntityError", "429": "RateLimitError",
                "529": "InternalServerError"}
    for code, cls in expected.items():
        got = report["errors"][code]
        assert got["raised"] == cls, (code, got)
        assert got["status"] == int(code.split("-")[0]), (code, got)
    assert report["errors"]["429"]["retryAfterMs"] is not None
