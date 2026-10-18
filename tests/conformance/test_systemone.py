"""Conformance suite for POST /v1/systemone and GET /v1/models against TypeSafe's API reference.

Runs against the fake backend, so it covers the contract (shapes, limits, status codes), not model quality.
"""
from __future__ import annotations

import gzip
import json
import math

import pytest
import zstandard
from fastapi.testclient import TestClient

from jez import models as registry
from jez.confidence import choice_confidence, score_confidence, score_value
from jez.server import MAX_BODY

from .helpers import API_KEY, check_answer_invariants, example_requests, make_app, with_model

HEADERS = {"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}


@pytest.fixture(scope="module")
def client():
    with TestClient(make_app(models=("jez-small-1.0.0", "jez-large-1.0.0"))) as c:
        yield c


@pytest.fixture(scope="module")
def small_only():
    with TestClient(make_app(models=("jez-small-1.0.0",))) as c:
        yield c


def post(client, body, headers=HEADERS, raw: bytes | None = None, **kw):
    data = raw if raw is not None else json.dumps(body).encode()
    return client.post("/v1/systemone", content=data, headers=headers, **kw)


def base_request(**questions):
    return {"state": "Help! My payouts have been failing for 3 days.", "model": "jez-small-latest", "questions": questions}


# ----------------------------------------------------------------------------- examples from the API reference

@pytest.mark.parametrize("name", sorted(example_requests()))
def test_api_reference_examples(client, name):
    req = with_model(example_requests()[name], "jez-small-latest")
    r = post(client, req)
    assert r.status_code == 200, r.text
    body = r.json()
    check_answer_invariants(req, body)
    assert body["model"] == "jez-small-1.0.0"
    assert "Jez-Latency-Ms" in r.headers and "Jez-Cost-Usd" in r.headers
    assert float(r.headers["Jez-Latency-Ms"]) >= 0 and float(r.headers["Jez-Cost-Usd"]) >= 0


def test_examples_with_jev_model_name_fail_loudly(client):
    for name, req in example_requests().items():
        r = post(client, req)  # "model": "jev-latest" verbatim
        assert r.status_code == 422, name
        assert r.json()["error"]["field"] == "model"


def test_score_example_shape_matches_reference(client):
    req = with_model(example_requests()["score"], "jez-small-latest")
    a = post(client, req).json()["answers"]["frustration"]
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert list(a["probabilities"]) == ["0", "1", "2"]


# ----------------------------------------------------------------------------- answer types and criteria shapes

def test_mixed_types_in_one_request(client):
    req = base_request(
        is_urgent={"type": "noul", "instructions": "Does this convey urgency?"},
        department={"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": None}},
        frustration={"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]},
    )
    r = post(client, req)
    assert r.status_code == 200
    check_answer_invariants(req, r.json())
    assert list(r.json()["answers"]) == ["is_urgent", "department", "frustration"]


def test_structured_criteria_everywhere(client):
    req = base_request(
        n={"type": "noul", "instructions": ["Evaluate", {"focus": "urgency"}],
           "criteria": {"true": {"means": "time-sensitive"}, "false": ["no deadline", "no pressure"]}},
        c={"type": "choice", "instructions": {"task": "route"},
           "criteria": {"a": {"when": "payments"}, "b": ["bugs", "outages"], "c": None, "d": ""}},
        s={"type": "score", "instructions": {"rate": "anger"}, "criteria": [{"level": "calm"}, ["mild"], "furious"]},
    )
    r = post(client, req)
    assert r.status_code == 200, r.text
    check_answer_invariants(req, r.json())
    assert r.json()["answers"]["s"]["legend"] == {"0": '{"level":"calm"}', "1": '["mild"]', "2": "furious"}


def test_noul_without_instructions_but_with_criteria(client):
    req = base_request(q={"type": "noul", "criteria": {"true": "Urgent", "false": "Not urgent"}})
    assert post(client, req).status_code == 200


@pytest.mark.parametrize("n", [2, 26, 27, 255])
def test_choice_option_counts(client, n):
    req = base_request(q={"type": "choice", "instructions": "Pick one", "criteria": {f"opt_{i}": f"option {i}" for i in range(n)}})
    r = post(client, req)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["q"]
    assert len(a["probabilities"]) == n
    check_answer_invariants(req, r.json())


@pytest.mark.parametrize("n", [1, 256])
def test_choice_option_counts_out_of_range(client, n):
    req = base_request(q={"type": "choice", "instructions": "Pick one", "criteria": {f"opt_{i}": None for i in range(n)}})
    r = post(client, req)
    assert r.status_code == 422 and r.json()["error"]["field"] == "questions.q.criteria"


@pytest.mark.parametrize("n", [2, 10])
def test_score_level_counts(client, n):
    req = base_request(q={"type": "score", "instructions": "Rate", "criteria": [f"level {i}" for i in range(n)]})
    r = post(client, req)
    assert r.status_code == 200
    assert list(r.json()["answers"]["q"]["probabilities"]) == [str(i) for i in range(n)]
    check_answer_invariants(req, r.json())


@pytest.mark.parametrize("n", [1, 11])
def test_score_level_counts_out_of_range(client, n):
    req = base_request(q={"type": "score", "instructions": "Rate", "criteria": [f"level {i}" for i in range(n)]})
    r = post(client, req)
    assert r.status_code == 422 and r.json()["error"]["field"] == "questions.q.criteria"


def test_confidence_formulas_match_typesafe_examples():
    assert round(choice_confidence([0.88, 0.12, 0.0]), 2) == 0.82  # TypeSafe prints 0.81 from the unrounded value
    assert math.isclose(score_value([0, 0.95, 0.05]), 1.05)
    assert math.isclose(score_confidence([0, 0.95, 0.05]), 0.925)
    assert math.isclose(score_confidence([0, 0.5, 0.5]), 0.25)
    assert score_confidence([0.5, 0, 0.5]) == 0.0
    assert round(score_confidence([0, 0.57, 0.43]), 2) == 0.35
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)


def test_identical_requests_return_identical_bytes(client):
    req = with_model(example_requests()["choice"], "jez-small-latest")
    a, b = post(client, req), post(client, req)
    assert a.content == b.content


def test_question_ids_never_reach_the_model(client):
    """Renaming a question id must not change its answer."""
    q = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs"}}
    a = post(client, base_request(first=q)).json()["answers"]["first"]
    b = post(client, base_request(totally_different_id=q)).json()["answers"]["totally_different_id"]
    assert a == b


def test_option_order_does_not_change_probabilities(client):
    """The fake backend has no position bias, so reordering options must give the same per-option probability."""
    crit = {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}
    a = post(client, base_request(q={"type": "choice", "instructions": "Which team?", "criteria": crit})).json()["answers"]["q"]
    rev = dict(reversed(list(crit.items())))
    b = post(client, base_request(q={"type": "choice", "instructions": "Which team?", "criteria": rev})).json()["answers"]["q"]
    for k in crit:
        assert math.isclose(a["probabilities"][k], b["probabilities"][k], abs_tol=1e-5)


# ----------------------------------------------------------------------------- models

def test_models_endpoint(client):
    r = client.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["models"]}
    assert names == {"jez-small-1.0.0", "jez-small-latest", "jez-large-1.0.0", "jez-large-latest", "jez-latest"}
    for m in r.json()["models"]:
        assert set(m) == {"name", "description", "release_date"}
        assert m["release_date"].count("-") == 2


def test_models_endpoint_requires_auth(client):
    assert client.get("/v1/models").status_code == 401


@pytest.mark.parametrize("name,expected", [
    ("jez-small-1.0.0", "jez-small-1.0.0"), ("jez-small-latest", "jez-small-1.0.0"),
    ("jez-large-1.0.0", "jez-large-1.0.0"), ("jez-large-latest", "jez-large-1.0.0"), ("jez-latest", "jez-large-1.0.0"),
])
def test_versioned_and_aliased_model_ids(client, name, expected):
    req = with_model(example_requests()["noul"], name)
    r = post(client, req)
    assert r.status_code == 200 and r.json()["model"] == expected


@pytest.mark.parametrize("name", ["jev-latest", "jev-preview", "jev-1.13.0", "jev-1.14.0", "gpt-4", "", "jez-medium-latest"])
def test_unknown_and_jev_model_ids(client, name):
    r = post(client, with_model(example_requests()["noul"], name))
    assert r.status_code == 422 and r.json()["error"]["field"] == "model"


def test_known_model_not_loaded_is_529(small_only):
    r = post(small_only, with_model(example_requests()["noul"], "jez-latest"))
    assert r.status_code == 529
    assert "retry-after" in r.headers
    assert r.json()["error"]["field"] == "model"


# ----------------------------------------------------------------------------- 422: fields

@pytest.mark.parametrize("mutate,field", [
    (lambda b: b.pop("state"), "state"),
    (lambda b: b.pop("model"), "model"),
    (lambda b: b.pop("questions"), "questions"),
    (lambda b: b.update(questions={}), "questions"),
    (lambda b: b.update(questions=[]), "questions"),
    (lambda b: b.update(state=42), "state"),
    (lambda b: b.update(assets={"img": {"type": "image", "data_base64": "AAAA"}}), "assets"),
    (lambda b: b["questions"]["q"].pop("type"), "questions.q.type"),
    (lambda b: b["questions"]["q"].update(type="box"), "questions.q.type"),
    (lambda b: b["questions"]["q"].update(type="bool"), "questions.q.type"),
    (lambda b: b["questions"]["q"].pop("instructions"), "questions.q.instructions"),
    (lambda b: b["questions"]["q"].update(instructions=7), "questions.q.instructions"),
    (lambda b: b["questions"]["q"].update(criteria={"true": "x", "maybe": "y"}), "questions.q.criteria"),
    (lambda b: b["questions"]["q"].update(type="choice"), "questions.q.criteria"),
    (lambda b: b["questions"]["q"].update(type="choice", criteria=["a", "b"]), "questions.q.criteria"),
    (lambda b: b["questions"]["q"].update(type="choice", criteria={"a": 1, "b": None}), "questions.q.criteria.a"),
    (lambda b: b["questions"]["q"].update(type="score"), "questions.q.criteria"),
    (lambda b: b["questions"]["q"].update(type="score", criteria={"0": "a", "1": "b"}), "questions.q.criteria"),
    (lambda b: b["questions"]["q"].update(type="score", criteria=["a", None]), "questions.q.criteria[1]"),
    (lambda b: b.update(questions={"q": "not an object"}), "questions.q"),
])
def test_422_names_the_field(client, mutate, field):
    body = base_request(q={"type": "noul", "instructions": "Is it urgent?"})
    mutate(body)
    r = post(client, body)
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert set(err) >= {"type", "message"} and err["field"] == field


def test_non_object_body_is_422(client):
    assert post(client, None, raw=b"[1, 2]").status_code == 422
    assert post(client, None, raw=b'"a string"').status_code == 422


# ----------------------------------------------------------------------------- 400: request rules

@pytest.mark.parametrize("raw", [b"", b"{", b'{"state": }', b"\xef\xbb\xbf{}", b"\xff\xfe{}", b'{"state": "\xff\xfe"}'])
def test_invalid_json_or_encoding_is_400(client, raw):
    r = post(client, None, raw=raw)
    assert r.status_code == 400, raw
    assert r.json()["error"]["type"] in ("invalid_json", "invalid_request")


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_non_finite_numbers_are_400(client, number):
    raw = ('{"state": {"amount": ' + number + '}, "model": "jez-small-latest", "questions": {"q": {"type": "noul", "instructions": "x"}}}').encode()
    r = post(client, None, raw=raw)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_json"


def _nested(depth: int):
    v = "leaf"
    for _ in range(depth):
        v = [v]
    return v


@pytest.mark.parametrize("where", ["state", "instructions", "criteria"])
def test_nesting_limit_64(client, where):
    def build(depth):
        body = base_request(q={"type": "noul", "instructions": "Is it urgent?"})
        if where == "state":
            body["state"] = _nested(depth)
        elif where == "instructions":
            body["questions"]["q"]["instructions"] = _nested(depth)
        else:
            body["questions"]["q"]["criteria"] = {"true": _nested(depth - 1), "false": "no"}  # the criteria object is one level
        return body

    assert post(client, build(64)).status_code == 200
    r = post(client, build(65))
    assert r.status_code == 400 and r.json()["error"]["type"] == "nesting_too_deep"


def test_nesting_far_beyond_recursion_limit_is_400(client):
    raw = ('{"state": ' + "[" * 20000 + "]" * 20000 + ', "model": "jez-small-latest", "questions": {"q": {"type": "noul", "instructions": "x"}}}').encode()
    assert post(client, None, raw=raw).status_code == 400


def test_token_limits_per_question_and_per_request(client):
    # The fake backend counts whitespace-separated words as tokens.
    words = lambda n: " ".join(["w"] * n)
    q = lambda n: {"type": "noul", "instructions": words(n)}
    ok = base_request(q=q(100))
    ok["state"] = words(registry.PER_QUESTION_CONTEXT - 200)
    assert post(client, ok).status_code == 200
    bad = base_request(q=q(registry.PER_QUESTION_CONTEXT))
    bad["state"] = words(100)
    r = post(client, bad)
    assert r.status_code == 400 and r.json()["error"]["type"] == "max_tokens_exceeded" and r.json()["error"]["field"] == "questions.q"
    per_request = base_request(**{f"q{i}": q(9_000) for i in range(6)})
    per_request["state"] = words(9_000)
    assert post(client, per_request).status_code == 200  # 63k < 64k, each question 18k < 32k
    per_request["state"] = words(11_000)  # 65k > 64k, each question 20k < 32k
    r = post(client, per_request)
    assert r.status_code == 400 and r.json()["error"]["type"] == "max_tokens_exceeded" and r.json()["error"]["field"] == "questions"


# ----------------------------------------------------------------------------- 413: body size

def _padded_body(total: int) -> bytes:
    head = b'{"state": "'
    tail = b'", "model": "jez-small-latest", "questions": {"q": {"type": "noul", "instructions": "x"}}}'
    return head + b"a" * (total - len(head) - len(tail)) + tail


def test_body_at_the_limit_and_one_byte_past(client):
    assert post(client, None, raw=_padded_body(MAX_BODY)).status_code == 200
    r = post(client, None, raw=_padded_body(MAX_BODY + 1))
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"


def test_decompressed_body_over_the_limit_is_413(client):
    raw = gzip.compress(_padded_body(MAX_BODY + 1024))
    assert len(raw) < MAX_BODY
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": "gzip"})
    assert r.status_code == 413


# ----------------------------------------------------------------------------- compression

@pytest.mark.parametrize("encoding", ["gzip", "x-gzip"])
def test_gzip_bodies(client, encoding):
    req = with_model(example_requests()["choice"], "jez-small-latest")
    raw = gzip.compress(json.dumps(req).encode())
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": encoding})
    assert r.status_code == 200, r.text
    check_answer_invariants(req, r.json())
    assert r.headers["Jez-Request-Encoding"] == encoding


def test_zstd_bodies(client):
    req = with_model(example_requests()["score"], "jez-small-latest")
    raw = zstandard.ZstdCompressor().compress(json.dumps(req).encode())
    assert len(raw) > 0
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": "zstd"})
    assert r.status_code == 200, r.text
    check_answer_invariants(req, r.json())


def test_zstd_window_within_8mib_is_accepted(client):
    req = with_model(example_requests()["score"], "jez-small-latest")
    params = zstandard.ZstdCompressionParameters(window_log=23)  # 8 MiB window
    raw = zstandard.ZstdCompressor(compression_params=params).compress(json.dumps(req).encode() + b" " * (12 * 1024 * 1024))
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": "zstd"})
    assert r.status_code == 200, r.text


def test_zstd_window_over_8mib_is_400(client):
    req = with_model(example_requests()["score"], "jez-small-latest")
    params = zstandard.ZstdCompressionParameters(window_log=25)  # 32 MiB window; the content must exceed 8 MiB for the frame to use it
    raw = zstandard.ZstdCompressor(compression_params=params).compress(json.dumps(req).encode() + b" " * (12 * 1024 * 1024))
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": "zstd"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_encoding"


@pytest.mark.parametrize("encoding", ["br", "deflate", "gzip, zstd", "zstd, gzip", "gzip, gzip"])
def test_unsupported_or_stacked_encodings_are_415(client, encoding):
    raw = gzip.compress(json.dumps(with_model(example_requests()["noul"], "jez-small-latest")).encode())
    r = post(client, None, raw=raw, headers={**HEADERS, "content-encoding": encoding})
    assert r.status_code == 415 and r.json()["error"]["type"] == "unsupported_encoding"


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_corrupt_compressed_bodies_are_400(client, encoding):
    r = post(client, None, raw=b"\x00\x01\x02not compressed", headers={**HEADERS, "content-encoding": encoding})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_encoding"


# ----------------------------------------------------------------------------- auth, queue, overload

@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer wrong"}, {"authorization": "Basic abc"}, {"authorization": "Bearer "}])
def test_missing_or_wrong_token_is_401(client, headers):
    r = post(client, example_requests()["noul"], headers={"content-type": "application/json", **headers})
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"


def test_no_auth_mode():
    with TestClient(make_app(no_auth=True, api_keys=())) as c:
        r = c.post("/v1/systemone", json=with_model(example_requests()["noul"], "jez-small-latest"))
        assert r.status_code == 200


def test_queue_limit_is_429_with_retry_after():
    with TestClient(make_app(max_queue=0)) as c:
        r = c.post("/v1/systemone", json=with_model(example_requests()["noul"], "jez-small-latest"), headers=HEADERS)
        assert r.status_code == 429 and "retry-after" in r.headers and r.json()["error"]["type"] == "rate_limited"


def test_out_of_memory_is_529():
    class OutOfMemoryError(RuntimeError):
        pass

    from jez.engine import FakeBackend

    class Exploding(FakeBackend):
        def score(self, prompts, labels):
            raise OutOfMemoryError("CUDA out of memory")

    with TestClient(make_app(backend_factory=lambda mid: Exploding())) as c:
        r = c.post("/v1/systemone", json=with_model(example_requests()["noul"], "jez-small-latest"), headers=HEADERS)
        assert r.status_code == 529 and "retry-after" in r.headers


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "jez-small-1.0.0" in r.json()["models"]
