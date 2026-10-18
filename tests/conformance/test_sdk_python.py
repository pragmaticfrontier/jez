"""TypeSafe's Python SDK (typesafe-sdk, pinned in pyproject) against a live Jez server.

Only TYPESAFE_BASE_URL and TYPESAFE_DEFAULT_MODEL change from a Jev deployment. The SDK compresses
bodies of 1 KiB and more with zstd, so the compression path is exercised for real.
"""
from __future__ import annotations

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")

from typesafe_sdk import (  # noqa: E402
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from .helpers import API_KEY, LiveServer, example_requests, make_app  # noqa: E402

NO_RETRY = RetryPolicy(max_retries=0)


@pytest.fixture(scope="module")
def server():
    with LiveServer(make_app(models=("jez-small-1.0.0",))) as s:
        yield s


@pytest.fixture
def env(server, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", API_KEY)
    monkeypatch.setenv("TYPESAFE_BASE_URL", server.url)
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jez-small-latest")
    yield server


@pytest.mark.parametrize("name", sorted(example_requests()))
def test_every_api_reference_example_through_the_sdk(env, name):
    req = example_requests()[name]
    with TypeSafeClient() as client:  # model comes from TYPESAFE_DEFAULT_MODEL, never from the example
        res = client.system_one(state=req["state"], questions=req["questions"])
    assert res.model == "jez-small-1.0.0"
    assert set(res.answers) == set(req["questions"])
    assert res.usage.output_tokens == 0 and res.usage.input_tokens > 0
    for qid, q in req["questions"].items():
        a = res.answers[qid]
        assert a.type == q["type"]
        if q["type"] == "noul":
            assert 0 <= a.noul <= 1
        elif q["type"] == "choice":
            assert a.choice in q["criteria"] and abs(sum(a.probabilities.values()) - 1) < 1e-6
        else:
            assert list(a.legend) == list(range(len(q["criteria"]))) and 0 <= a.confidence <= 1


def test_typed_question_objects_and_grouped_answers(env):
    with TypeSafeClient() as client:
        res = client.system_one(
            state={"message": "I was charged twice. Please help.", "subject": "Duplicate charge"},
            questions={
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
                "urgency": Score(instructions="How urgent?", criteria=["Can wait", "This week", "Today"]),
            },
        )
    assert set(res.nouls) == {"billing"} and set(res.choices) == {"tone"} and set(res.scores) == {"urgency"}
    assert res.scores["urgency"].legend == {0: "Can wait", 1: "This week", 2: "Today"}


def test_large_body_is_sent_zstd_compressed_and_accepted(env):
    long_state = "word " * 5000  # 25 KB, well over the SDK's 1 KiB compression threshold
    with TypeSafeClient() as client:
        res = client.system_one(state=long_state, questions={"q": Noul(instructions="Is this long?")})
    assert res.raw_http_response.headers["Jez-Request-Encoding"] == ("zstd" if client._config.compression else "identity")


def test_models_list(env):
    with TypeSafeClient() as client:
        names = {m.name for m in client.models.list().models}
    assert names == {"jez-small-1.0.0", "jez-small-latest"}


def test_400_raises_bad_request(env):
    nested = "x"
    for _ in range(70):
        nested = [nested]
    with TypeSafeClient(retry=NO_RETRY) as client, pytest.raises(TypeSafeBadRequestError) as e:
        client.system_one(state=nested, questions={"q": Noul(instructions="deep?")})
    assert e.value.status == 400


def test_401_raises_authentication_error(env):
    with TypeSafeClient(api_key="wrong-key", retry=NO_RETRY) as client, pytest.raises(TypeSafeAuthenticationError) as e:
        client.system_one(state="x", questions={"q": Noul(instructions="y")})
    assert e.value.status == 401


def test_413_raises_api_error(env):
    huge = "a" * (32 * 1024 * 1024 + 64)
    with TypeSafeClient(retry=NO_RETRY, timeout=120) as client, pytest.raises(TypeSafeAPIError) as e:
        client.system_one(state=huge, questions={"q": Noul(instructions="big?")})
    assert e.value.status == 413 and type(e.value) is TypeSafeAPIError


def test_415_raises_api_error(env):
    with TypeSafeClient(retry=NO_RETRY, compression=False) as client, pytest.raises(TypeSafeAPIError) as e:
        client.system_one(state="x", questions={"q": Noul(instructions="y")}, extra_headers={"Content-Encoding": "br"})
    assert e.value.status == 415 and type(e.value) is TypeSafeAPIError


def test_422_raises_unprocessable_entity(env):
    with TypeSafeClient(retry=NO_RETRY) as client, pytest.raises(TypeSafeUnprocessableEntityError) as e:
        client.system_one(state="x", questions={"q": {"type": "choice", "instructions": "y", "criteria": {"only": None}}})
    assert e.value.status == 422 and "questions.q.criteria" in str(e.value)


def test_422_for_jev_model_name(env):
    with TypeSafeClient(retry=NO_RETRY) as client, pytest.raises(TypeSafeUnprocessableEntityError):
        client.system_one(state="x", questions={"q": Noul(instructions="y")}, model="jev-latest")


def test_429_raises_rate_limit_error_after_retries():
    with LiveServer(make_app(max_queue=0)) as s:
        policy = RetryPolicy(max_retries=1, backoff_initial=0.1, backoff_max=0.2, timeout=10.0)
        with TypeSafeClient(api_key=API_KEY, base_url=s.url, model="jez-small-latest", retry=policy) as client, \
                pytest.raises(TypeSafeRateLimitError) as e:
            client.system_one(state="x", questions={"q": Noul(instructions="y")})
        assert e.value.status == 429 and e.value.retry_after_ms is not None


def test_529_raises_internal_server_error(env):
    with TypeSafeClient(retry=NO_RETRY) as client, pytest.raises(TypeSafeInternalServerError) as e:
        client.system_one(state="x", questions={"q": Noul(instructions="y")}, model="jez-latest")  # not loaded here
    assert e.value.status == 529
