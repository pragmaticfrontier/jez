"""Engine checks on a real checkpoint. Needs CUDA and the Qwen3.5-0.8B weights in the Hub cache.

The shared-state path (prefix once, suffixes against the replicated cache) must match a single
unpadded forward pass. In fp32 the match is exact to 1e-3; in bf16 the same comparison shows
0.1 to 0.2 log-prob of accumulation noise, which is why the strict check runs in fp32.
Gemma 4 E4B is checked the same way by tools/engine_check.py under tools/gpu_guard.py
(31 GB in fp32), not here.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jez.decider import Decider  # noqa: E402
from jez.prompt import plan  # noqa: E402
from jez.schema import validate_request  # noqa: E402

CHECKPOINT = "Qwen/Qwen3.5-0.8B"


def _cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(CHECKPOINT, "config.json"), str)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not _cached(), reason="needs CUDA and the cached Qwen3.5-0.8B")


@pytest.fixture(scope="module")
def backend():
    from jez.engine.torch_backend import TorchBackend

    be = TorchBackend(CHECKPOINT, dtype=torch.float32)
    yield be
    be.release()


def _probes(backend):
    req = validate_request({
        "state": {"ticket": 4411, "text": "My card was charged twice for the monthly subscription. Fix it today."},
        "model": "jez-small-latest",
        "questions": {
            "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "sales": None}},
            "urgent": {"type": "noul", "instructions": "Does the customer demand same-day action?"},
            "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
            "wide": {"type": "choice", "instructions": "Pick any", "criteria": {f"o{i}": None for i in range(30)}},
        },
    })
    probes = plan(req, backend.table)
    dec = Decider(backend, "t")
    return req, [backend.render(p.messages) for p in probes], [dec._labels_for(p) for p in probes]


def test_shared_state_matches_single_forward_within_0_02(backend):
    _, prompts, labels = _probes(backend)
    backend.share_state = False
    single = [backend.score([p], [l])[0] for p, l in zip(prompts, labels)]
    backend.share_state = True
    shared = backend.score(prompts, labels)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(single, shared))
    assert worst < 0.02, worst


def test_scoring_is_deterministic_and_normalised(backend):
    _, prompts, labels = _probes(backend)
    a = backend.score(prompts, labels)
    b = backend.score(prompts, labels)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for lp in a:
        assert abs(np.exp(lp).sum() - 1.0) < 1e-6


def test_label_only_head_matches_full_head(backend):
    tok, model = backend.tok, backend.model
    text = backend.render([{"role": "user", "content": "The sky is blue. Reply with only A or B. A) true B) false"}])
    ids = torch.tensor([backend.encode(text)], device=backend.device)
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    labels = backend.table.choice_labels(2)
    restricted = backend.score([text], [labels])[0]
    full_pair = torch.stack([torch.logsumexp(full[list(lab.ids)], 0) for lab in labels])
    full_lp = torch.log_softmax(full_pair, 0).cpu().numpy()
    assert np.abs(full_lp - restricted).max() < 1e-3


@pytest.mark.xfail(reason="acceptance gate 4: holds for the fine-tuned release, not for the zero-shot base", strict=False)
def test_order_reversal_moves_no_probability_more_than_0_05(backend):
    """Acceptance gate 4: reordering options must not move any probability by more than 0.05.

    Two-order averaging removes the first-position bias exactly, but the other permutations of three options
    still move the zero-shot base by more than 0.05, so this is expected to fail until the fine-tune lands.
    eval/order_stability.py reports the same measurement on whole test sets.
    """
    import itertools

    dec = Decider(backend, "t")
    crit = {"card": "a card payment problem", "transfer": "a bank transfer problem", "login": "cannot sign in"}
    state = "I sent money to my friend yesterday and it still has not arrived in her account."
    results = []
    for order in itertools.permutations(crit):
        req = validate_request({"state": state, "model": "jez-small-latest",
                                "questions": {"q": {"type": "choice", "instructions": "What is the issue about?",
                                                    "criteria": {k: crit[k] for k in order}}}})
        results.append(dec.decide(req).answers["q"]["probabilities"])
    for k in crit:
        vals = [r[k] for r in results]
        assert max(vals) - min(vals) <= 0.05, (k, vals)
    assert len({max(r, key=r.get) for r in results}) == 1
