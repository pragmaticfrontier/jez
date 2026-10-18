"""Training sources and the mix builder's pure parts: sibling questions, overlap n-grams, validity."""
from __future__ import annotations

import random

from jez.schema import validate_request
from train import build_mix, sources


def _valid(example: dict) -> None:
    req = validate_request({"state": example["state"], "model": "jez-small-latest", "questions": {"q": example["question"]}})
    q = req.questions[0]
    keys = q.names if q.type == "choice" else (["true", "false"] if q.type == "noul" else [str(i) for i in range(q.n)])
    assert example["gold"] in keys, (example["source"], example["gold"], keys[:5])


def test_nli_builder_yields_three_siblings_on_one_state():
    rng = random.Random(0)
    sib = sources.nli("multinli")({"premise": "A man sleeps.", "hypothesis": "A person is resting.", "label": 0}, rng)
    assert [e["question"]["type"] for e in sib] == ["choice", "noul", "score"]
    assert len({str(e["state"]) for e in sib}) == 1
    assert [e["gold"] for e in sib] == ["entailment", "true", "2"]
    for e in sib:
        _valid(e)
    assert sources.nli("multinli")({"premise": "p", "hypothesis": "h", "label": -1}, rng) == []


def test_derived_noul_and_subset_choice_keep_the_gold_option():
    rng = random.Random(1)
    row = {"text": "wake me at nine", "label": "alarm_set", "_intent_names": [f"intent_{i}" for i in range(59)] + ["alarm_set"]}
    full, subset, yesno = sources.massive_builder("en")(row, rng)
    assert full["question"]["type"] == "choice" and len(full["question"]["criteria"]) == 60 and full["gold"] == "alarm_set"
    assert subset["gold"] == "alarm_set" and "alarm_set" in subset["question"]["criteria"] and len(subset["question"]["criteria"]) < 60
    assert yesno["question"]["type"] == "noul" and yesno["gold"] in ("true", "false")
    for e in (full, subset, yesno):
        _valid(e)
    for _ in range(20):
        d = sources.derived_noul(full, rng)
        option = d["question"]["instructions"].split(": ")[-1].rstrip("?")
        assert (d["gold"] == "true") == (option == "alarm_set")


def test_jigsaw_and_ultrafeedback_levels():
    rng = random.Random(2)
    sib = sources.jigsaw({"comment_text": "you idiot", "toxic": 1, "severe_toxic": 0, "insult": 1, "obscene": 0, "threat": 0}, rng)
    assert [e["gold"] for e in sib] == ["true", "2", "false"]
    for e in sib:
        _valid(e)
    row = {"instruction": "Say hi", "completions": [{"response": "Hi!", "overall_score": 9.0}, {"response": "...", "overall_score": 2.0}]}
    for _ in range(5):
        s10, s5, ok = sources.ultrafeedback(row, rng)
        assert len(s10["question"]["criteria"]) == 10 and len(s5["question"]["criteria"]) == 5
        assert int(s5["gold"]) == int(s10["gold"]) // 2
        assert (ok["gold"] == "true") == (int(s10["gold"]) >= 6)
        for e in (s10, s5, ok):
            _valid(e)


def test_every_source_declares_a_licence_and_a_commercial_flag():
    for s in sources.SOURCES.values():
        assert s.licence and s.commercial_ok in (True, False, None)
        assert s.qtypes and all(q in ("choice", "noul", "score") for q in s.qtypes)
    assert all(not sources.SOURCES[n].commercial_ok for n in ("sciq", "race", "yelp_polarity", "xnli_de"))


def test_overlap_ngrams_catch_a_13_token_window():
    eval_text = "the quick brown fox jumps over the lazy dog while the cat sleeps on the warm mat today"
    grams = build_mix._ngrams(build_mix._tokens(eval_text))
    assert build_mix._ngrams(build_mix._tokens("Prefix: " + eval_text + " and more")) & grams
    assert not build_mix._ngrams(build_mix._tokens("the quick brown fox jumps over the lazy dog while the cat")) & grams  # 12 tokens
    assert not build_mix._ngrams(build_mix._tokens({"text": "something else entirely " * 5})) & grams


def test_parquet_sampling_reads_only_the_wanted_rows_across_shards(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from train import parquet

    paths = []
    for shard, start in enumerate((0, 7)):
        n = 7 if shard == 0 else 5
        table = pa.table({"i": list(range(start, start + n)), "label": [f"l{(start + k) % 3}" for k in range(n)]})
        path = tmp_path / f"{shard:04d}.parquet"
        pq.write_table(table, path, row_group_size=3)
        paths.append(path)
    assert parquet.num_rows(paths) == 12
    assert parquet.spread_indices(12, 4) == [0, 3, 7, 11]
    rows = parquet.take_rows(paths, [0, 3, 7, 11])
    assert [r["i"] for r in rows] == [0, 3, 7, 11]
    assert [r["i"] for r in parquet.take_rows(paths, list(range(12)))] == list(range(12))


def test_micro_batches_respect_sequence_and_token_budgets():
    from train.finetune import Item, micro_batches

    def item(n):
        return Item(ids=list(range(n)), labels=[], gold=0, teacher=None)

    items = sorted([item(n) for n in (10, 10, 10, 10, 100, 100, 4000, 4000, 4000)], key=lambda it: len(it.ids))
    chunks = micro_batches(items, micro=4, micro_tokens=8000)
    assert [len(c) for c in chunks] == [4, 2, 2, 1]
    assert all(len(c) <= 4 and len(c) * max(len(it.ids) for it in c) <= 8000 for c in chunks)
    assert sum(len(c) for c in chunks) == len(items)
    assert [len(c) for c in micro_batches([item(9000)], micro=4, micro_tokens=8000)] == [1]  # never drops a sequence


def test_take_quotas_balances_large_yes_no_families_and_keeps_small_ones():
    def noul(source, instr, gold):
        return {"state": "s", "question": {"type": "noul", "instructions": instr}, "gold": gold, "source": source, "language": "en"}
    groups = [[noul("tox", "Is this toxic?", "false")] for _ in range(900)] + [[noul("tox", "Is this toxic?", "true")] for _ in range(100)]
    groups += [[noul("tiny", f"Is the intent {i}?", "false" if i % 4 else "true")] for i in range(40)]  # families of one: left alone
    rng = random.Random(0)
    groups += [[noul("fair", "Is it fair?", rng.choice(["true", "false"]))] for _ in range(2000)]  # balanced by construction
    rng.shuffle(groups)
    choice = {"state": "s", "question": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}, "gold": "a",
              "source": "tox", "language": "en"}
    groups.append([choice, noul("tox", "Is this toxic?", "false")])  # the sibling survives when the yes/no member is dropped
    balanced = build_mix.noul_families(groups, min_pool=200)
    assert set(balanced) == {("tox", "Is this toxic?"), ("fair", "Is it fair?")}
    assert balanced[("tox", "Is this toxic?")] == {"pool_true": 100, "pool_false": 901}
    quota = {"choice": 10, "noul": 10_000, "score": 0}
    mix, taken, balance = build_mix.take_quotas(groups, quota, balanced)
    tox = [e for e in mix if e["source"] == "tox" and e["question"]["type"] == "noul"]
    n_true, n_false = sum(e["gold"] == "true" for e in tox), sum(e["gold"] == "false" for e in tox)
    assert n_true == 100 and 100 <= n_false <= 111  # the commoner answer may lead by 5% of the family (plus one)
    assert sum(e["source"] == "tiny" for e in mix) == 40
    assert balance["dropped"]["tox"] == 901 - n_false and balance["dropped"].get("fair", 0) < 40  # a fair family loses almost nothing
    assert balance["families"]["tox"] == {"Is this toxic?": {"pool_true": 100, "pool_false": 901, "true": 100, "false": n_false}}
    assert choice in mix and len({e["request_id"] for e in mix}) == len(mix)
    # the balance holds when the quota reaches only part of the pool
    mix, taken, balance = build_mix.take_quotas(groups, {"choice": 10, "noul": 300, "score": 0}, balanced)
    tox = [e for e in mix if e["source"] == "tox" and e["question"]["type"] == "noul"]
    n_true, n_false = sum(e["gold"] == "true" for e in tox), sum(e["gold"] == "false" for e in tox)
    assert taken["noul"] == 300 and n_false <= n_true + max(5, 0.05 * (n_true + n_false)) + 1
    unbalanced, taken2, balance2 = build_mix.take_quotas(groups, quota, {})
    assert taken2["noul"] == 3041 and balance2["dropped"] == {}
