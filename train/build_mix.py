"""Build the training mix: sample sources to the quotas, drop evaluation overlaps, write the manifest.

    .venv/bin/python -m train.build_mix --total 300000 --out train/data/mix-v1 [--allow-noncommercial]

Quotas (from the task document): 50% choice, 30% noul, 20% score; at least 5% of choice examples
with more than 26 options; score examples spanning 2 to 10 levels; 40% of states as JSON objects or
arrays; 30% of requests holding 2 to 8 questions (recorded as ``request_id`` groups); 30% of examples
non-English. Each example is one Jev question with its gold answer, and multi-question requests are
groups of examples that share a state.

Yes/no balance: within every family of yes/no questions (one source, one instruction text) with at least
200 examples in the pool, the mix takes about as many "true" as "false" answers (one may lead by 5%),
dropping the surplus of the commoner answer as it comes. Without it the Jigsaw toxicity questions are 94% "false" and the NLI
entailment question 2:1 "false", and a model trained on that answers "false" with high confidence to
every safety-flavoured question (mix-v1; ToxicChat F1 fell from 0.42 to 0.18). Small families, such as
derived questions whose instruction names one option, are balanced by construction and left alone.
``--no-balance-noul`` turns the balance off; the manifest records the caps and the drops per source.

Overlap filter: any training state that shares a 13-token window (whitespace tokens, lower-cased)
with an evaluation item is dropped, and the drop count per source goes into the manifest. The
evaluation items are every cached Tier 1 sample under eval/data plus every Tier 2 file.

Outputs ``<out>/train.jsonl``, ``<out>/val.jsonl`` (5%), ``<out>/calibration.jsonl`` (2%, for fitting
temperatures) and ``<out>/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import urllib.error

from eval.data import cached_rows, fetch_rows, label_names
from eval.tasks import TASKS

from . import parquet
from .sources import SOURCES, Example

log = logging.getLogger("train.mix")
NGRAM = 13
NOUL_BALANCE_MIN_POOL = 200
NOUL_BALANCE_SLACK = 0.05  # one answer may lead the other by 5% of the family (at least 5 examples)


def _tokens(value) -> list[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text.lower().split()


def _ngrams(tokens: list[str], n: int = NGRAM):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def evaluation_ngrams() -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for task in TASKS.values():
        if task.tier == 2:
            try:
                rows = task.rows(None)
            except FileNotFoundError:
                continue
        else:
            rows = cached_rows(task.name)  # the samples (or full splits after a --full run) already on disk
            if not rows:
                log.warning("%s: no cached evaluation rows; run eval first so the overlap filter covers it", task.name)
        for r in rows:
            state, q, _ = task.build(r)
            grams |= _ngrams(_tokens(state))
            if q.get("instructions"):
                grams |= _ngrams(_tokens(q["instructions"]))
    return grams


def noul_family(e: Example) -> tuple[str, str]:
    """The question family of a yes/no example: its source and instruction text."""
    return e["source"], e["question"].get("instructions", "")


def noul_families(groups: list[list[Example]], min_pool: int = NOUL_BALANCE_MIN_POOL) -> dict[tuple[str, str], dict[str, int]]:
    """The yes/no families with at least ``min_pool`` examples in the pool, with their pool counts per answer. These
    are the families the mix balances; smaller ones are left alone."""
    counts: Counter = Counter()
    for g in groups:
        for e in g:
            if e["question"]["type"] == "noul":
                counts[(noul_family(e), e["gold"])] += 1
    families: dict[tuple[str, str], dict[str, int]] = {}
    for (family, gold), n in counts.items():
        other = counts.get((family, "false" if gold == "true" else "true"), 0)
        if n + other >= min_pool:
            families[family] = {"pool_true": counts[(family, "true")], "pool_false": counts[(family, "false")]}
    return families


def take_quotas(ordered: list[list[Example]], quota: dict[str, int], balanced: dict[tuple[str, str], dict[str, int]]):
    """Walk the groups in order and take each member while its type's quota is open. A yes/no member of a balanced
    family is dropped when its answer already leads the other answer taken for that family by more than the slack
    (5% of the family, at least 5), so the family ends near 50/50 whatever the order and however much of the pool the
    quota reaches, while a family that is balanced by construction loses almost nothing to the random walk of its
    counts. The members taken from one group share a request_id. Returns the mix, the per-type counts and the
    balance record for the manifest."""
    taken: Counter = Counter()
    answers: Counter = Counter()
    dropped: Counter = Counter()
    mix: list[Example] = []
    request_id = 0
    for g in ordered:
        members = []
        for e in g:
            t = e["question"]["type"]
            if taken[t] >= quota[t]:
                continue
            if t == "noul":
                family = noul_family(e)
                if family in balanced:
                    other = "false" if e["gold"] == "true" else "true"
                    mine, theirs = answers[(family, e["gold"])], answers[(family, other)]
                    if mine > theirs + max(5, NOUL_BALANCE_SLACK * (mine + theirs)):
                        dropped[e["source"]] += 1
                        continue
                answers[(family, e["gold"])] += 1
            taken[t] += 1
            members.append(e)
        if not members:
            continue
        for e in members:
            e["request_id"] = request_id
        mix.extend(members)
        request_id += 1
    families: dict[str, dict[str, dict[str, int]]] = {}
    for (source, instructions), pool in sorted(balanced.items()):
        families.setdefault(source, {})[instructions] = dict(
            pool, true=answers[((source, instructions), "true")], false=answers[((source, instructions), "false")])
    return mix, taken, {"families": families, "dropped": dict(dropped)}


def _collect_values(dataset: str, config: str, split: str, feature: str, pages: int = 30) -> list[str]:
    """Distinct values of a string feature over the split (intent lists): the whole column from the parquet
    shards, or pages spread across the split through the rows API when there is no parquet conversion."""
    from eval.data import CACHE_DIR

    try:
        return parquet.column_values(dataset, config, split, feature)
    except (FileNotFoundError, urllib.error.HTTPError, KeyError) as e:
        log.warning("%s/%s: no parquet conversion (%s); collecting %s over %d pages", dataset, config, e, feature, pages)
    cache = CACHE_DIR / f"labels-{dataset.replace('/', '__')}-{config}-{feature}.collected.json"
    if cache.exists():
        return json.loads(cache.read_text())
    first, total = fetch_rows(dataset, config, split, 0, 100)
    values = {str(r[feature]) for r in first}
    for i in range(1, pages):
        rows, _ = fetch_rows(dataset, config, split, int(i * (total - 100) / (pages - 1)), 100)
        values |= {str(r[feature]) for r in rows}
    cache.write_text(json.dumps(sorted(values)))
    return sorted(values)


def _rows_values(dataset: str, config: str, split: str, field: str) -> list[str]:
    """Every row's value of ``field`` in a small auxiliary config (an intents list), in row order; '*' takes the first string field."""
    rows, total = fetch_rows(dataset, config, split, 0, 100)
    offset = 100
    while len(rows) < total:
        more, _ = fetch_rows(dataset, config, split, offset, 100)
        if not more:
            break
        rows.extend(more)
        offset += 100
    if field == "*":
        field = next(k for k, v in rows[0].items() if isinstance(v, str))
    return [str(r[field]) for r in rows]


def _source_rows(src, limit: int):
    """``limit`` rows spread over the split: from the parquet shards (one download per shard, cached under
    train/data/parquet) or, when the dataset has no parquet conversion, page by page through the rows API."""
    try:
        rows, total = parquet.sample_rows(src.dataset, src.config, src.split, limit)
        log.info("%s: %d of %d rows from parquet", src.name, len(rows), total)
        yield from rows
        return
    except (FileNotFoundError, urllib.error.HTTPError, KeyError) as e:
        log.warning("%s: no parquet conversion (%s); paging through the rows API", src.name, e)
    first, total = fetch_rows(src.dataset, src.config, src.split, 0, 100)
    want = min(limit, total)
    n_chunks = max(1, -(-want // 100))
    offsets = [0] if n_chunks == 1 else [int(i * (total - 100) / (n_chunks - 1)) for i in range(n_chunks)]
    log.info("%s: %d of %d rows in %d pages", src.name, want, total, n_chunks)
    for i, off in enumerate(offsets):
        if i and i % 25 == 0:
            log.info("%s: page %d/%d", src.name, i, n_chunks)
        yield from (first if i == 0 else fetch_rows(src.dataset, src.config, src.split, off, 100)[0])


def iter_source(src, limit: int, rng: random.Random):
    """Sibling examples per row of a source, rows spread over the split.

    ``needs`` values: ``feature`` (ClassLabel names from the info endpoint), ``collect:feature`` (distinct string values
    seen in the split) or ``rows:config/split:field`` (an auxiliary config listing the names in order).
    """
    extra = {}
    for key, feat in src.needs.items():
        if feat.startswith("collect:"):
            extra[key] = _collect_values(src.dataset, src.config, src.split, feat.split(":", 1)[1])
        elif feat.startswith("rows:"):
            spec, field = feat[len("rows:"):].rsplit(":", 1)
            config, split = spec.split("/", 1)
            extra[key] = _rows_values(src.dataset, config, split, field)
        else:
            extra[key] = label_names(src.dataset, src.config, feat)
    for r in _source_rows(src, limit):
        yield src.build(dict(r, **extra), rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=300_000)
    ap.add_argument("--out", default="train/data/mix-v1")
    ap.add_argument("--allow-noncommercial", action="store_true", help="include sources whose licence forbids commercial use")
    ap.add_argument("--sources", nargs="*", help="restrict to these sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance-noul", dest="balance_noul", action="store_false",
                    help="take yes/no questions as they come instead of balancing true and false per question family")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = [s for s in SOURCES.values() if not args.sources or s.name in args.sources]
    allowed, excluded = [], []
    for s in sources:
        if s.commercial_ok is True or args.allow_noncommercial:
            allowed.append(s)
        else:
            excluded.append(s)
    for s in excluded:
        log.info("excluded %s (%s)", s.name, s.licence)

    quota = {"choice": int(args.total * 0.5), "noul": int(args.total * 0.3), "score": args.total - int(args.total * 0.8)}
    per_type_sources = {t: [s for s in allowed if t in s.qtypes] for t in quota}
    # rows to fetch per source: its share of each type's quota, divided by the examples a row yields on average
    # (all siblings for the 30% multi-question groups, one otherwise), times three so that the quota trim below has
    # a surplus to choose from when some sources are small or capped by the non-English target
    per_source_limit = {s.name: 0 for s in allowed}
    for t, srcs in per_type_sources.items():
        for s in srcs:
            per_source_limit[s.name] += quota[t] // max(1, len(srcs))
    for s in allowed:
        per_row = 0.3 * min(len(s.qtypes), 8) + 0.7
        per_source_limit[s.name] = int(per_source_limit[s.name] / per_row * 3) + 100

    eval_grams = evaluation_ngrams()
    log.info("evaluation 13-grams: %d", len(eval_grams))
    groups: list[list[Example]] = []  # sibling questions about one row's state
    drops = Counter()
    counts = Counter()
    for s in allowed:
        n = 0
        for siblings in iter_source(s, per_source_limit[s.name], rng):
            if not siblings:
                continue
            counts[s.name] += 1
            if _ngrams(_tokens(siblings[0]["state"])) & eval_grams:
                drops[s.name] += 1
                continue
            # 30% of requests hold several questions about one state; the rest hold one
            if len(siblings) >= 2 and rng.random() < 0.30:
                groups.append(siblings[:8])
            else:
                groups.append([rng.choice(siblings)])
            n += 1
        log.info("%s: %d rows kept, %d dropped for overlap", s.name, n, drops[s.name])

    # trim to the type quotas group by group. Wide choice sets (more than 26 options) go first until they make
    # 5% of the choice quota, then non-English groups until they make 30% of the total, then the English groups,
    # and the surplus non-English groups last so that they only fill what the English sources cannot.
    rng.shuffle(groups)
    wide = lambda g: any(e["question"]["type"] == "choice" and len(e["question"]["criteria"]) > 26 for e in g)
    foreign = lambda g: any(e.get("language", "en") != "en" for e in g)
    need_wide = int(quota["choice"] * 0.05)
    need_foreign = int(args.total * 0.30)
    ordered: list[list[Example]] = []
    rest: list[list[Example]] = []
    for g in groups:
        if wide(g) and need_wide > 0:
            ordered.append(g)
            need_wide -= sum(1 for e in g if e["question"]["type"] == "choice" and len(e["question"]["criteria"]) > 26)
        else:
            rest.append(g)
    english: list[list[Example]] = []
    surplus_foreign: list[list[Example]] = []
    for g in rest:
        if foreign(g) and need_foreign > 0:
            ordered.append(g)
            need_foreign -= len(g)
        elif foreign(g):
            surplus_foreign.append(g)
        else:
            english.append(g)
    ordered.extend(english)
    ordered.extend(surplus_foreign)
    balanced = noul_families(groups) if args.balance_noul else {}
    mix, taken, balance = take_quotas(ordered, quota, balanced)
    balance["min_pool"] = NOUL_BALANCE_MIN_POOL if args.balance_noul else None
    for source, n in sorted(balance["dropped"].items()):
        log.info("%s: %d yes/no examples dropped to balance true and false", source, n)
    for t in quota:
        if taken[t] < quota[t]:
            log.warning("%s quota not filled: %d of %d", t, taken[t], quota[t])
    # split by request so that the questions about one state never straddle train and validation
    requests: dict[int, list[Example]] = {}
    for e in mix:
        requests.setdefault(e["request_id"], []).append(e)
    order = list(requests)
    rng.shuffle(order)
    val, cal, train = [], [], []
    for rid in order:
        members = requests[rid]
        target = val if len(val) < int(len(mix) * 0.05) else cal if len(cal) < int(len(mix) * 0.02) else train
        target.extend(members)
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val), ("calibration", cal)):
        with (out / f"{name}.jsonl").open("w") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def stats(rows):
        c = Counter(e["question"]["type"] for e in rows)
        return {
            "n": len(rows), "by_type": dict(c),
            "choice_over_26_options": sum(1 for e in rows if e["question"]["type"] == "choice" and len(e["question"]["criteria"]) > 26),
            "score_levels": dict(Counter(len(e["question"]["criteria"]) for e in rows if e["question"]["type"] == "score")),
            "json_states": sum(1 for e in rows if not isinstance(e["state"], str)),
            "non_english": sum(1 for e in rows if e.get("language", "en") != "en"),
            "requests": len({e["request_id"] for e in rows}),
            "multi_question_requests": sum(1 for _, k in Counter(e["request_id"] for e in rows).items() if k > 1),
            "by_source": dict(Counter(e["source"] for e in rows)),
        }

    manifest = {
        "total_requested": args.total, "seed": args.seed, "ngram": NGRAM, "evaluation_ngrams": len(eval_grams),
        "sources": [{"name": s.name, "dataset": s.dataset, "config": s.config, "split": s.split, "licence": s.licence,
                     "commercial_ok": s.commercial_ok, "included": s in allowed, "rows_seen": counts[s.name],
                     "dropped_for_overlap": drops[s.name]} for s in sources],
        "noul_balance": balance,
        "splits": {"train": stats(train), "val": stats(val), "calibration": stats(cal)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log.info("wrote %s: train %d, val %d, calibration %d", out, len(train), len(val), len(cal))
    print(json.dumps({k: v for k, v in manifest["splits"]["train"].items() if k != "by_source"}, indent=1))


if __name__ == "__main__":
    main()
