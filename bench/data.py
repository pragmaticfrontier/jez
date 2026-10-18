"""Benchmark tasks, fetched from the Hugging Face datasets-server (no `datasets` dependency).

Each task maps a public dataset onto one of the three question primitives.
Rows are cached as JSONL under bench/data/.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(__file__).parent / "data"
ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass
class Task:
    name: str
    dataset: str
    config: str
    split: str
    qtype: str  # choice | bool | score
    state: Callable[[dict], Any]
    question: Callable[[dict], dict]
    gold: Callable[[dict], str]  # answer key: option name, "true"/"false", or level as string
    description: str = ""


SST2_CRITERIA = {
    "negative": "The reviewer expresses a negative opinion of the film",
    "positive": "The reviewer expresses a positive opinion of the film",
}
AGNEWS_LABELS = ["world", "sports", "business", "sci_tech"]
AGNEWS_CRITERIA = {
    "world": "World news: international affairs, politics, conflicts",
    "sports": "Sports: games, athletes, teams, results",
    "business": "Business: companies, markets, finance, economy",
    "sci_tech": "Science and technology: research, software, hardware, space",
}
YELP_RUBRIC = {"1": "terrible, would never return", "2": "poor, mostly negative", "3": "average or mixed",
               "4": "good, mostly positive", "5": "excellent, enthusiastic"}

TASKS: dict[str, Task] = {
    "sst2": Task(
        name="sst2", dataset="stanfordnlp/sst2", config="default", split="validation", qtype="choice",
        state=lambda r: r["sentence"].strip(),
        question=lambda r: {"type": "choice", "instructions": "What is the sentiment of this movie review sentence?",
                            "criteria": SST2_CRITERIA},
        gold=lambda r: ["negative", "positive"][int(r["label"])],
        description="SST-2 sentiment (2-way choice)",
    ),
    "agnews": Task(
        name="agnews", dataset="fancyzhx/ag_news", config="default", split="test", qtype="choice",
        state=lambda r: r["text"].strip(),
        question=lambda r: {"type": "choice", "instructions": "Which section of a newspaper does this article belong to?",
                            "criteria": AGNEWS_CRITERIA},
        gold=lambda r: AGNEWS_LABELS[int(r["label"])],
        description="AG News topic (4-way choice)",
    ),
    "boolq": Task(
        name="boolq", dataset="google/boolq", config="default", split="validation", qtype="bool",
        state=lambda r: r["passage"].strip(),
        question=lambda r: {"type": "bool", "instructions": r["question"].strip().rstrip("?").capitalize() + "?"},
        gold=lambda r: "true" if r["answer"] in (True, "True", "true") else "false",
        description="BoolQ reading comprehension (bool)",
    ),
    "yelp": Task(
        name="yelp", dataset="Yelp/yelp_review_full", config="yelp_review_full", split="test", qtype="score",
        state=lambda r: r["text"].strip(),
        question=lambda r: {"type": "score", "instructions": "How many stars did the reviewer give this business?",
                            "min": 1, "max": 5, "criteria": YELP_RUBRIC},
        gold=lambda r: str(int(r["label"]) + 1),
        description="Yelp review stars (score 1-5)",
    ),
}


def _fetch(dataset: str, config: str, split: str, offset: int, length: int, retries: int = 5) -> tuple[list[dict], int]:
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ROWS_API}?{qs}", timeout=60) as r:
                data = json.load(r)
            return [row["row"] for row in data["rows"]], int(data["num_rows_total"])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:  # datasets-server returns 502s now and then
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_rows(task: Task, n: int, chunk: int = 100) -> list[dict]:
    """Return n rows, taken as `n // chunk` chunks spread evenly across the split (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{task.name}.jsonl"
    rows: list[dict] = []
    if cache.exists():
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    if len(rows) >= n:
        return rows[:n]

    first, total = _fetch(task.dataset, task.config, task.split, 0, min(chunk, n))
    n_chunks = max(1, -(-n // chunk))
    offsets = [int(i * (total - chunk) / max(1, n_chunks - 1)) for i in range(n_chunks)] if n_chunks > 1 else [0]
    rows = []
    for i, off in enumerate(offsets):
        got = first if i == 0 else _fetch(task.dataset, task.config, task.split, off, min(chunk, n - len(rows)))[0]
        rows.extend(got)
        if len(rows) >= n:
            break
    rows = rows[:n]
    cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows
