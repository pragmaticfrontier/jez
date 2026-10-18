"""Rows from the Hugging Face datasets-server, cached as JSONL under eval/data/.

No ``datasets`` dependency: the rows API serves 100 rows per call. A sample of n rows is
taken as chunks spread evenly across the split, so it is not the first n rows. The full
split is fetched when n is None.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data"
ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = "https://datasets-server.huggingface.co/info"
PAGE = 100


_PACE = 0.25  # seconds between calls; the datasets-server rate-limits bursts with 429


def _get(url: str, retries: int = 8) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(_PACE)
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if attempt == retries - 1:
                raise
            if e.code == 429:
                wait = float(e.headers.get("Retry-After", 0) or 0) or 10.0 * (attempt + 1)
                time.sleep(min(wait, 120))
            elif e.code in (500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> tuple[list[dict], int]:
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    data = _get(f"{ROWS_API}?{qs}")
    return [row["row"] for row in data["rows"]], int(data["num_rows_total"])


def label_names(dataset: str, config: str, feature: str = "label") -> list[str]:
    """ClassLabel names of a feature, from the datasets-server info endpoint (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"labels-{dataset.replace('/', '__')}-{config}-{feature}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    info = _get(f"{INFO_API}?{urllib.parse.urlencode({'dataset': dataset, 'config': config})}")
    names = info["dataset_info"]["features"][feature]["names"]
    cache.write_text(json.dumps(names))
    return names


def chunk_offsets(n: int, total: int) -> list[int]:
    """Start offsets of the PAGE-row chunks that make up a sample of n rows spread across total rows."""
    n_chunks = max(1, -(-n // PAGE))
    if n_chunks == 1:
        return [0]
    return [int(i * (total - PAGE) / (n_chunks - 1)) for i in range(n_chunks)]


def chunk_sample(rows: list[dict], n: int) -> list[dict]:
    """The same n-row sample the chunked download takes, cut from a full split held in memory."""
    if n >= len(rows):
        return rows
    out: list[dict] = []
    for off in chunk_offsets(n, len(rows)):
        out.extend(rows[off:off + PAGE])
        if len(out) >= n:
            break
    return out[:n]


def load_rows(name: str, dataset: str, config: str, split: str, n: int | None) -> list[dict]:
    """n rows of a split (all rows when n is None), spread across the split and cached.

    A sample is the same whether it is downloaded chunk by chunk or cut from a cached full split.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}.jsonl"
    meta = CACHE_DIR / f"{name}.meta.json"
    if cache.exists() and meta.exists():
        info = json.loads(meta.read_text())
        rows = [json.loads(line) for line in cache.read_text().split("\n") if line.strip()]
        if info.get("full"):
            return rows if n is None else chunk_sample(rows, n)
        if n is not None and len(rows) >= n and info.get("n") == len(rows):
            return rows[:n]

    first, total = fetch_rows(dataset, config, split, 0, PAGE)
    if n is None or n >= total:
        rows = list(first)
        offset = PAGE
        while len(rows) < total:
            got, _ = fetch_rows(dataset, config, split, offset, PAGE)
            if not got:
                break
            rows.extend(got)
            offset += PAGE
        full = True
    else:
        offsets = chunk_offsets(n, total)
        rows = []
        for i, off in enumerate(offsets):
            got = first if i == 0 else fetch_rows(dataset, config, split, off, PAGE)[0]
            rows.extend(got)
            if len(rows) >= n:
                break
        rows = rows[:n]
        full = False
    tmp = cache.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    tmp.replace(cache)  # atomic: readers never see a half-written file
    meta.write_text(json.dumps({"dataset": dataset, "config": config, "split": split, "total": total, "full": full,
                                "n": len(rows)}))
    return rows


def cached_rows(name: str) -> list[dict]:
    """Whatever sample of a task is cached locally, without touching the network."""
    cache = CACHE_DIR / f"{name}.jsonl"
    if not cache.exists():
        return []
    return [json.loads(line) for line in cache.read_text().split("\n") if line.strip()]
