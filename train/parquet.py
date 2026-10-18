"""Whole splits from the Hugging Face parquet conversion, one download per shard, cached on disk.

The datasets-server rows API serves 100 rows per paced call, which makes a 300k-example mix a two-hour
fetch. The same server lists the parquet shards it is built from (``/parquet``), and those download in
seconds from huggingface.co. ``sample_rows`` reads only the row groups that hold the wanted rows, so a
multi-gigabyte source never has to fit in memory, and ``column_values`` lists the distinct values of a
string column over the whole split (intent lists). Everything lands under ``train/data/parquet/``.
"""
from __future__ import annotations

import json
import logging
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from eval.data import _get

log = logging.getLogger("train.parquet")
PARQUET_API = "https://datasets-server.huggingface.co/parquet"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "parquet"


def _slug(dataset: str, config: str, split: str) -> str:
    return f"{dataset.replace('/', '__')}__{config}__{split}"


def shards(dataset: str, config: str, split: str) -> list[Path]:
    """Local paths of the split's parquet shards, downloading what is missing (atomic: tmp file then rename)."""
    folder = CACHE_DIR / _slug(dataset, config, split)
    folder.mkdir(parents=True, exist_ok=True)
    listing = folder / "shards.json"
    if listing.exists():
        files = json.loads(listing.read_text())
    else:
        qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split})
        data = _get(f"{PARQUET_API}?{qs}")
        files = [{"url": f["url"], "filename": f["filename"], "size": f["size"]}
                 for f in data["parquet_files"] if f["config"] == config and f["split"] == split]
        if not files:
            raise FileNotFoundError(f"no parquet shards for {dataset}/{config}/{split}")
        listing.write_text(json.dumps(files))
    paths = []
    for f in sorted(files, key=lambda f: f["filename"]):
        path = folder / f["filename"]
        if not path.exists() or path.stat().st_size != f["size"]:
            tmp = path.with_suffix(".tmp")
            log.info("%s: downloading %s (%.1f MiB)", _slug(dataset, config, split), f["filename"], f["size"] / 2**20)
            with urllib.request.urlopen(f["url"], timeout=120) as r, tmp.open("wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            if tmp.stat().st_size != f["size"]:
                tmp.unlink()
                raise OSError(f"short download of {f['url']}: {tmp.stat().st_size} of {f['size']} bytes")
            tmp.replace(path)
        paths.append(path)
    return paths


def num_rows(paths: list[Path]) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def spread_indices(total: int, want: int) -> list[int]:
    """``want`` distinct row indices spread evenly over ``total`` rows (all rows when want >= total)."""
    want = min(want, total)
    if want <= 0:
        return []
    if want == total:
        return list(range(total))
    if want == 1:
        return [0]
    return sorted({int(i * (total - 1) / (want - 1)) for i in range(want)})


def take_rows(paths: list[Path], indices: list[int]) -> list[dict]:
    """Rows at global ``indices`` (sorted) across the shards, read one row group at a time."""
    rows: list[dict] = []
    base = 0
    pos = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            local = []
            while pos < len(indices) and indices[pos] < base + n:
                local.append(indices[pos] - base)
                pos += 1
            if local:
                rows.extend(pf.read_row_group(g).take(local).to_pylist())
            base += n
            if pos >= len(indices):
                return rows
    return rows


def sample_rows(dataset: str, config: str, split: str, want: int) -> tuple[list[dict], int]:
    """``want`` rows spread evenly over the split, and the split's row count."""
    paths = shards(dataset, config, split)
    total = num_rows(paths)
    return take_rows(paths, spread_indices(total, want)), total


def column_values(dataset: str, config: str, split: str, column: str) -> list[str]:
    """Sorted distinct values of a column over the whole split."""
    values: set[str] = set()
    for p in shards(dataset, config, split):
        for chunk in pq.ParquetFile(p).iter_batches(columns=[column]):
            values |= {str(v) for v in chunk.column(0).to_pylist()}
    return sorted(values)
