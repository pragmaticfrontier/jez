"""Evaluation data cache: sampling is reproducible and records with line-separator characters survive."""
from __future__ import annotations

import json

import pytest

from eval import data


def test_chunk_sample_matches_chunked_download_offsets():
    rows = [{"i": i} for i in range(5083)]
    sample = data.chunk_sample(rows, 1000)
    assert len(sample) == 1000
    offsets = data.chunk_offsets(1000, 5083)
    expected = [i for off in offsets for i in range(off, off + data.PAGE)][:1000]
    assert [r["i"] for r in sample] == expected
    assert data.chunk_sample(rows, 10000) is rows


def test_cached_rows_keep_unicode_line_separators(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    rows = [{"text": "first part"}, {"text": "second\x85part"}, {"text": "plain"}]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    assert data.cached_rows("t") == rows


def test_load_rows_serves_samples_from_a_full_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "fetch_rows", lambda *a, **k: pytest.fail("network call on a full cache"))
    rows = [{"i": i, "text": "x y"} for i in range(1234)]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (tmp_path / "t.meta.json").write_text(json.dumps({"full": True, "n": 1234, "total": 1234}))
    got = data.load_rows("t", "d", "c", "s", 300)
    assert len(got) == 300 and got[0]["i"] == 0 and got[-1]["i"] == 1233
    assert data.load_rows("t", "d", "c", "s", None) == rows
