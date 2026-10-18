"""Milestone 1 label check: 255 single-token choice labels, 10 digit labels and Yes/No per tokenizer."""
from __future__ import annotations

import pytest

from jez.labels import MAX_CHOICE, build_label_table, candidate_labels, check_label_table

TOKENIZERS = ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"]


def _cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_id, "tokenizer_config.json"), str)


def test_candidate_order_is_letters_then_pairs():
    labs = candidate_labels(MAX_CHOICE)
    assert labs[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert labs[26:29] == ["AA", "AB", "AC"]
    assert len(labs) == 255 and len(set(labs)) == 255


@pytest.mark.parametrize("model_id", TOKENIZERS)
def test_real_tokenizers_supply_255_single_token_labels(model_id):
    if not _cached(model_id):
        pytest.skip(f"{model_id} tokenizer is not in the Hugging Face cache")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    table = build_label_table(tok, name=model_id)
    assert check_label_table(table) == []
    assert table.n_single_token == 255
    assert [lab.text for lab in table.score] == list("0123456789")
    assert all(not lab.multi_token for lab in table.score)
    assert [lab.text for lab in table.noul] == ["Yes", "No"]
    ids = [i for lab in table.choice for i in lab.ids]
    assert len(ids) == len(set(ids)), "two labels share a token id"


class _TwoLetterFreeTokenizer:
    """Only single characters are single tokens, so two-letter labels must be teacher forced."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text.strip()] if text.strip() else []


def test_multi_token_fallback_is_marked():
    table = build_label_table(_TwoLetterFreeTokenizer())
    assert len(table.choice) == 255
    assert table.n_single_token == 26
    assert all(lab.multi_token for lab in table.choice[26:])
    assert table.choice[26].text == "AA" and table.choice[26].ids == (65, 65)
    problems = check_label_table(table)
    assert any("teacher forcing" in p for p in problems)
