import json

import pytest

from llm_reasoning.config import TokenizerConfig
from llm_reasoning.tokenizer import ByteBPETokenizer


def test_round_trip_unseen_unicode_math_and_specials(tmp_path) -> None:
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=300))
    tokenizer.train(["Solve 12 + 30 = 42.\n" for _ in range(30)])
    text = "Tiếng Việt 🧮\n-1,234.50 = x² <think>literal</think>"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.special_to_id["<think>"] not in ids
    tagged = "<think>12 + 30</think><answer>42</answer><eos>"
    special_ids = tokenizer.encode(tagged, allowed_special=True)
    assert special_ids[0] == tokenizer.special_to_id["<think>"]
    assert tokenizer.decode(special_ids) == tagged
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.encode(text) == ids
    assert loaded.encode(tagged, allowed_special=True) == special_ids


def test_training_deterministic_and_reduces_token_count() -> None:
    corpus = ["banana banana banana", "banana bandana"]
    a = ByteBPETokenizer(TokenizerConfig(vocab_size=280))
    b = ByteBPETokenizer(TokenizerConfig(vocab_size=280))
    before = len(a.encode(corpus[0]))
    assert a.train(corpus) == b.train(corpus)
    assert a.merges == b.merges
    assert len(a.encode(corpus[0])) < before
    assert a.vocab_size <= 280


def test_invalid_training_and_corrupt_file(tmp_path) -> None:
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=270))
    with pytest.raises(ValueError, match="no usable text"):
        tokenizer.train([""])
    with pytest.raises(ValueError, match="unknown token"):
        tokenizer.decode([9999])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format_version": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        ByteBPETokenizer.load(path)
