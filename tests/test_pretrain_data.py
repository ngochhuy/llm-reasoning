from array import array

from llm_reasoning.config import TokenizerConfig
from llm_reasoning.pretrain_data import interleave_sources, train_tokenizer, write_source
from llm_reasoning.tokenizer import ByteBPETokenizer


def test_training_sample_includes_both_sources() -> None:
    first = ({"text": f"education {i}", "split": "train"} for i in range(8))
    second = ({"text": f"math {i}", "split": "train"} for i in range(2))
    mixed = list(interleave_sources(first, second))
    assert [row["text"] for row in mixed[:5]] == ["education 0", "education 1", "education 2",
                                                   "education 3", "math 0"]
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=300))
    assert train_tokenizer(mixed, tokenizer, 1000) > 262


def test_shards_are_fixed_blocks_with_source_budget_and_eos(tmp_path) -> None:
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=300))
    tokenizer.train(["one two three four five"] * 10)
    result = write_source(({"text": f"doc {i} has content"} for i in range(30)),
                          source="fineweb", revision="rev", token_target=64,
                          tokenizer=tokenizer, output_dir=tmp_path,
                          validation_fraction=0.01, block_size=8, blocks_per_shard=2)
    assert result["source"]["train_tokens"] == 64
    assert result["shards"]["train"]["written_tokens"] == 64
    paths = sorted((tmp_path / "fineweb" / "train").glob("*.bin"))
    assert len(paths) == 4
    tokens = array("H")
    for path in paths:
        with path.open("rb") as handle:
            tokens.fromfile(handle, path.stat().st_size // 2)
    assert len(tokens) == 64
    assert tokenizer.special_to_id["<eos>"] in tokens
