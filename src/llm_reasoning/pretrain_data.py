"""Prepare BPE pretraining shards from streamed text datasets."""

import argparse
import hashlib
import json
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from llm_reasoning.config import make_config
from llm_reasoning.data_prep import SplitManifest, prepare_pretrain_documents
from llm_reasoning.tokenizer import ByteBPETokenizer


@dataclass
class SourceStats:
    source: str
    revision: str
    token_target: int
    train_tokens: int = 0
    validation_tokens: int = 0
    documents: int = 0


class BlockWriter:
    """Write fixed-length uint16 token blocks to bounded binary shards."""

    def __init__(self, directory: Path, block_size: int = 512, blocks_per_shard: int = 2048) -> None:
        if block_size <= 0 or blocks_per_shard <= 0:
            raise ValueError("block_size and blocks_per_shard must be positive")
        self.directory = directory
        self.block_size = block_size
        self.blocks_per_shard = blocks_per_shard
        self.pending = array("H")
        self.shard = array("H")
        self.paths: list[str] = []
        self.written_tokens = 0
        self.dropped_tail_tokens = 0
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, ids: Iterable[int]) -> None:
        for token in ids:
            if not 0 <= token <= 65535:
                raise ValueError("uint16 shard cannot store token ID outside 0..65535")
            self.pending.append(token)
            if len(self.pending) == self.block_size:
                self.shard.extend(self.pending)
                self.pending = array("H")
                self.written_tokens += self.block_size
                if len(self.shard) >= self.block_size * self.blocks_per_shard:
                    self._flush()

    def _flush(self) -> None:
        if not self.shard:
            return
        path = self.directory / f"shard_{len(self.paths):05d}.bin"
        with path.open("wb") as output:
            self.shard.tofile(output)
        self.paths.append(path.name)
        self.shard = array("H")

    def close(self) -> None:
        self.dropped_tail_tokens = len(self.pending)
        self.pending = array("H")
        self._flush()


def train_tokenizer(texts: Iterable[Mapping[str, object]], tokenizer: ByteBPETokenizer,
                    max_training_bytes: int) -> int:
    """Train only on rows classified as train by the deterministic split."""
    return tokenizer.train((row["text"] for row in texts if row.get("split") == "train"),
                           max_training_bytes=max_training_bytes)


def interleave_sources(first: Iterable[Mapping[str, object]], second: Iterable[Mapping[str, object]]):
    """Sample four education rows then one math row while both streams last."""
    streams = (iter(first), iter(second))
    active = [True, True]
    while any(active):
        for index in (0, 0, 0, 0, 1):
            if active[index]:
                try:
                    yield next(streams[index])
                except StopIteration:
                    active[index] = False


def write_source(
    rows: Iterable[Mapping[str, object]], *, source: str, revision: str, token_target: int,
    tokenizer: ByteBPETokenizer, output_dir: Path, seed: int = 42,
    validation_fraction: float = 0.01, block_size: int = 512,
    blocks_per_shard: int = 2048, seen_keys: set[str] | None = None,
) -> dict:
    if token_target <= 0:
        raise ValueError("token_target must be positive")
    manifest = SplitManifest(source, revision, seed)
    stats = SourceStats(source, revision, token_target)
    writers = {split: BlockWriter(output_dir / source / split, block_size, blocks_per_shard)
               for split in ("train", "validation")}
    eos = tokenizer.special_to_id["<eos>"]
    for doc in prepare_pretrain_documents(rows, source=source, revision=revision, seed=seed,
                                          validation_fraction=validation_fraction,
                                          seen_keys=seen_keys, manifest=manifest):
        ids = tokenizer.encode(doc.text)
        if doc.split == "train":
            remaining = token_target - stats.train_tokens
            if remaining <= 0:
                break
            # A partial final document still terminates at EOS.
            ids = ids[:max(0, remaining - 1)] + [eos]
            stats.train_tokens += len(ids)
        else:
            ids.append(eos)
            stats.validation_tokens += len(ids)
        writers[doc.split].add(ids)
        stats.documents += 1
        if stats.train_tokens >= token_target:
            break
    for writer in writers.values():
        writer.close()
    result = {"source": asdict(stats), "cleaning": asdict(manifest),
              "shards": {split: {"files": writer.paths, "written_tokens": writer.written_tokens,
                                 "dropped_tail_tokens": writer.dropped_tail_tokens}
                         for split, writer in writers.items()}}
    (output_dir / source / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream datasets and prepare pretraining uint16 shards")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tokenizer-bytes", type=int, default=2_000_000)
    parser.add_argument("--fineweb-tokens", type=int, default=None)
    parser.add_argument("--math-tokens", type=int, default=None)
    parser.add_argument("--blocks-per-shard", type=int, default=2048)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the data group first: uv sync --group data") from exc

    config = make_config("target", runtime="kaggle", artifact_root=args.output)
    root = config.paths.data / "pretrain"
    root.mkdir(parents=True, exist_ok=True)
    sources = (
        ("fineweb", config.data.fineweb_repo, "sample-10BT", args.fineweb_tokens or config.data.fineweb_tokens),
        ("math", config.data.math_repo, None, args.math_tokens or config.data.math_tokens),
    )
    api = HfApi()
    revisions = {label: api.repo_info(repo_id=repo, repo_type="dataset").sha for label, repo, _, _ in sources}

    def stream(repo: str, subset: str | None, revision: str):
        return load_dataset(repo, name=subset, split="train", revision=revision, streaming=True)

    # Use deterministic document hashes to exclude validation rows from tokenizer training.
    from llm_reasoning.data_prep import prepare_pretrain_documents
    training_streams = []
    for label, repo, subset, _ in sources:
        docs = prepare_pretrain_documents(stream(repo, subset, revisions[label]), source=label,
                                          revision=revisions[label], seed=config.data.seed,
                                          validation_fraction=config.data.pretrain_validation_fraction)
        training_streams.append(({"text": doc.text, "split": doc.split} for doc in docs))
    tokenizer = ByteBPETokenizer(config.tokenizer)
    actual_vocab = train_tokenizer(interleave_sources(*training_streams), tokenizer, args.tokenizer_bytes)
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    seen: set[str] = set()
    manifests = {}
    for label, repo, subset, target in sources:
        manifests[label] = write_source(stream(repo, subset, revisions[label]), source=label,
                                        revision=revisions[label], token_target=target,
                                        tokenizer=tokenizer, output_dir=root, seed=config.data.seed,
                                        validation_fraction=config.data.pretrain_validation_fraction,
                                        block_size=config.train.pretrain_seq_len,
                                        blocks_per_shard=args.blocks_per_shard, seen_keys=seen)
    summary = {"tokenizer_file": tokenizer_path.name,
               "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
               "vocab_size": actual_vocab, "block_size": config.train.pretrain_seq_len,
               "token_dtype": "uint16", "byte_order": sys.byteorder,
               "sources": manifests}
    (root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"vocab_size": actual_vocab,
                      "train_tokens": {key: value["source"]["train_tokens"] for key, value in manifests.items()},
                      "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
