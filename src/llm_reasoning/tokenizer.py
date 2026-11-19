"""Small byte-level BPE tokenizer implemented without a tokenizer library."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from llm_reasoning.config import TokenizerConfig


_PRETOKEN = re.compile(r"\s+|[^\W\d_]+|\d+|[^\w\s]+|_", re.UNICODE)


class ByteBPETokenizer:
    """Train byte pair merges on a bounded text sample and encode any UTF-8 text."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.special_to_id = {token: 256 + i for i, token in enumerate(self.config.special_tokens)}
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[int, int]] = []
        self.merge_to_id: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_to_id) + len(self.merges)

    def train(self, texts: Iterable[str], *, max_training_bytes: int = 2_000_000,
              min_pair_frequency: int = 2) -> int:
        """Fit merges from a bounded UTF-8 sample; return actual vocabulary size."""
        if max_training_bytes <= 0 or min_pair_frequency < 2:
            raise ValueError("max_training_bytes must be positive and min_pair_frequency >= 2")
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_to_id = {}
        words: Counter[tuple[int, ...]] = Counter()
        used_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("training texts must be strings")
            for piece in _PRETOKEN.findall(text):
                encoded = piece.encode("utf-8")
                if used_bytes + len(encoded) > max_training_bytes:
                    break
                words[tuple(encoded)] += 1
                used_bytes += len(encoded)
            if used_bytes >= max_training_bytes:
                break
        if not words:
            raise ValueError("training sample has no usable text")
        while self.vocab_size < self.config.vocab_size:
            pairs: Counter[tuple[int, int]] = Counter()
            for word, freq in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += freq
            if not pairs:
                break
            pair, count = min(pairs.items(), key=lambda item: (-item[1], item[0]))
            if count < min_pair_frequency:
                break
            new_id = self.vocab_size
            self.merges.append(pair)
            self.merge_to_id[pair] = new_id
            self.id_to_bytes[new_id] = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            updated: Counter[tuple[int, ...]] = Counter()
            for word, freq in words.items():
                merged: list[int] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                updated[tuple(merged)] += freq
            words = updated
        return self.vocab_size

    def _encode_piece(self, piece: str) -> list[int]:
        ids = list(piece.encode("utf-8"))
        while len(ids) > 1:
            ranked = [(self.merge_to_id[(a, b)], i) for i, (a, b) in enumerate(zip(ids, ids[1:]))
                      if (a, b) in self.merge_to_id]
            if not ranked:
                break
            new_id, index = min(ranked)
            ids[index:index + 2] = [new_id]
        return ids

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        """Encode text; special tokens are recognized only when explicitly allowed."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        segments: list[str] = [text]
        if allowed_special:
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.special_to_id, key=len, reverse=True)) + ")"
            segments = re.split(pattern, text)
        result: list[int] = []
        for segment in segments:
            if allowed_special and segment in self.special_to_id:
                result.append(self.special_to_id[segment])
            else:
                for piece in _PRETOKEN.findall(segment):
                    result.extend(self._encode_piece(piece))
        return result

    def decode(self, ids: Iterable[int]) -> str:
        buffer = bytearray()
        output: list[str] = []
        specials = {value: key for key, value in self.special_to_id.items()}
        for token_id in ids:
            if token_id in specials:
                output.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()
                output.append(specials[token_id])
            elif token_id in self.id_to_bytes:
                buffer.extend(self.id_to_bytes[token_id])
            else:
                raise ValueError(f"unknown token ID: {token_id}")
        output.append(buffer.decode("utf-8", errors="replace"))
        return "".join(output)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "vocab_size": self.config.vocab_size,
                   "special_tokens": list(self.config.special_tokens), "merges": self.merges}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported tokenizer format")
        config = TokenizerConfig(vocab_size=payload["vocab_size"],
                                 special_tokens=tuple(payload["special_tokens"]))
        tokenizer = cls(config)
        if len(payload["merges"]) > config.vocab_size - 256 - len(config.special_tokens):
            raise ValueError("too many merges for configured vocabulary")
        for raw_pair in payload["merges"]:
            if not isinstance(raw_pair, list) or len(raw_pair) != 2 or any(not isinstance(i, int) for i in raw_pair):
                raise ValueError("invalid merge pair")
            pair = tuple(raw_pair)
            if pair in tokenizer.merge_to_id or any(i not in tokenizer.id_to_bytes for i in pair):
                raise ValueError("duplicate merge or unknown parent token")
            new_id = tokenizer.vocab_size
            tokenizer.merges.append(pair)
            tokenizer.merge_to_id[pair] = new_id
            tokenizer.id_to_bytes[new_id] = tokenizer.id_to_bytes[pair[0]] + tokenizer.id_to_bytes[pair[1]]
        return tokenizer
