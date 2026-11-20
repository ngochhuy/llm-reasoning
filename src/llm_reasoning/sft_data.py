"""Prepare response-masked mathematical SFT examples from GSM8K only."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from llm_reasoning.config import make_config
from llm_reasoning.data_prep import content_key, split_gsm8k
from llm_reasoning.tokenizer import ByteBPETokenizer


_NUMBER = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
    r"(?![\w,]|\.\d)"
)
_GSM_FINAL = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class SplitStats:
    examined: int = 0
    accepted: int = 0
    invalid_answer: int = 0
    too_short: int = 0
    too_long: int = 0
    token_total: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    lengths: list[int] = field(default_factory=list, repr=False)

    def record(self, length: int) -> None:
        self.accepted += 1
        self.token_total += length
        self.min_tokens = min(self.min_tokens, length) if self.accepted > 1 else length
        self.max_tokens = max(self.max_tokens, length)
        self.lengths.append(length)

    def summary(self) -> dict:
        result = {key: value for key, value in asdict(self).items() if key != "lengths"}
        ordered = sorted(self.lengths)
        result["p50_tokens"] = ordered[len(ordered) // 2] if ordered else None
        result["p95_tokens"] = ordered[int((len(ordered) - 1) * 0.95)] if ordered else None
        return result


def normalize_number(value: str) -> str:
    return value.replace(",", "").strip()


def extract_gsm_answer(text: str) -> tuple[str, str] | None:
    match = _GSM_FINAL.search(text)
    if not match:
        return None
    solution = re.sub(r"<<[^<>]*>>", "", text[:match.start()]).strip()
    if not solution:
        return None
    return solution, normalize_number(match.group(1))


def extract_orca_answer(text: str) -> tuple[str, str] | None:
    """Accept only an unambiguous scalar in the final sentence.

    Orca has no structured answer field. Taking its last number often captures a
    repeated value from the question instead of the result, so ambiguous endings
    must be reviewed or discarded rather than silently mislabeled.
    """
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 20 or len(_NUMBER.findall(text)) < 2:
        return None
    final_paragraph = re.split(r"\n\s*\n+", text.strip())[-1]
    final_sentence = _SENTENCE_BOUNDARY.split(final_paragraph)[-1]
    if re.search(r"(?:π|\bpi\b)", final_sentence, flags=re.IGNORECASE):
        return None
    candidates = [match.group() for match in _NUMBER.finditer(final_sentence)]
    if len(candidates) != 1:
        return None
    return text.strip(), normalize_number(candidates[0])


def encode_example(tokenizer: ByteBPETokenizer, *, source: str, question: str,
                   solution: str, answer: str, max_seq_len: int) -> dict | None:
    prefix = f"Question: {question}\n<think>\n"
    response = f"{solution}\n</think>\n<answer>\n{answer}\n</answer><eos>"
    prefix_ids = tokenizer.encode(prefix, allowed_special=True)
    response_ids = tokenizer.encode(response, allowed_special=True)
    ids = prefix_ids + response_ids
    if len(ids) > max_seq_len:
        return None
    return {"source": source, "question_key": content_key(question), "input_ids": ids,
            "loss_mask": [0] * len(prefix_ids) + [1] * len(response_ids),
            "answer": answer, "token_count": len(ids)}


def prepare_split(rows: Iterable[Mapping[str, str]], *, source: str,
                  tokenizer: ByteBPETokenizer, max_seq_len: int,
                  output_path: Path, gsm: bool, accepted_limit: int | None = None) -> dict:
    stats = SplitStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            stats.examined += 1
            extracted = extract_gsm_answer(row["answer"]) if gsm else extract_orca_answer(row["answer"])
            if extracted is None:
                stats.invalid_answer += 1
                continue
            solution, answer = extracted
            if len(solution.split()) < 4:
                stats.too_short += 1
                continue
            encoded = encode_example(tokenizer, source=source, question=row["question"],
                                     solution=solution, answer=answer, max_seq_len=max_seq_len)
            if encoded is None:
                stats.too_long += 1
                continue
            output.write(json.dumps(encoded, ensure_ascii=False) + "\n")
            stats.record(encoded["token_count"])
            if accepted_limit is not None and stats.accepted >= accepted_limit:
                break
    return stats.summary()


def prepare_sft(
    gsm_train: Iterable[Mapping[str, object]], gsm_test: Iterable[Mapping[str, object]],
    *, tokenizer: ByteBPETokenizer,
    output_dir: Path, gsm_revision: str,
    validation_count: int = 500,
    max_seq_len: int = 1_024, seed: int = 42,
) -> dict:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if (output_dir / "orca_train.jsonl").exists():
        raise ValueError("Legacy Orca artifact found; choose a new GSM8K-only output directory")
    gsm = split_gsm8k(gsm_train, gsm_test, revision=gsm_revision, seed=seed,
                      validation_count=validation_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "gsm_train": prepare_split(gsm.train, source="gsm8k", tokenizer=tokenizer,
                                  max_seq_len=max_seq_len, output_path=output_dir / "gsm_train.jsonl", gsm=True),
        "gsm_validation": prepare_split(gsm.validation, source="gsm8k", tokenizer=tokenizer,
                                       max_seq_len=max_seq_len, output_path=output_dir / "gsm_validation.jsonl", gsm=True),
    }
    # Preserve the original held-out test for evaluation. It is never tokenized into SFT examples.
    with (output_dir / "gsm_test.jsonl").open("w", encoding="utf-8") as output:
        for row in gsm.test:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"seed": seed, "max_seq_len": max_seq_len, "gsm_revision": gsm_revision,
                "sft_sources": ["gsm8k"], "gsm_split": asdict(gsm.manifest),
                "splits": splits, "test_examples": len(gsm.test)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare math SFT JSONL from GSM8K only")
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the data group first: uv sync --group data") from exc
    config = make_config("target", runtime="kaggle")
    tokenizer_path = args.pretrain_dir / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    api = HfApi()
    gsm_revision = api.repo_info(repo_id=config.data.gsm8k_repo, repo_type="dataset").sha
    gsm_train = load_dataset(config.data.gsm8k_repo, "main", split="train", revision=gsm_revision)
    gsm_test = load_dataset(config.data.gsm8k_repo, "main", split="test", revision=gsm_revision)
    manifest = prepare_sft(gsm_train, gsm_test, tokenizer=tokenizer, output_dir=args.output,
                           gsm_revision=gsm_revision,
                           validation_count=config.data.gsm8k_validation_examples,
                           max_seq_len=config.train.sft_seq_len,
                           seed=config.data.seed)
    manifest["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"splits": manifest["splits"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
