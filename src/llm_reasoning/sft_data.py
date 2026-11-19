"""Prepare response-masked mathematical SFT examples from Orca Math and GSM8K."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from llm_reasoning.config import make_config
from llm_reasoning.data_prep import content_key, prepare_orca_rows, split_gsm8k, stable_bucket
from llm_reasoning.tokenizer import ByteBPETokenizer


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:/\d+)?(?![\w.])")
_GSM_FINAL = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*$")


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
    """Conservative heuristic: require multi-step text and a final numeric answer."""
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 20 or len(_NUMBER.findall(text)) < 2:
        return None
    tail = text[-180:]
    candidates = list(_NUMBER.finditer(tail))
    if not candidates:
        return None
    final = candidates[-1]
    # Avoid interpreting an intermediate number as a final answer.
    if len(tail[final.end():].strip(" .!\n")) > 35:
        return None
    return text.strip(), normalize_number(final.group())


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
    orca_rows: Iterable[Mapping[str, object]], *, tokenizer: ByteBPETokenizer,
    output_dir: Path, gsm_revision: str, orca_revision: str,
    validation_count: int = 500, orca_limit: int = 40_000,
    max_seq_len: int = 1_024, seed: int = 42,
) -> dict:
    if orca_limit <= 0 or max_seq_len <= 0:
        raise ValueError("orca_limit and max_seq_len must be positive")
    gsm = split_gsm8k(gsm_train, gsm_test, revision=gsm_revision, seed=seed,
                      validation_count=validation_count)
    orca, orca_manifest = prepare_orca_rows(orca_rows, revision=orca_revision,
                                            protected_question_keys=gsm.protected_question_keys, seed=seed)
    # Sort by seeded question hash; selection is independent of source order.
    orca.sort(key=lambda row: (stable_bucket(content_key(row["question"]), seed),
                               content_key(row["question"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "orca_train": prepare_split(orca, source="orca", tokenizer=tokenizer,
                                    max_seq_len=max_seq_len, output_path=output_dir / "orca_train.jsonl",
                                    gsm=False, accepted_limit=orca_limit),
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
                "orca_revision": orca_revision, "gsm_split": asdict(gsm.manifest),
                "orca_cleaning": asdict(orca_manifest), "orca_limit": orca_limit,
                "splits": splits, "test_examples": len(gsm.test)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare math SFT JSONL from Orca Math and GSM8K")
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orca-limit", type=int, default=40_000)
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
    orca_revision = api.repo_info(repo_id=config.data.orca_repo, repo_type="dataset").sha
    gsm_train = load_dataset(config.data.gsm8k_repo, "main", split="train", revision=gsm_revision)
    gsm_test = load_dataset(config.data.gsm8k_repo, "main", split="test", revision=gsm_revision)
    orca = load_dataset(config.data.orca_repo, split="train", revision=orca_revision)
    manifest = prepare_sft(gsm_train, gsm_test, orca, tokenizer=tokenizer, output_dir=args.output,
                           gsm_revision=gsm_revision, orca_revision=orca_revision,
                           validation_count=config.data.gsm8k_validation_examples,
                           orca_limit=args.orca_limit, max_seq_len=config.train.sft_seq_len,
                           seed=config.data.seed)
    manifest["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"splits": manifest["splits"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
