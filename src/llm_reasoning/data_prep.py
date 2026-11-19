"""Deterministic cleaning and splits before tokenizer training or packing."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping


def normalize_text(value: str) -> str:
    """Preserve math punctuation and line boundaries while normalizing spacing."""
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def content_key(value: str) -> str:
    canonical = re.sub(r"\s+", " ", normalize_text(value)).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass
class SplitManifest:
    source: str
    revision: str
    seed: int
    examined: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    dropped_protected: int = 0
    notes: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CleanDocument:
    source: str
    text: str
    key: str
    split: str


def prepare_pretrain_documents(
    rows: Iterable[Mapping[str, object]], *, source: str, revision: str,
    seed: int = 42, validation_fraction: float = 0.01,
    seen_keys: set[str] | None = None,
    manifest: SplitManifest | None = None,
) -> Iterator[CleanDocument]:
    """Stream clean documents; the same seen_keys can deduplicate across sources."""
    if not source or not revision:
        raise ValueError("source and revision are required for reproducibility")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if manifest is None:
        manifest = SplitManifest(source, revision, seed)
    elif (manifest.source, manifest.revision, manifest.seed) != (source, revision, seed):
        raise ValueError("manifest metadata does not match the input")
    seen = seen_keys if seen_keys is not None else set()
    for row in rows:
        manifest.examined += 1
        raw = row.get("text")
        text = normalize_text(raw) if isinstance(raw, str) else ""
        if not text:
            manifest.dropped_empty += 1
            continue
        key = content_key(text)
        if key in seen:
            manifest.dropped_duplicate += 1
            continue
        seen.add(key)
        split = "validation" if stable_bucket(key, seed) < validation_fraction else "train"
        setattr(manifest, split, getattr(manifest, split) + 1)
        yield CleanDocument(source, text, key, split)


@dataclass(frozen=True)
class GSM8KSplit:
    train: tuple[dict[str, str], ...]
    validation: tuple[dict[str, str], ...]
    test: tuple[dict[str, str], ...]
    protected_question_keys: frozenset[str]
    manifest: SplitManifest


def split_gsm8k(
    train_rows: Iterable[Mapping[str, object]], test_rows: Iterable[Mapping[str, object]], *,
    revision: str, seed: int = 42, validation_count: int = 500,
) -> GSM8KSplit:
    """Reserve exactly validation_count unique train questions; keep test separate."""
    if not revision or validation_count <= 0:
        raise ValueError("revision and positive validation_count are required")
    manifest = SplitManifest("openai/gsm8k", revision, seed)
    cleaned: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    seen: set[str] = set()
    for split_name, rows, output in (("train", train_rows, cleaned), ("test", test_rows, test)):
        for row in rows:
            manifest.examined += 1
            question = normalize_text(row["question"]) if isinstance(row.get("question"), str) else ""
            answer = normalize_text(row["answer"]) if isinstance(row.get("answer"), str) else ""
            if not question or not answer:
                manifest.dropped_empty += 1
                continue
            key = content_key(question)
            if key in seen:
                manifest.dropped_duplicate += 1
                if split_name == "test":
                    raise ValueError("GSM8K test question overlaps train or another test row")
                continue
            seen.add(key)
            output.append({"question": question, "answer": answer})
    if len(cleaned) <= validation_count:
        raise ValueError("not enough unique GSM8K train questions for validation split")
    ranked = sorted(cleaned, key=lambda row: (stable_bucket(content_key(row["question"]), seed), content_key(row["question"])))
    validation = ranked[:validation_count]
    train = ranked[validation_count:]
    manifest.train = len(train)
    manifest.validation = len(validation)
    manifest.test = len(test)
    manifest.notes["split_method"] = "question-hash ranking; lowest scores assigned to validation"
    return GSM8KSplit(tuple(train), tuple(validation), tuple(test), frozenset(seen), manifest)


def prepare_orca_rows(
    rows: Iterable[Mapping[str, object]], *, revision: str,
    protected_question_keys: frozenset[str], seed: int = 42,
) -> tuple[list[dict[str, str]], SplitManifest]:
    """Clean Orca questions and exclude all GSM8K question matches."""
    if not revision:
        raise ValueError("revision is required")
    manifest = SplitManifest("microsoft/orca-math-word-problems-200k", revision, seed)
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        manifest.examined += 1
        question = normalize_text(row["question"]) if isinstance(row.get("question"), str) else ""
        answer = normalize_text(row["answer"]) if isinstance(row.get("answer"), str) else ""
        if not question or not answer:
            manifest.dropped_empty += 1
            continue
        key = content_key(question)
        if key in protected_question_keys:
            manifest.dropped_protected += 1
            continue
        if key in seen:
            manifest.dropped_duplicate += 1
            continue
        seen.add(key)
        output.append({"question": question, "answer": answer})
    manifest.train = len(output)
    manifest.notes["status"] = "cleaned only; CoT quality and answer extraction are task 1.7"
    return output, manifest
