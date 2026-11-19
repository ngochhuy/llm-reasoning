import json

import pytest

from llm_reasoning.data_prep import (
    SplitManifest,
    content_key,
    normalize_text,
    prepare_orca_rows,
    prepare_pretrain_documents,
    split_gsm8k,
)


def test_pretrain_split_is_stable_and_deduplicates_across_sources(tmp_path) -> None:
    text = "  x² + 2 = 6\r\n\r\n Solve  x.  "
    assert normalize_text(text) == "x² + 2 = 6\n\nSolve x."
    assert content_key(text) == content_key("X² + 2 = 6 Solve x.")
    rows = [{"text": text}, {"text": "x² + 2 = 6 Solve x."}, {"text": " "}, {"text": "Different proof."}]
    seen: set[str] = set()
    manifest = SplitManifest("source-a", "rev-1", 42)
    first = list(prepare_pretrain_documents(rows, source="source-a", revision="rev-1", seen_keys=seen, manifest=manifest))
    second = list(prepare_pretrain_documents([{"text": text}], source="source-b", revision="rev-2", seen_keys=seen))
    again = list(prepare_pretrain_documents(reversed(rows), source="source-a", revision="rev-1"))
    assert len(first) == len(again) == 2 and second == []
    assert {row.key: row.split for row in first} == {row.key: row.split for row in again}
    assert (manifest.examined, manifest.dropped_duplicate, manifest.dropped_empty) == (4, 1, 1)
    manifest.save(tmp_path / "manifest.json")
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["revision"] == "rev-1"


def test_gsm_validation_and_orca_protection() -> None:
    train = [{"question": f"Question {n}?", "answer": f"Work. #### {n}"} for n in range(8)]
    test = [{"question": "Held out?", "answer": "Work. #### 9"}]
    split = split_gsm8k(train, test, revision="gsm-rev", validation_count=3)
    assert (len(split.train), len(split.validation), len(split.test)) == (5, 3, 1)
    reordered = split_gsm8k(reversed(train), test, revision="gsm-rev", validation_count=3)
    assert {row["question"] for row in split.validation} == {row["question"] for row in reordered.validation}
    orca, manifest = prepare_orca_rows(
        [{"question": "Held out?", "answer": "Leak"}, {"question": " Question 0? ", "answer": "Leak"},
         {"question": "New?", "answer": "Solve 2+2=4"}, {"question": "New?", "answer": "Duplicate"}],
        revision="orca-rev", protected_question_keys=split.protected_question_keys,
    )
    assert orca == [{"question": "New?", "answer": "Solve 2+2=4"}]
    assert (manifest.dropped_protected, manifest.dropped_duplicate) == (2, 1)


def test_gsm_test_overlap_is_explicit_error() -> None:
    rows = [{"question": "Same?", "answer": "#### 1"}, {"question": "Other?", "answer": "#### 2"}]
    with pytest.raises(ValueError, match="overlaps"):
        split_gsm8k(rows, [rows[0]], revision="rev", validation_count=1)
