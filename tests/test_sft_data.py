import json

import pytest

from llm_reasoning.config import TokenizerConfig
from llm_reasoning.sft_data import extract_gsm_answer, extract_orca_answer, prepare_sft
from llm_reasoning.tokenizer import ByteBPETokenizer


def test_answer_conversion_and_masked_response(tmp_path) -> None:
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=320))
    tokenizer.train(["Question: How many apples? We add 2 + 3 = 5. Answer 5."] * 20)
    gsm_train = [
        {"question": f"How many apples in case {i}?",
         "answer": "We have 2 apples and add 3. 2 + 3 = <<2+3=5>>5. #### 5"}
        for i in range(4)
    ]
    gsm_test = [{"question": "Held out apples?", "answer": "Add 2 and 2. #### 4"}]
    manifest = prepare_sft(gsm_train, gsm_test, tokenizer=tokenizer,
                           output_dir=tmp_path, gsm_revision="g1",
                           validation_count=1, max_seq_len=512)
    assert manifest["sft_sources"] == ["gsm8k"]
    assert set(manifest["splits"]) == {"gsm_train", "gsm_validation"}
    assert not (tmp_path / "orca_train.jsonl").exists()
    example = json.loads((tmp_path / "gsm_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(example["input_ids"]) == len(example["loss_mask"])
    assert example["loss_mask"][0] == 0 and example["loss_mask"][-1] == 1
    decoded = tokenizer.decode(example["input_ids"])
    assert decoded.endswith("</answer><eos>")
    assert example["answer"] == "5"
    assert len((tmp_path / "gsm_test.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert "input_ids" not in (tmp_path / "gsm_test.jsonl").read_text(encoding="utf-8")


def test_rejects_unreliable_answer_and_overlong_without_truncating(tmp_path) -> None:
    assert extract_gsm_answer("Step. #### unknown") is None
    assert extract_gsm_answer("2 + 3 = <<2+3=5>>5. #### 5") == ("2 + 3 = 5.", "5")
    assert extract_orca_answer("The answer is 4.") is None
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=270))
    gsm_train = [{"question": f"Question {i}?", "answer": "Count one, then another. #### 2"}
                 for i in range(3)]
    result = prepare_sft(gsm_train, [], tokenizer=tokenizer,
                         output_dir=tmp_path, gsm_revision="g",
                         validation_count=1, max_seq_len=10)
    assert result["splits"]["gsm_train"]["too_long"] == 2
    assert result["splits"]["gsm_train"]["accepted"] == 0


@pytest.mark.parametrize(("ending", "expected"), [
    ("Therefore, the total profit is Rs 42,000.", "42000"),
    ("Therefore, the compound interest is approximately Rs. 3,981.85.", "3981.85"),
    ("So, Tom has 9 balloons.", "9"),
    ("So, it takes Emily 5 days to walk 230 meters farther.", None),
    ("The remainder of 3^19 when divided by 10 is 7.", None),
    ("The answer is 1:25.", None),
    ("The exact area is π * 156.8.", None),
])
def test_orca_final_answer_requires_one_scalar(ending: str, expected: str | None) -> None:
    solution = "First we examine the quantities in the question. " \
               "Then we calculate 2 plus 3 and verify the result. " + ending
    extracted = extract_orca_answer(solution)
    assert (extracted[1] if extracted else None) == expected


def test_refuses_legacy_mixed_artifact_directory(tmp_path) -> None:
    (tmp_path / "orca_train.jsonl").write_text("legacy", encoding="utf-8")
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=270))
    with pytest.raises(ValueError, match="Legacy Orca"):
        prepare_sft([], [], tokenizer=tokenizer, output_dir=tmp_path, gsm_revision="g")
