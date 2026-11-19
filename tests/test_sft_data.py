import json

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
    long_solution = "First we count 2 apples. Next we add 3 more apples. Finally 2 + 3 = 5, so the answer is 5."
    orca = [
        {"question": "Held out apples?", "answer": long_solution},
        {"question": "Fresh apples?", "answer": long_solution},
        {"question": "Short?", "answer": "The answer is 4."},
    ]
    manifest = prepare_sft(gsm_train, gsm_test, orca, tokenizer=tokenizer,
                           output_dir=tmp_path, gsm_revision="g1", orca_revision="o1",
                           validation_count=1, orca_limit=1, max_seq_len=512)
    assert manifest["orca_cleaning"]["dropped_protected"] == 1
    assert manifest["splits"]["orca_train"]["accepted"] == 1
    example = json.loads((tmp_path / "orca_train.jsonl").read_text(encoding="utf-8"))
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
    result = prepare_sft(gsm_train, [], [], tokenizer=tokenizer,
                         output_dir=tmp_path, gsm_revision="g", orca_revision="o",
                         validation_count=1, max_seq_len=10)
    assert result["splits"]["gsm_train"]["too_long"] == 2
    assert result["splits"]["gsm_train"]["accepted"] == 0
