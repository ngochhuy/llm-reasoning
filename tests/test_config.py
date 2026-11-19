from dataclasses import replace
from pathlib import Path

import pytest

from llm_reasoning.config import ModelConfig, ProjectConfig, make_config


def test_presets_and_runtime_paths(tmp_path: Path) -> None:
    smoke = make_config("smoke", artifact_root=tmp_path)
    target = make_config("target", runtime="kaggle")
    assert smoke.paths.data == tmp_path / "data"
    assert smoke.train.pretrain_seq_len < target.train.pretrain_seq_len
    assert target.model.n_layers == 10
    assert target.paths.root.as_posix() == "/kaggle/working/llm-reasoning"
    assert not smoke.paths.checkpoints.exists()


def test_invalid_model_geometry_fails_early() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=510, n_heads=8)
    with pytest.raises(ValueError, match="RoPE"):
        ModelConfig(d_model=120, n_heads=8)


def test_context_budget_fails_early() -> None:
    config = make_config("target")
    with pytest.raises(ValueError, match="benchmark prompt"):
        replace(config, model=replace(config.model, max_context_tokens=2_048))
    with pytest.raises(ValueError, match="training sequence"):
        ProjectConfig(
            config.paths, config.data, config.tokenizer,
            replace(config.model, max_context_tokens=512),
            config.train, replace(config.evaluation, prompt_lengths=(128,), benchmark_new_tokens=128),
        )
