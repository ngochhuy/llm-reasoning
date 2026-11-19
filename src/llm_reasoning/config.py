"""Shared, validated configuration for local checks and Kaggle runs."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal


@dataclass(frozen=True)
class PathsConfig:
    root: Path | PurePosixPath

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class DataConfig:
    fineweb_repo: str = "HuggingFaceFW/fineweb-edu"
    math_repo: str = "open-web-math/open-web-math"
    orca_repo: str = "microsoft/orca-math-word-problems-200k"
    gsm8k_repo: str = "openai/gsm8k"
    fineweb_tokens: int = 80_000_000
    math_tokens: int = 20_000_000
    pretrain_validation_fraction: float = 0.01
    gsm8k_validation_examples: int = 500
    orca_min_examples: int = 20_000
    orca_max_examples: int = 40_000
    sft_min_tokens: int = 5_000_000
    sft_max_tokens: int = 8_000_000
    seed: int = 42

    def __post_init__(self) -> None:
        if not all((self.fineweb_repo, self.math_repo, self.orca_repo, self.gsm8k_repo)):
            raise ValueError("dataset repository names must be nonempty")
        if min(self.fineweb_tokens, self.math_tokens, self.gsm8k_validation_examples,
               self.orca_min_examples, self.sft_min_tokens) <= 0:
            raise ValueError("dataset sizes and token budgets must be positive")
        if not 0 < self.pretrain_validation_fraction < 1:
            raise ValueError("pretrain_validation_fraction must be between 0 and 1")
        if self.orca_max_examples < self.orca_min_examples or self.sft_max_tokens < self.sft_min_tokens:
            raise ValueError("dataset maximum must be at least its minimum")


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 8_000
    special_tokens: tuple[str, ...] = ("<eos>", "<pad>", "<think>", "</think>", "<answer>", "</answer>")

    def __post_init__(self) -> None:
        if self.vocab_size < 256 + len(self.special_tokens):
            raise ValueError("vocab_size must fit all byte and special tokens")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 10
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2_048
    max_context_tokens: int = 2_304
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if min(self.n_layers, self.d_model, self.n_heads, self.d_ff, self.max_context_tokens) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")


@dataclass(frozen=True)
class TrainConfig:
    pretrain_seq_len: int = 512
    sft_seq_len: int = 1_024
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    pretrain_lr: float = 3e-4
    sft_lr: float = 3e-5
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    checkpoint_interval_minutes: int = 30
    precision: Literal["fp16", "fp32"] = "fp16"

    def __post_init__(self) -> None:
        if min(self.pretrain_seq_len, self.sft_seq_len, self.micro_batch_size,
               self.gradient_accumulation_steps, self.checkpoint_interval_minutes) <= 0:
            raise ValueError("training lengths, batch sizes and interval must be positive")
        if min(self.pretrain_lr, self.sft_lr, self.gradient_clip_norm) <= 0:
            raise ValueError("learning rates and gradient clip norm must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.precision not in ("fp16", "fp32"):
            raise ValueError("precision must be fp16 or fp32")


@dataclass(frozen=True)
class EvalConfig:
    prompt_lengths: tuple[int, ...] = (128, 512, 1_024, 2_048)
    benchmark_new_tokens: int = 128
    benchmark_repetitions: int = 5
    candidate_counts: tuple[int, ...] = (3, 5, 10)
    max_answer_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self) -> None:
        if not self.prompt_lengths or not self.candidate_counts:
            raise ValueError("evaluation lengths and candidate counts cannot be empty")
        if min(*self.prompt_lengths, *self.candidate_counts, self.benchmark_new_tokens,
               self.benchmark_repetitions, self.max_answer_tokens) <= 0:
            raise ValueError("evaluation lengths and counts must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p in (0, 1]")


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvalConfig

    def __post_init__(self) -> None:
        if max(self.train.pretrain_seq_len, self.train.sft_seq_len) > self.model.max_context_tokens:
            raise ValueError("training sequence length exceeds model context")
        if max(self.evaluation.prompt_lengths) + self.evaluation.benchmark_new_tokens > self.model.max_context_tokens:
            raise ValueError("benchmark prompt plus generated tokens exceeds model context")


def make_config(
    preset: Literal["smoke", "target"] = "target",
    *,
    runtime: Literal["local", "kaggle"] = "local",
    artifact_root: Path | None = None,
) -> ProjectConfig:
    """Return a local smoke-test or Kaggle target configuration without creating files."""
    if preset not in ("smoke", "target"):
        raise ValueError("preset must be smoke or target")
    if runtime not in ("local", "kaggle"):
        raise ValueError("runtime must be local or kaggle")
    if artifact_root is None:
        repo_root = Path(__file__).resolve().parents[2]
        root = repo_root / "artifacts" if runtime == "local" else PurePosixPath("/kaggle/working/llm-reasoning")
    else:
        root = Path(artifact_root).expanduser().resolve()
    if preset == "smoke":
        return ProjectConfig(
            paths=PathsConfig(root), data=DataConfig(), tokenizer=TokenizerConfig(vocab_size=512),
            model=ModelConfig(n_layers=2, d_model=128, n_heads=4, d_ff=512, max_context_tokens=256),
            train=TrainConfig(pretrain_seq_len=64, sft_seq_len=128, micro_batch_size=2,
                              gradient_accumulation_steps=1, precision="fp32"),
            evaluation=EvalConfig(prompt_lengths=(32, 64), benchmark_new_tokens=32,
                                  benchmark_repetitions=2, candidate_counts=(2, 3), max_answer_tokens=64),
        )
    return ProjectConfig(PathsConfig(root), DataConfig(), TokenizerConfig(), ModelConfig(), TrainConfig(), EvalConfig())
