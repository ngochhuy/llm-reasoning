# LLM Reasoning

Small decoder-only language model for the Track C mathematical reasoning project.

The implementation follows [`De_cuong_ky_thuat_llm-reasoning_v3.md`](De_cuong_ky_thuat_llm-reasoning_v3.md) and the staged [`KE_HOACH_THUC_HIEN.md`](KE_HOACH_THUC_HIEN.md).

## Local development

Install Python 3.11 and [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync --dev
uv run pytest
```

Kaggle GPU setup, data preparation, training, and evaluation commands will be added in their respective tasks. Generated data and checkpoints are stored outside Git.

## Configuration

`llm_reasoning.config.make_config()` returns validated settings. Use `preset="smoke"` for local checks and `preset="target"` for the planned 10-layer model. Set `runtime="kaggle"` to place artifacts under `/kaggle/working/llm-reasoning`, or pass `artifact_root=Path(...)` to choose another location. The function only describes paths; it does not create directories or download data.

```python
from llm_reasoning.config import make_config

config = make_config("smoke")
print(config.model, config.paths.checkpoints)
```

## Prepare pretraining data on Kaggle CPU

Install the optional data dependencies and run the streaming preparation command in a Kaggle CPU notebook or terminal:

```bash
uv sync --group data
uv run --group data python -m llm_reasoning.pretrain_data --output /kaggle/working/llm-reasoning
```

The command reads FineWeb-Edu `sample-10BT` and OpenWebMath, trains the custom BPE on train documents, then writes uint16 blocks of 512 tokens by source and split. Outputs include `tokenizer.json`, source manifests, a combined manifest and binary shards under `data/pretrain/`. The default targets are 80M education tokens and 20M math tokens, counted with this tokenizer. Use `--fineweb-tokens` and `--math-tokens` for a small dry run before the full preparation. The actual data volumes and processing time have not yet been verified on Kaggle.

## Prepare math SFT data on Kaggle CPU

After the pretraining tokenizer exists, prepare Orca Math and GSM8K with the same tokenizer:

```bash
uv run --group data python -m llm_reasoning.sft_data \
  --pretrain-dir /kaggle/working/llm-reasoning/data/pretrain \
  --output /kaggle/working/llm-reasoning/data/sft
```

The command writes `orca_train.jsonl`, `gsm_train.jsonl`, `gsm_validation.jsonl`, the original `gsm_test.jsonl`, and `manifest.json`. Training records contain `input_ids`, `loss_mask`, and the final numeric answer. The `loss_mask` is zero for the question prompt and one for the solution and answer; the future training loop will shift it with causal LM targets. The test file contains no SFT tokens. Review the manifest and sample answers before using these files for GPU training.
