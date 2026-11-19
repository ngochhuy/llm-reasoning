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
