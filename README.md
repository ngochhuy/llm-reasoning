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

## Stage 2 notebook handoff (task 2.10)

Notebooks `03_model_sanity_checks.ipynb`, `04_pretraining.ipynb`, `05_kv_cache_profiling.ipynb`, and `06_math_sft.ipynb` are generated from `scripts/build_stage2_notebooks.py`. Notebook 03 checks the core, toy overfit and exact CPU resume. Notebooks 04–06 default to `RUN_MODE='inspect'`; change to `smoke` for a tiny CPU flow on the prepared artifacts. Production/T4 remains gated by `RUN_MODE='production'` (or `official` for profiling), pinned Git SHA, CUDA and reviewed data. A pretrain checkpoint moves into notebook 05 and 06; SFT loads its weights but starts a fresh optimizer. Both training notebooks save full-state `latest.pt`, CSV/TensorBoard logs and a run manifest. See [task 2.10 report](docs/task_2_10_notebooks_report.md) for input paths, smoke verification and the next review gate.

```powershell
uv sync --group model --group train --group data --group profile --group notebook --group dev
uv run python scripts/build_stage2_notebooks.py
uv run python scripts/verify_stage2_notebooks.py --smoke --sanity
uv run pytest -q
```

## Model building blocks (task 2.1)

Install the model dependency and run the checks on CPU:

```powershell
uv sync --group model --group dev
uv run --group model pytest tests/test_layers.py -q
```

`llm_reasoning.layers` provides `RMSNorm`, `RotaryEmbedding` and bias-free GELU `FeedForward`. RoPE takes `[batch, heads, sequence, head_dim]` and an explicit `offset` for future KV cache decoding. It rotates adjacent coordinate pairs. Decoder assembly will apply normalization before attention/FFN and add residual connections. These layers are not a trained language model; attention and decoder assembly follow in tasks 2.2 and 2.3.

Task 2.2 adds `llm_reasoning.attention.CausalSelfAttention`: bias-free Q/K/V and output projections, RoPE and an explicit causal mask. Score computation and softmax use FP32 under mixed precision. Run `uv run --group model pytest tests/test_attention.py -q` for its checks. KV cache and full decoder assembly follow in later tasks.

Task 2.3 adds `llm_reasoning.model.TransformerLM(config.model, vocab_size=tokenizer.vocab_size)`, returning logits shaped `[batch, sequence, vocabulary]`. The target configuration has 35,564,032 unique parameters at vocabulary size 8,000 with tied embeddings. Run `uv run --group model pytest tests/test_model.py -q` to check model assembly, causal outputs, backward and weight tying. Generation, KV cache and training are implemented in subsequent tasks.

Task 2.4 adds `llm_reasoning.generation.generate(model, input_ids, max_new_tokens=32, eos_token_id=..., temperature=0.0, top_p=1.0)`. Supply one unpadded prompt `[1, T]` on the model's device; the result includes the prompt and generated tokens, stopping at a new EOS or the token/context limit. Temperature zero selects greedy; positive temperature enables sampling with optional `torch.Generator`. Decoding currently recomputes the full prefix. Run `uv run --group model pytest tests/test_generation.py -q`; see `docs/task_2_4_generation_report.md` for a tokenizer example. Random weights demonstrate the mechanics only.

Task 2.5 adds optional `generate(..., use_cache=True)`: prefill once, then decode one token per step with a fresh per-request cache. For direct inference, create `KVCache(model.config.n_layers)` and pass `model(new_input_ids, cache=cache)` inside `model.eval()` plus `torch.inference_mode()`. Only pass new tokens after prefill. Call `cache.reset()` between prompts or after updating weights. `use_cache=False` remains the default reference path. Run `uv run --group model pytest tests/test_kv_cache.py -q`; correctness checks and limitations are documented in `docs/task_2_5_kv_cache_report.md`. These CPU checks are not T4 profiling measurements.

## Training loop checks (task 2.6)

`llm_reasoning.training.Trainer` supports pretrain/SFT shifted loss, response masks, AdamW, token-weighted gradient accumulation, clipping, warmup/cosine decay, CUDA FP16 GradScaler, validation and CSV/TensorBoard logging. CPU checks use FP32; the CUDA FP16 test is skipped without a GPU. Checkpoint/resume follows in task 2.7.

```powershell
uv sync --group model --group train --group dev
uv run --group model --group train pytest tests/test_training.py -q
uv run --group model --group train python scripts/task_2_6_smoke.py --output artifacts/task_2_6_smoke_new
uv run --group model --group train tensorboard --logdir artifacts/task_2_6_smoke_verified
```

The smoke script overfits synthetic data on a tiny CPU model; it does not start production training. Choose a new output directory for each run. See `docs/task_2_6_training_report.md` for real results, mask semantics and remaining integration work.

## Checkpoint/resume checks (task 2.7)

`checkpoint.save_checkpoint` saves model/config, optimizer, scheduler, scaler, counters, data-stream position/order, tokenizer SHA256 and Python/Torch/NumPy RNG at a completed accumulation boundary. Use `ResumableBatchStream` with immutable indexable micro-batches and a fingerprint covering data plus batching settings. `load_checkpoint` reconstructs the trainer and restores RNG last. `load_model_checkpoint` loads weights only for inference or starting an SFT optimizer. `PeriodicCheckpoint` is an update callback for interval saves; explicitly save again at stage/session end.

```powershell
uv run --group model --group train pytest tests/test_checkpoint.py -q
uv run --group model --group train python scripts/task_2_7_smoke.py --output artifacts/task_2_7_smoke_new
```

Choose a new output directory. The CPU smoke compares continuous training with interruption/resume on synthetic data, exporting checkpoints, CSV, TensorBoard and `summary.json`. Explicit `TrainingLogger(directory, resume_step=...)` appends logs only if existing rows are not ahead of the restored step; otherwise use a new log segment. See `docs/task_2_7_checkpoint_report.md` for results and the remaining CUDA/data-adapter checks. Save Kaggle outputs persistently before ending a session.

## One-command core demo (task 2.8)

Create a tiny random checkpoint, save and reload it, check forward logits and generate tokens:

```powershell
uv run --group model python -m llm_reasoning.demo --smoke
```

This does not train the model. The JSON output explicitly identifies the smoke checkpoint, reports zero optimizer updates, finite forward logits, model size and generated token IDs/text. Random output is not expected to answer questions. If the default `artifacts/task_2_8_demo` directory exists, pass a new `--smoke-dir`.

Reload an existing artifact in a fresh process:

```powershell
uv run --group model python -m llm_reasoning.demo --checkpoint artifacts/task_2_8_demo_verified/random_model.pt --tokenizer artifacts/task_2_8_demo_verified/tokenizer.json --prompt "2 + 3 = ?" --max-new-tokens 8
```

Use `--no-cache` for the full-prefix reference path, or `--temperature 0.7 --top-p 0.95 --seed 42` for sampling. `--prompt-file question.txt` reads UTF-8; `--prompt-format raw` disables the question template. Optional `--output new_report.json` saves a report without overwriting files. Default device is CPU; `--device cuda` requires CUDA. Demo timings are not the official profiling benchmark. See `docs/task_2_8_demo_report.md` for evidence and limitations.

## KV cache profiling tool (task 2.9)

```powershell
uv run --group model --group profile python -m llm_reasoning.profiling --smoke --output artifacts/task_2_9_profile_new
```

The quick CPU smoke creates random weights and compares fixed-length greedy cache ON/OFF. A checkpoint run defaults to prompt lengths 128/512/1024/2048, 128 new tokens, 5 measured repetitions and 1 warmup. Example for a post-pretrain checkpoint on T4:

```bash
python -m llm_reasoning.profiling --checkpoint /path/to/pretrain.pt --tokenizer /path/to/tokenizer.json --device cuda --dtype float16 --new-tokens 128 --repetitions 5 --warmup 2 --output /kaggle/working/kv_profile
```

Results include raw JSON/CSV, median summary CSV, PNG/PDF plots and `report.md`. TTFT includes prefill; next-token latency averages later steps within each run. CUDA timings synchronize per token and memory peaks reset per run. EOS stopping is disabled to keep work equal. CPU VRAM is unavailable, never reported as zero. OOM is recorded explicitly; missing or mismatched ON/OFF pairs return exit code 1 after exporting results. Use a new output directory. Current CPU/random-weight results are preparation evidence, not the official T4 report. See `docs/task_2_9_profiling_report.md`.

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

After the pretraining tokenizer exists, prepare GSM8K only with the same tokenizer:

```bash
uv run --group data python -m llm_reasoning.sft_data \
  --pretrain-dir /kaggle/working/llm-reasoning/data/pretrain \
  --output /kaggle/working/llm-reasoning/data/sft
```

The command writes `gsm_train.jsonl`, `gsm_validation.jsonl`, the original `gsm_test.jsonl`, and `manifest.json`. Training records contain `input_ids`, `loss_mask`, and the final numeric answer. The `loss_mask` is zero for the question prompt and one for the solution and answer; the future training loop will shift it with causal LM targets. The test file contains no SFT tokens. Review the manifest and sample answers before using these files for GPU training.

## Stage 1 notebooks

Run `notebooks/01_data_exploration.ipynb` first on Kaggle CPU with Internet enabled. It inspects 100 initial rows from each dataset and prints schema, sample lengths, examples, dataset revisions, and preliminary answer parsing. These are exploratory rows, not representative random samples.

Both notebooks can also run locally in VS Code using the project's `.venv` kernel. Run `uv sync --group data --group notebook`, open the notebook, and select `.venv` as its Python kernel. Internet access is needed for real dataset cells; the synthetic smoke cell in notebook 02 runs offline. Local artifacts are written under `artifacts/` in this repository. Start with the dry run before attempting the full 100M-token preparation, which can take substantial CPU time and disk space.

Next run `notebooks/02_tokenizer_data_preparation.ipynb`. Its synthetic smoke cell works without network. Set `RUN_PRETRAIN` or `RUN_SFT` to `True` in its preparation cell to create real artifacts, using `DRY_RUN=True` for small token budgets first. Set `DRY_RUN=False` for target budgets after reviewing the data and CPU timing. Both notebooks have a `REPO_REF` field; pin it to a Git commit SHA before the full run. In a new Kaggle session, attach the pretraining output and update `PRETRAIN_DIR` to that attached path before running SFT. Save and verify the Kaggle output before ending the session.

### GSM8K-only SFT

Notebook 01 now includes dataset metadata inventory, sampled pretraining quality checks, full GSM8K train exploration, length histograms, and optional project-tokenizer length measurements. It exports `artifacts/eda/summary.json` and SVG charts locally. Metadata counts describe the selected source subset; sample quality statistics do not describe the entire corpus. Run it on CPU with Internet to fetch metadata and pretraining samples.

SFT now uses GSM8K only (no Orca download or filtering). Use `--output artifacts/data/sft_gsm8k` with `python -m llm_reasoning.sft_data --pretrain-dir artifacts/data/pretrain`. Notebook 02 uses this new directory. Legacy mixed directories containing `orca_train.jsonl` are rejected. Reserve 500 train questions for validation and all 1,319 test questions for final evaluation. Plan 1-3 epochs selected by validation with an 8M processed-token ceiling; there is no 5M minimum. Recreate SFT artifacts after training the final tokenizer.
