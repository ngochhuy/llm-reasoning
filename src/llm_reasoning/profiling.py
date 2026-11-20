"""Instrumented fixed-length greedy benchmark on the same model and prompts."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from llm_reasoning.checkpoint import file_sha256, load_model_artifact
from llm_reasoning.config import EvalConfig
from llm_reasoning.demo import create_smoke_artifact
from llm_reasoning.kv_cache import KVCache
from llm_reasoning.tokenizer import ByteBPETokenizer


METRICS = ('prefill_ms', 'first_token_ms', 'next_token_ms', 'total_ms',
           'tokens_per_second', 'decode_tokens_per_second', 'baseline_allocated_bytes',
           'peak_allocated_bytes', 'peak_extra_allocated_bytes', 'peak_reserved_bytes', 'cache_bytes')


def make_prompts(tokenizer: ByteBPETokenizer, lengths: list[int]) -> dict[int, torch.Tensor]:
    """Exact lengths, common prefix: repeated ordinary text, no special tokens."""
    base = tokenizer.encode('A student reads a book and solves a simple math problem. Two plus three is five. ')
    if not base or not lengths or any(length <= 0 for length in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError('prompt lengths must be distinct positive integers')
    return {length: torch.tensor([(base * ((length + len(base)-1)//len(base)))[:length]], dtype=torch.long)
            for length in lengths}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_once(model, prompt: torch.Tensor, *, new_tokens: int, use_cache: bool) -> dict:
    """Time N greedy tokens, ignoring EOS so ON/OFF do identical-length work.

    Token timings include synchronized wall time, selection and concatenation;
    prefill_ms measures the initial forward only. Tokenization/loading excluded.
    """
    device = prompt.device
    gc.collect()
    _sync(device)
    if device.type == 'cuda':
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    cache = KVCache(len(model.blocks)) if use_cache else None
    tokens = prompt.clone()
    _sync(device)
    start = time.perf_counter()
    logits = model(tokens) if cache is None else model(tokens, cache=cache)
    _sync(device)
    prefill = time.perf_counter() - start
    if not torch.isfinite(logits[:, -1]).all():
        raise FloatingPointError('nonfinite profiling logits')
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    del logits
    tokens = torch.cat((tokens, next_id), dim=1)
    _sync(device)
    first_token = time.perf_counter() - start
    latencies = []
    for _ in range(new_tokens - 1):
        tick = time.perf_counter()
        logits = model(tokens) if cache is None else model(tokens[:, -1:], cache=cache)
        if not torch.isfinite(logits[:, -1]).all():
            raise FloatingPointError('nonfinite profiling logits')
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        del logits
        tokens = torch.cat((tokens, next_id), dim=1)
        _sync(device)
        latencies.append(time.perf_counter() - tick)
    elapsed = time.perf_counter() - start
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak = reserved = None
    # Transfer token IDs to CPU only after the timed interval and memory reading.
    return dict(prefill_ms=prefill*1000, first_token_ms=first_token*1000,
                next_token_ms=statistics.mean(latencies)*1000, total_ms=elapsed*1000,
                tokens_per_second=new_tokens/elapsed,
                decode_tokens_per_second=(new_tokens-1)/sum(latencies),
                baseline_allocated_bytes=baseline, peak_allocated_bytes=peak,
                peak_extra_allocated_bytes=None if peak is None else peak-baseline,
                peak_reserved_bytes=reserved, cache_bytes=0 if cache is None else cache.nbytes,
                next_token_latencies_ms=[value*1000 for value in latencies],
                generated_token_ids=tokens[0, prompt.shape[1]:].tolist())


def summarize(runs: list[dict], lengths: list[int], repetitions: int) -> list[dict]:
    rows = []
    for length in lengths:
        for use_cache in (False, True):
            matching = [r for r in runs if r['prompt_tokens'] == length and r['use_cache'] == use_cache]
            measured = [r for r in matching if r['phase'] == 'measure']
            valid = [r for r in measured if r['status'] == 'ok']
            warmup_oom = any(r['phase'] == 'warmup' and r['status'] == 'oom' for r in matching)
            status = 'warmup_oom' if warmup_oom else 'ok' if len(valid) == repetitions else 'partial_oom' if valid else 'oom'
            row = dict(prompt_tokens=length, use_cache=use_cache, status=status,
                       successful_runs=len(valid), requested_runs=repetitions,
                       measured_oom_runs=sum(r['status'] == 'oom' for r in measured))
            row.update({name: statistics.median([r[name] for r in valid if r[name] is not None])
                        if any(r[name] is not None for r in valid) else None for name in METRICS})
            rows.append(row)
    return rows


def profile_model(model, prompts: dict[int, torch.Tensor], *, new_tokens: int = 128,
                  repetitions: int = 5, warmup: int = 1) -> dict:
    """Alternate ON/OFF order across repeats; keep actual OOMs explicit."""
    if new_tokens < 2 or repetitions < 1 or warmup < 1 or not prompts:
        raise ValueError('need >=2 new tokens, >=1 repetition and >=1 warmup')
    device = next(model.parameters()).device
    for length, prompt in prompts.items():
        if (prompt.ndim != 2 or prompt.shape != (1, length) or length < 1
                or prompt.dtype != torch.long or length+new_tokens > model.config.max_context_tokens):
            raise ValueError('invalid prompt shape/length or prompt plus output exceeds context')
        if (prompt < 0).any() or (prompt >= model.vocab_size).any():
            raise ValueError('prompt token ID outside vocabulary')
    flags = [(module, module.training) for module in model.modules()]
    runs, checks = [], []
    try:
        model.eval()
        for length, cpu_prompt in prompts.items():
            prompt = cpu_prompt.to(device)
            failed = set()
            for phase, count in [('warmup', warmup), ('measure', repetitions)]:
                for repeat in range(count):
                    order = (False, True) if repeat % 2 == 0 else (True, False)
                    for use_cache in order:
                        if use_cache in failed:
                            continue
                        identity = dict(phase=phase, repetition=repeat, prompt_tokens=length, use_cache=use_cache)
                        try:
                            values = measure_once(model, prompt, new_tokens=new_tokens, use_cache=use_cache)
                        except torch.OutOfMemoryError as error:
                            runs.append(dict(**identity, status='oom', error=str(error)))
                            if phase == 'warmup':
                                failed.add(use_cache)
                        else:
                            runs.append(dict(**identity, status='ok', **values))
                        # No allocator clearing between successful runs. After OOM,
                        # release exception frames and allocator blocks before retry.
                        if runs[-1]['status'] == 'oom':
                            gc.collect()
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
            measured = [r for r in runs if r['prompt_tokens'] == length and r['phase'] == 'measure']
            for repeat in range(repetitions):
                pair = [r for r in measured if r['repetition'] == repeat and r['status'] == 'ok']
                checks.append(dict(prompt_tokens=length, repetition=repeat,
                                   greedy_match=None if len(pair) != 2 else pair[0]['generated_token_ids'] == pair[1]['generated_token_ids']))
            del prompt
    finally:
        for module, was_training in flags:
            module.training = was_training
    return dict(runs=runs, summary=summarize(runs, list(prompts), repetitions), correctness=checks,
                all_pairs_match=all(r['greedy_match'] is True for r in checks))


def export_report(directory: Path, result: dict) -> None:
    """Export raw evidence, scalar CSV, PNG/PDF figure and a short report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for name, rows in [('summary', result['summary']), ('runs', result['runs'])]:
        excluded = {'generated_token_ids', 'next_token_latencies_ms'}
        fields = list(dict.fromkeys(k for row in rows for k in row if k not in excluded))
        with (directory/(name+'.csv')).open('w', encoding='utf-8', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: v for k, v in row.items() if k not in excluded} for row in rows)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    plots = [('tokens_per_second', 'Generation throughput', 'tokens/s'),
             ('first_token_ms', 'First token latency', 'ms'),
             ('next_token_ms', 'Next token latency (mean per run)', 'ms'),
             ('peak_allocated_bytes', 'Peak allocated VRAM', 'MiB')]
    for axis, (metric, title, unit) in zip(axes.flat, plots):
        has_data = False
        for mode in (False, True):
            rows = [r for r in result['summary'] if r['use_cache'] == mode and r[metric] is not None]
            if rows:
                divisor = 2**20 if metric.endswith('_bytes') else 1
                axis.plot([r['prompt_tokens'] for r in rows], [r[metric]/divisor for r in rows],
                          marker='o', label='Cache ON' if mode else 'Cache OFF')
                has_data = True
        axis.set(title=title, xlabel='Prompt length (tokens)', ylabel=unit)
        axis.grid(alpha=.3)
        if has_data:
            axis.legend()
        else:
            axis.text(.5, .5, 'Not available (CPU or OOM)', ha='center', transform=axis.transAxes)
    figure.suptitle('KV cache: median across measured runs | ' + result['metadata']['report_label'])
    figure.tight_layout()
    figure.savefig(directory/'profiling.png', dpi=180)
    figure.savefig(directory/'profiling.pdf')
    plt.close(figure)
    meta = result['metadata']
    lines = ['# KV cache profiling', '', meta['report_label'], '',
             f"Checkpoint SHA256: `{meta['checkpoint_sha256']}`",
             f"Device: {meta['device']} / {meta['device_name']}; dtype: {meta['dtype']}; parameters: {meta['parameters']}",
             f"Batch: {meta['batch_size']}; new tokens: {meta['new_tokens']}; warmup: {meta['warmup']}; repetitions: {meta['repetitions']}", '',
             'Medians exclude warmup and OOM runs; see JSON for per-token timings and token IDs.', '',
             '| Prompt | Cache | Status | Valid runs | Tokens/s | Prefill ms | TTFT ms | Next ms | Peak allocated MiB |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in result['summary']:
        def fmt(key, divisor=1):
            return 'N/A' if row[key] is None else f"{row[key]/divisor:.3f}"
        lines.append(f"| {row['prompt_tokens']} | {'ON' if row['use_cache'] else 'OFF'} | {row['status']} | {row['successful_runs']} | {fmt('tokens_per_second')} | {fmt('prefill_ms')} | {fmt('first_token_ms')} | {fmt('next_token_ms')} | {fmt('peak_allocated_bytes', 2**20)} |")
    lines += ['', 'CPU VRAM is N/A, not zero. Reserved memory is allocator state, not incremental KV usage.',
              'Fixed-length greedy ignores EOS. Each token uses synchronized wall-clock timing including finite-logit checks and instrumentation overhead.',
              'No tokenizer/checkpoint loading in timed regions. See metadata for dtype, hardware, configuration and checkpoint provenance.',
              'All measured greedy pairs match: ' + str(result['all_pairs_match']),
              'A training-tagged checkpoint is self-described provenance, not proof of model quality.', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def _git_metadata() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke', action='store_true')
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/task_2_9_profile'))
    parser.add_argument('--prompt-lengths', nargs='+', type=int)
    parser.add_argument('--new-tokens', type=int)
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['checkpoint', 'float32', 'float16', 'bfloat16'], default='checkpoint')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    defaults = EvalConfig()
    lengths = args.prompt_lengths if args.prompt_lengths is not None else [16, 32] if args.smoke else list(defaults.prompt_lengths)
    new_tokens = args.new_tokens if args.new_tokens is not None else 8 if args.smoke else defaults.benchmark_new_tokens
    repetitions = args.repetitions if args.repetitions is not None else 2 if args.smoke else defaults.benchmark_repetitions
    if not args.smoke and args.tokenizer is None:
        parser.error('--checkpoint requires --tokenizer')
    if args.smoke and args.tokenizer is not None:
        parser.error('--smoke creates its own tokenizer')
    if (not lengths or min(lengths) <= 0 or len(set(lengths)) != len(lengths)
            or new_tokens < 2 or repetitions < 1 or args.warmup < 1 or args.threads < 1):
        parser.error('invalid lengths, new-tokens (>=2), repetitions, warmup or threads')
    try:
        if args.output.exists():
            raise FileExistsError('output directory exists; choose a new directory')
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('select CPU or an available CUDA device')
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        checkpoint, tokenizer_path = (create_smoke_artifact(args.output.with_name(args.output.name+'_checkpoint'), args.seed,
                                                          max_context_tokens=max(2304, max(lengths)+new_tokens)) if args.smoke else
                                      (args.checkpoint, args.tokenizer))
        model, provenance = load_model_artifact(checkpoint, tokenizer_path=tokenizer_path, device=device)
        if args.dtype != 'checkpoint':
            model.to(dtype=getattr(torch, args.dtype))
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
        prompts = make_prompts(tokenizer, lengths)
        result = profile_model(model, prompts, new_tokens=new_tokens, repetitions=repetitions, warmup=args.warmup)
        result['prompt_ids'] = {str(k): v[0].tolist() for k, v in prompts.items()}
        result['metadata'] = dict(
            report_label='CPU/smoke preparation only; not the official post-pretrain T4 report'
            if device.type == 'cpu' or provenance['checkpoint_kind'] != 'training' else
            'Checkpoint measurement; confirm post-pretrain T4 protocol before final submission',
            checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), provenance=provenance,
            tokenizer_sha256=file_sha256(tokenizer_path), parameters=model.num_parameters(),
            device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            dtype=str(next(model.parameters()).dtype), batch_size=1, new_tokens=new_tokens,
            repetitions=repetitions, warmup=args.warmup, seed=args.seed, threads=torch.get_num_threads(),
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            python_version=platform.python_version(), git=_git_metadata(),
            prompt_sha256={str(k): hashlib.sha256(json.dumps(v[0].tolist()).encode()).hexdigest() for k, v in prompts.items()},
            protocol=dict(greedy=True, eos_stopping=False, alternating_order=True,
                          tokenizer_and_load_excluded=True, cuda_sync_per_token=True,
                          allocator_cleared_between_successful_runs=False))
        export_report(args.output, result)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f'Report: {args.output / "report.md"}; all greedy pairs match: {result["all_pairs_match"]}')
    if not result['all_pairs_match']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
