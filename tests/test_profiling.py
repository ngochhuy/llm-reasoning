import json
import os
import subprocess
import sys

import pytest
import torch

from llm_reasoning.config import ModelConfig, TokenizerConfig
from llm_reasoning.generation import generate
from llm_reasoning.model import TransformerLM
from llm_reasoning.profiling import make_prompts, profile_model
from llm_reasoning.tokenizer import ByteBPETokenizer


def model_and_prompts(device='cpu'):
    torch.manual_seed(42)
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=262))
    model = TransformerLM(ModelConfig(n_layers=2, d_model=16, n_heads=4,
                                      d_ff=32, max_context_tokens=32), tokenizer.vocab_size).to(device)
    return model, make_prompts(tokenizer, [4, 8])


def test_fixed_length_measurements_match_generation_and_median():
    model, prompts = model_and_prompts()
    result = profile_model(model, prompts, new_tokens=3, repetitions=2, warmup=1)
    assert model.training  # Original train/eval modes restored.
    assert result['all_pairs_match'] and len(result['runs']) == 12
    for row in result['runs']:
        expected = generate(model, prompts[row['prompt_tokens']], max_new_tokens=3,
                            eos_token_id=None, use_cache=row['use_cache'])
        assert row['generated_token_ids'] == expected[0, -3:].tolist()
        assert len(row['next_token_latencies_ms']) == 2
        assert row['first_token_ms'] >= row['prefill_ms'] > 0
        assert row['tokens_per_second'] == pytest.approx(3000/row['total_ms'])
        assert row['next_token_ms'] == pytest.approx(sum(row['next_token_latencies_ms'])/2)
        assert row['peak_allocated_bytes'] is None
        expected_bytes = 2 * 2 * (row['prompt_tokens']+2) * 16 * 4 if row['use_cache'] else 0
        assert row['cache_bytes'] == expected_bytes
    for summary in result['summary']:
        measured = [r for r in result['runs'] if r['phase'] == 'measure'
                    and r['prompt_tokens'] == summary['prompt_tokens'] and r['use_cache'] == summary['use_cache']]
        assert summary['tokens_per_second'] == pytest.approx(sum(r['tokens_per_second'] for r in measured)/2)
        assert summary['successful_runs'] == 2 and summary['status'] == 'ok'


def test_warmup_oom_is_explicit_and_does_not_change_workload(monkeypatch):
    import llm_reasoning.profiling as module
    original = module.measure_once
    calls = []
    def simulate(model, prompt, *, new_tokens, use_cache):
        calls.append((prompt.shape[1], new_tokens, use_cache))
        if use_cache:
            raise torch.OutOfMemoryError('simulated OOM')
        return original(model, prompt, new_tokens=new_tokens, use_cache=use_cache)
    monkeypatch.setattr(module, 'measure_once', simulate)
    model, prompts = model_and_prompts()
    result = profile_model(model, {4: prompts[4]}, new_tokens=3, repetitions=2)
    assert not result['all_pairs_match']
    assert result['summary'][1]['status'] == 'warmup_oom'
    assert result['summary'][1]['peak_allocated_bytes'] is None
    assert all(length == 4 and count == 3 for length, count, _ in calls)


def test_partial_oom_excludes_failed_run_from_median(monkeypatch):
    import llm_reasoning.profiling as module
    original = module.measure_once
    off_count = 0
    def simulate(model, prompt, *, new_tokens, use_cache):
        nonlocal off_count
        if not use_cache:
            off_count += 1
            if off_count == 2:  # First measured OFF, after successful warmup.
                raise torch.OutOfMemoryError('simulated OOM')
        return original(model, prompt, new_tokens=new_tokens, use_cache=use_cache)
    monkeypatch.setattr(module, 'measure_once', simulate)
    model, prompts = model_and_prompts()
    result = profile_model(model, {4: prompts[4]}, new_tokens=3, repetitions=2)
    assert result['summary'][0]['status'] == 'partial_oom'
    assert result['summary'][0]['successful_runs'] == 1
    assert result['summary'][0]['measured_oom_runs'] == 1


def test_nonfinite_logits_fail_instead_of_reporting_fake_throughput():
    model, prompts = model_and_prompts()
    def inject_nan(module, args, logits):
        return logits * float('nan')
    hook = model.register_forward_hook(inject_nan)
    try:
        with pytest.raises(FloatingPointError, match='nonfinite'):
            profile_model(model, prompts, new_tokens=3, repetitions=1)
    finally:
        hook.remove()
    assert model.training


def test_invalid_configuration_and_exact_prompt_lengths():
    model, prompts = model_and_prompts()
    assert prompts[4].shape == (1, 4)
    assert torch.equal(prompts[4], prompts[8][:, :4])
    with pytest.raises(ValueError):
        profile_model(model, prompts, new_tokens=1)
    with pytest.raises(ValueError, match='context'):
        profile_model(model, prompts, new_tokens=30)
    with pytest.raises(ValueError):
        profile_model(model, prompts, warmup=0)
    with pytest.raises(ValueError):
        make_prompts(ByteBPETokenizer(), [4, 4])


def test_cli_exports_raw_csv_json_report_and_plots(tmp_path):
    output = tmp_path/'profile'
    command = [sys.executable, '-m', 'llm_reasoning.profiling', '--smoke', '--output', str(output),
               '--prompt-lengths', '4', '8', '--new-tokens', '3', '--repetitions', '2']
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                               env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=90)
    assert completed.returncode == 0, completed.stderr
    for name in ['results.json', 'summary.csv', 'runs.csv', 'report.md', 'profiling.png', 'profiling.pdf']:
        assert (output/name).stat().st_size > 0
    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
    assert result['all_pairs_match']
    assert result['metadata']['provenance']['checkpoint_kind'] == 'smoke'
    assert result['metadata']['new_tokens'] == 3
    assert result['metadata']['batch_size'] == 1
    assert result['metadata']['tokenizer_sha256']
    assert 'not the official' in result['metadata']['report_label']
    assert all(row['peak_allocated_bytes'] is None for row in result['summary'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA memory profiling needs GPU')
def test_cuda_memory_counters():
    model, prompts = model_and_prompts(device='cuda')
    result = profile_model(model, prompts, new_tokens=3, repetitions=1)
    for row in result['summary']:
        assert row['peak_allocated_bytes'] >= row['baseline_allocated_bytes'] > 0
        assert row['peak_reserved_bytes'] >= row['peak_allocated_bytes']
        assert row['peak_extra_allocated_bytes'] == row['peak_allocated_bytes']-row['baseline_allocated_bytes']
