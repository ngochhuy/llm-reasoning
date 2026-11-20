import json
import os
import subprocess
import sys

import pytest

from llm_reasoning.demo import create_smoke_artifact, main, run_demo


@pytest.fixture
def smoke(tmp_path):
    return create_smoke_artifact(tmp_path/'smoke', seed=42)


def test_cli_smoke_creates_checkpoint_and_reloads_in_fresh_process(tmp_path):
    directory = tmp_path/'artifacts'
    report = directory/'demo.json'
    prompt_file = tmp_path/'question.txt'
    prompt_file.write_text('Tiếng Việt: 2 + 3 = ?', encoding='utf-8')
    command = [sys.executable, '-m', 'llm_reasoning.demo', '--smoke', '--smoke-dir', str(directory),
               '--max-new-tokens', '4', '--output', str(report), '--prompt-file', str(prompt_file)]
    result = subprocess.run(command, text=True, encoding='utf-8', capture_output=True,
                            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=60)
    assert result.returncode == 0, result.stderr
    saved = json.loads(report.read_text(encoding='utf-8'))
    assert json.loads(result.stdout) == saved
    assert saved['checkpoint_kind'] == 'smoke'
    assert saved['prompt'] == 'Tiếng Việt: 2 + 3 = ?'
    assert saved['saved_train_state']['optimizer_steps'] == 0
    assert saved['forward']['all_finite']
    assert saved['forward']['logits_shape'] == [1, saved['prompt_tokens'], saved['vocab_size']]
    assert 0 < saved['generated_tokens'] <= 4
    assert all(0 <= token < saved['vocab_size'] for token in saved['generated_token_ids'])
    command = [sys.executable, '-m', 'llm_reasoning.demo', '--checkpoint', str(directory/'random_model.pt'),
               '--tokenizer', str(directory/'tokenizer.json'), '--max-new-tokens', '4', '--no-cache',
               '--prompt-file', str(prompt_file)]
    second = subprocess.run(command, text=True, encoding='utf-8', capture_output=True, timeout=60)
    assert second.returncode == 0, second.stderr
    replay = json.loads(second.stdout)
    assert replay['generated_token_ids'] == saved['generated_token_ids']
    assert replay['checkpoint_sha256'] == saved['checkpoint_sha256']


def test_sampling_reproducible_from_loaded_checkpoint(smoke):
    checkpoint, tokenizer = smoke
    kwargs = dict(prompt='2 + 3 = ?', max_new_tokens=5, temperature=.7, top_p=.95, seed=11)
    first = run_demo(checkpoint, tokenizer, **kwargs)
    second = run_demo(checkpoint, tokenizer, **kwargs)
    assert first['generated_token_ids'] == second['generated_token_ids']
    assert first['decoding']['use_cache']
    assert first['checkpoint_kind'] == 'smoke'


def test_raw_prompt_zero_budget_and_context_error(smoke):
    checkpoint, tokenizer = smoke
    result = run_demo(checkpoint, tokenizer, prompt='Hello', prompt_format='raw', max_new_tokens=0)
    assert result['encoded_prompt'] == 'Hello'
    assert result['generated_tokens'] == 0 and result['generated_text'] == ''
    assert result['stop_reason'] == 'max_new_tokens'
    with pytest.raises(ValueError, match='context'):
        run_demo(checkpoint, tokenizer, prompt='x '*1000)


def test_wrong_tokenizer_rejected(smoke, tmp_path):
    checkpoint, tokenizer = smoke
    altered = tmp_path/'altered.json'
    altered.write_bytes(tokenizer.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA256'):
        run_demo(checkpoint, altered, prompt='Hello')


@pytest.mark.parametrize('args', [[], ['--checkpoint', 'missing.pt'],
                                  ['--smoke', '--temperature', '-1'], ['--smoke', '--top-p', '0'],
                                  ['--smoke', '--max-new-tokens', '-1']])
def test_invalid_cli_arguments_exit_with_usage_error(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_smoke_refuses_overwriting_existing_artifact(smoke):
    checkpoint, _ = smoke
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        create_smoke_artifact(checkpoint.parent, seed=9)
    assert before == checkpoint.read_bytes()


def test_old_checkpoint_without_kind_is_explicitly_unclassified(smoke, tmp_path):
    import torch
    checkpoint, tokenizer = smoke
    payload = torch.load(checkpoint, weights_only=True)
    del payload['checkpoint_kind']
    older = tmp_path/'old.pt'
    torch.save(payload, older)
    result = run_demo(older, tokenizer, prompt='Hello', max_new_tokens=1)
    assert result['checkpoint_kind'] == 'unclassified'
