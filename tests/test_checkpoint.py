import csv
import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from llm_reasoning.checkpoint import (PeriodicCheckpoint, file_sha256, load_checkpoint,
                                     load_model_checkpoint, save_checkpoint)
from llm_reasoning.config import ModelConfig, TokenizerConfig, TrainConfig
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer, TrainingLogger


class RandomBatches:
    """Consume all three global RNGs so exact replay requires their restoration."""
    def __len__(self):
        return 13

    def __getitem__(self, index):
        offset = index + random.randrange(10) + int(np.random.randint(10))
        ids = (torch.randint(0, 30, (1, 6)) + offset) % 40
        return dict(input_ids=ids, loss_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]))


def new_stream(fingerprint='immutable-synthetic-batches-v1'):
    return ResumableBatchStream(RandomBatches(), fingerprint=fingerprint, seed=9)


def new_trainer(stage='pretrain', device='cpu', precision='fp32'):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = TransformerLM(ModelConfig(n_layers=1, d_model=8, n_heads=2, d_ff=16,
                                      max_context_tokens=16), 262).to(device)
    cfg = TrainConfig(precision=precision, gradient_accumulation_steps=2, warmup_fraction=.2)
    return Trainer(model, cfg, stage=stage, total_steps=10)


def tokenizer_file(tmp_path):
    path = tmp_path/'tokenizer.json'
    ByteBPETokenizer(TokenizerConfig(vocab_size=262)).save(path)
    return path


def assert_equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal_nested(x, y)
    else:
        assert a == b


def finish(trainer, stream):
    rows = trainer.train_epoch(stream)
    if trainer.state.attempted_steps < trainer.total_steps:
        stream.next_epoch()
        rows += trainer.train_epoch(stream)
    return rows


@pytest.mark.parametrize('stage', ['pretrain', 'sft'])
def test_exact_resume_matches_continuous_across_epochs_and_all_rngs(tmp_path, stage):
    tok = tokenizer_file(tmp_path)
    continuous, full_stream = new_trainer(stage), new_stream()
    expected_rows = finish(continuous, full_stream)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))

    interrupted, first_stream = new_trainer(stage), new_stream()
    first_rows = interrupted.train_epoch(first_stream, max_updates=3)
    checkpoint = tmp_path/'latest.pt'
    save_checkpoint(checkpoint, interrupted, first_stream, tokenizer_path=tok)
    assert first_stream.cursor == 6
    # Deliberately disturb RNG as a fresh notebook/model construction would.
    random.random(); np.random.random(); torch.rand(100)
    restored_stream = new_stream()
    resumed = load_checkpoint(checkpoint, restored_stream, tokenizer_path=tok, expected_stage=stage)
    rows = first_rows + finish(resumed, restored_stream)
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    assert expected_rng[:2] == actual_rng[:2]
    assert_equal_nested(expected_rng[2], actual_rng[2])
    for key in ['loss', 'lr', 'processed_tokens', 'supervised_tokens', 'update_skipped']:
        assert [r[key] for r in rows] == [r[key] for r in expected_rows]
    assert_equal_nested(continuous.model.state_dict(), resumed.model.state_dict())
    assert_equal_nested(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_equal_nested(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert_equal_nested(continuous.scaler.state_dict(), resumed.scaler.state_dict())
    assert asdict(continuous.state) == asdict(resumed.state)
    assert_equal_nested(full_stream.state_dict(), restored_stream.state_dict())
    assert resumed.model.lm_head.weight is resumed.model.token_embedding.weight


def test_tokenizer_data_stage_and_format_mismatch_are_rejected(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    checkpoint = tmp_path/'latest.pt'
    save_checkpoint(checkpoint, trainer, stream, tokenizer_path=tok)
    with pytest.raises(ValueError, match='fingerprint'):
        load_checkpoint(checkpoint, new_stream('changed'), tokenizer_path=tok)
    with pytest.raises(ValueError, match='stage'):
        load_checkpoint(checkpoint, new_stream(), tokenizer_path=tok, expected_stage='sft')
    changed_tok = tmp_path/'changed_tokenizer.json'
    changed_tok.write_bytes(tok.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='SHA256'):
        load_checkpoint(checkpoint, new_stream(), tokenizer_path=changed_tok)
    payload = torch.load(checkpoint, weights_only=True)
    payload['format_version'] = 999
    torch.save(payload, tmp_path/'bad.pt')
    with pytest.raises(ValueError, match='format'):
        load_checkpoint(tmp_path/'bad.pt', new_stream(), tokenizer_path=tok)


def test_model_only_transfer_resets_stage_optimizer_and_matches_logits(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    trainer.train_epoch(stream, max_updates=2)
    checkpoint = tmp_path/'pretrain.pt'
    save_checkpoint(checkpoint, trainer, stream, tokenizer_path=tok)
    model = load_model_checkpoint(checkpoint, tokenizer_path=tok)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids), trainer.model(ids), rtol=0, atol=0)
    assert not model.training
    sft = Trainer(model, trainer.config, stage='sft', total_steps=4)
    assert sft.state.optimizer_steps == 0 and not sft.optimizer.state
    assert sft.stage == 'sft'
    assert not any('cache' in key for key in model.state_dict())


def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import llm_reasoning.checkpoint as module
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    path = tmp_path/'latest.pt'
    save_checkpoint(path, trainer, stream, tokenizer_path=tok)
    old_hash = file_sha256(path)
    def fail(*args, **kwargs):
        raise OSError('simulated disk write failure')
    monkeypatch.setattr(module.torch, 'save', fail)
    with pytest.raises(OSError, match='disk write'):
        save_checkpoint(path, trainer, stream, tokenizer_path=tok)
    assert file_sha256(path) == old_hash
    assert not list(tmp_path.glob('*.tmp'))


def test_update_hook_periodic_save_and_boundary_guard(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    path = tmp_path/'latest.pt'
    periodic = PeriodicCheckpoint(path, stream, tokenizer_path=tok, interval_minutes=1)
    periodic.last_save -= 61
    trainer.train_epoch(stream, max_updates=1, on_update=periodic)
    assert path.exists()
    payload = torch.load(path, weights_only=True)
    assert payload['data_stream']['cursor'] == 2
    assert payload['train_state']['attempted_steps'] == 1
    trainer._at_update_boundary = False
    with pytest.raises(ValueError, match='boundary'):
        save_checkpoint(path, trainer, stream, tokenizer_path=tok)


def test_logging_resume_appends_unique_steps_and_refuses_ahead_logs(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    directory = tmp_path/'logs'
    checkpoint = tmp_path/'latest.pt'
    with TrainingLogger(directory) as logger:
        trainer.logger = logger
        trainer.train_epoch(stream, max_updates=2)
        save_checkpoint(checkpoint, trainer, stream, tokenizer_path=tok)
    with TrainingLogger(directory, resume_step=2) as logger:
        resumed_stream = new_stream()
        resumed = load_checkpoint(checkpoint, resumed_stream, tokenizer_path=tok, logger=logger)
        resumed.train_epoch(resumed_stream, max_updates=1)
    with (directory/'metrics.csv').open(encoding='utf-8') as source:
        assert [int(row['step']) for row in csv.DictReader(source)] == [1, 2, 3]
    with pytest.raises(ValueError, match='ahead'):
        TrainingLogger(directory, resume_step=2)


def test_stream_epoch_guard_and_invalid_order():
    stream = new_stream()
    with pytest.raises(ValueError, match='consuming'):
        stream.next_epoch()
    state = stream.state_dict()
    state['order'] = torch.zeros(13, dtype=torch.long)
    with pytest.raises(ValueError, match='cursor/order'):
        stream.load_state_dict(state)


def test_wrong_stream_and_mid_accumulation_save_are_rejected(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(), new_stream()
    trainer.train_epoch(stream, max_updates=1)
    with pytest.raises(ValueError, match='same object'):
        save_checkpoint(tmp_path/'wrong.pt', trainer, new_stream(), tokenizer_path=tok)
    def attempt_unsafe_save(*args):
        save_checkpoint(tmp_path/'unsafe.pt', trainer, stream, tokenizer_path=tok)
    hook = trainer.model.register_forward_pre_hook(attempt_unsafe_save)
    try:
        with pytest.raises(ValueError, match='boundary'):
            trainer.train_epoch(stream, max_updates=1)
    finally:
        hook.remove()
    assert not (tmp_path/'unsafe.pt').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA resume/FP16 requires GPU')
def test_cuda_fp16_scaler_and_optimizer_resume(tmp_path):
    tok = tokenizer_file(tmp_path)
    trainer, stream = new_trainer(device='cuda', precision='fp16'), new_stream()
    trainer.train_epoch(stream, max_updates=2)
    path = tmp_path/'cuda.pt'
    save_checkpoint(path, trainer, stream, tokenizer_path=tok)
    restored_stream = new_stream()
    restored = load_checkpoint(path, restored_stream, tokenizer_path=tok, device='cuda')
    assert_equal_nested(trainer.scaler.state_dict(), restored.scaler.state_dict())
    assert_equal_nested(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert restored.train_epoch(restored_stream, max_updates=1)
