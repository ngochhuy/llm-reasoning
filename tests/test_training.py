import copy
import csv
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from llm_reasoning.config import ModelConfig, TrainConfig
from llm_reasoning.model import TransformerLM
from llm_reasoning.training import Trainer, TrainingLogger, causal_lm_loss, collate_sft, lr_multiplier, target_mask


def small_model():
    return TransformerLM(ModelConfig(n_layers=1, d_model=16, n_heads=4, d_ff=32,
                                     max_context_tokens=16), 17)


def train_config(**kwargs):
    return replace(TrainConfig(precision='fp32', gradient_accumulation_steps=1,
                               warmup_fraction=0, gradient_clip_norm=100), **kwargs)


def test_shifted_response_loss_excludes_prompt_and_padding_gradients():
    batch = collate_sft([
        dict(input_ids=[1, 2, 3, 4], loss_mask=[0, 0, 1, 1]),
        dict(input_ids=[5, 6, 7], loss_mask=[0, 0, 1]),
    ], pad_token_id=0)
    logits = torch.randn(2, 4, 17, requires_grad=True)
    summed, count = causal_lm_loss(logits, batch, sft=True)
    expected = F.cross_entropy(torch.stack([logits[0, 1], logits[0, 2], logits[1, 1]]),
                               torch.tensor([3, 4, 7]), reduction='sum')
    assert count == 3
    torch.testing.assert_close(summed, expected)
    summed.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 3]) == 0
    assert torch.count_nonzero(logits.grad[1, 2]) == 0
    # First response token is learned from the last prompt position.
    assert logits.grad[0, 1].abs().sum() > 0
    _, pretrain_count = causal_lm_loss(logits, batch, sft=False)
    assert pretrain_count == 5


def test_accumulation_matches_large_batch_with_unequal_response_counts():
    torch.manual_seed(4)
    a = small_model()
    b = copy.deepcopy(a)
    records = [dict(input_ids=[1, 2, 3, 4], loss_mask=[0, 0, 0, 1]),
               dict(input_ids=[5, 6, 7, 8], loss_mask=[0, 1, 1, 1])]
    large = Trainer(a, train_config(sft_lr=.001), stage='sft', total_steps=1)
    micro = Trainer(b, train_config(sft_lr=.001, gradient_accumulation_steps=2), stage='sft', total_steps=1)
    row_a = large.train_epoch([collate_sft(records, 0)])[0]
    row_b = micro.train_epoch([collate_sft([r], 0) for r in records])[0]
    assert abs(row_a['loss'] - row_b['loss']) < 1e-6
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
    assert micro.state.supervised_tokens == 4 and micro.state.processed_tokens == 8


def test_partial_accumulation_window_and_token_weighted_validation():
    model = small_model()
    trainer = Trainer(model, train_config(gradient_accumulation_steps=2), stage='pretrain', total_steps=3)
    batches = [{'input_ids': torch.tensor([[1, 2, 3]])} for _ in range(3)]
    rows = trainer.train_epoch(batches)
    assert len(rows) == trainer.state.optimizer_steps == 2
    assert trainer.state.processed_tokens == 9 and trainer.state.supervised_tokens == 6
    model.train()
    model.blocks[0].eval()
    flags = [m.training for m in model.modules()]
    val = [batches[0], {'input_ids': torch.tensor([[4, 5, 6, 7, 8]])}]
    with torch.no_grad():
        losses = [causal_lm_loss(model(x['input_ids']), x) for x in val]
    expected = sum(float(x[0]) for x in losses) / sum(x[1] for x in losses)
    actual = trainer.evaluate(val)
    assert abs(actual['loss'] - expected) < 1e-6
    assert flags == [m.training for m in model.modules()]


@pytest.mark.parametrize('stage', ['pretrain', 'sft'])
def test_overfit_one_batch(stage):
    torch.manual_seed(42)
    model = small_model()
    trainer = Trainer(model, train_config(pretrain_lr=.01, sft_lr=.01), stage=stage, total_steps=60)
    batch = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])}
    if stage == 'sft':
        batch['loss_mask'] = torch.tensor([[0, 0, 1, 1, 1, 1]]*2)
    before = trainer.evaluate([batch])['loss']
    trainer.train_epoch([batch]*60)
    after = trainer.evaluate([batch])['loss']
    assert after < before * .15, (before, after)


def test_schedule_warmup_cosine_and_parameter_groups():
    factors = [lr_multiplier(i, total_steps=10, warmup_steps=2) for i in range(10)]
    assert factors[:3] == [.5, 1., 1.]
    assert factors[-1] == pytest.approx(.1)
    assert all(x >= y for x, y in zip(factors[2:], factors[3:]))
    trainer = Trainer(small_model(), train_config(pretrain_lr=.001, warmup_fraction=.2), stage='pretrain', total_steps=10)
    assert trainer.optimizer.param_groups[0]['lr'] == .0005
    assert trainer.optimizer.param_groups[1]['weight_decay'] == 0
    rows = trainer.train_epoch([{'input_ids': torch.tensor([[1, 2, 3]])}]*3)
    assert [r['lr'] for r in rows] == [.0005, .001, .001]
    assert trainer.state.optimizer_steps == 3


def test_clipping_is_applied_before_optimizer_step():
    trainer = Trainer(small_model(), train_config(gradient_clip_norm=.001), stage='pretrain', total_steps=1)
    norms = []
    hook = trainer.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: norms.append(float(torch.stack([
            p.grad.norm()**2 for p in trainer.model.parameters() if p.grad is not None]).sum().sqrt())))
    try:
        rows = trainer.train_epoch([{'input_ids': torch.tensor([[1, 2, 3, 4]])}])
    finally:
        hook.remove()
    assert rows[0]['grad_norm'] > .001
    assert norms[0] <= .001001


def test_csv_and_tensorboard_logs_can_be_read(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with TrainingLogger(tmp_path) as logger:
        trainer = Trainer(small_model(), train_config(), stage='sft', total_steps=1, logger=logger)
        batch = collate_sft([dict(input_ids=[1, 2, 3], loss_mask=[0, 1, 1])], 0)
        trainer.train_epoch([batch])
        trainer.evaluate([batch])
    rows = list(csv.DictReader((tmp_path/'metrics.csv').open(encoding='utf-8')))
    assert [r['kind'] for r in rows] == ['train', 'validation']
    events = EventAccumulator(str(tmp_path/'tensorboard')).Reload()
    assert len(events.Scalars('sft/train/loss')) == 1
    assert len(events.Scalars('sft/validation/loss')) == 1
    with pytest.raises(FileExistsError):
        TrainingLogger(tmp_path)


def test_rejects_missing_masks_empty_targets_left_padding_and_cpu_fp16():
    model = small_model()
    batch = {'input_ids': torch.tensor([[1, 2, 3]])}
    with pytest.raises(ValueError, match='CUDA'):
        Trainer(model, TrainConfig(), stage='pretrain', total_steps=1)
    trainer = Trainer(model, train_config(), stage='sft', total_steps=1)
    with pytest.raises(ValueError, match='loss_mask'):
        trainer.train_epoch([batch])
    # A failed window may have consumed input: resume from checkpoint or use a
    # fresh trainer for this separate invalid-input case.
    with pytest.raises(ValueError, match='previous update failed'):
        trainer.train_epoch([batch])
    trainer = Trainer(model, train_config(), stage='sft', total_steps=1)
    with pytest.raises(ValueError, match='no supervised'):
        trainer.train_epoch([{**batch, 'loss_mask': torch.zeros(1, 3)}])
    with pytest.raises(ValueError, match='right padding'):
        target_mask({**batch, 'attention_mask': torch.tensor([[0, 1, 1]])}, sft=False)
    with pytest.raises(ValueError, match='binary'):
        target_mask({**batch, 'loss_mask': torch.tensor([[0, 2, 1]])}, sft=True)
    with pytest.raises(ValueError, match='no supervised'):
        trainer.evaluate([])


def test_skipped_optimizer_update_does_not_advance_schedule():
    trainer = Trainer(small_model(), train_config(), stage='pretrain', total_steps=1)
    original = [p.detach().clone() for p in trainer.model.parameters()]
    class SkippingScaler:
        value = 2.0
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def get_scale(self):
            return self.value
        def step(self, optimizer):
            pass
        def update(self):
            self.value /= 2
    trainer.scaler = SkippingScaler()
    scheduler_step = trainer.scheduler.last_epoch
    row = trainer.train_epoch([{'input_ids': torch.tensor([[1, 2, 3]])}])[0]
    assert row['update_skipped'] == 1
    assert trainer.scheduler.last_epoch == scheduler_step
    assert trainer.state.optimizer_steps == 0 and trainer.state.attempted_steps == 1
    for before, after in zip(original, trainer.model.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FP16 requires GPU')
def test_cuda_fp16_gradscaler_smoke():
    trainer = Trainer(small_model().cuda(), train_config(precision='fp16'), stage='pretrain', total_steps=2)
    rows = trainer.train_epoch([{'input_ids': torch.tensor([[1, 2, 3, 4]])}]*2)
    assert all(torch.isfinite(torch.tensor(row['loss'])) for row in rows)
    assert trainer.scaler.is_enabled()
