"""Single-device pretrain/SFT loop with update-boundary checkpoint hooks."""

import csv
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from llm_reasoning.config import TrainConfig
from llm_reasoning.model import TransformerLM


def collate_sft(records: list[dict], pad_token_id: int) -> dict[str, Tensor]:
    """Right-pad records, keeping response-loss and valid-token masks distinct."""
    if not records or any(len(r['input_ids']) < 2 or len(r['input_ids']) != len(r['loss_mask']) for r in records):
        raise ValueError("SFT records must have matching IDs/masks and at least two tokens")
    length = max(len(r['input_ids']) for r in records)
    ids = torch.full((len(records), length), pad_token_id, dtype=torch.long)
    response = torch.zeros_like(ids, dtype=torch.bool)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, record in enumerate(records):
        n = len(record['input_ids'])
        if any(v not in (0, 1) for v in record['loss_mask']):
            raise ValueError("loss_mask must be binary")
        ids[i, :n] = torch.tensor(record['input_ids'])
        response[i, :n] = torch.tensor(record['loss_mask'], dtype=torch.bool)
        valid[i, :n] = True
    return {'input_ids': ids, 'loss_mask': response, 'attention_mask': valid}


def target_mask(batch: dict[str, Tensor], *, sft: bool) -> Tensor:
    ids = batch['input_ids']
    if ids.ndim != 2 or ids.shape[1] < 2 or ids.shape[0] == 0 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("training IDs must be integer [nonempty batch, sequence >= 2]")
    valid = torch.ones_like(ids, dtype=torch.bool)
    for name in ('attention_mask', 'loss_mask'):
        if name in batch:
            mask = batch[name]
            if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all():
                raise ValueError(f"{name} must be binary with input shape")
    if 'attention_mask' in batch:
        valid = batch['attention_mask'].bool()
        if (valid[:, 1:] & ~valid[:, :-1]).any() or not valid[:, 0].all():
            raise ValueError("only contiguous right padding is supported")
    selected = valid[:, 1:] & valid[:, :-1]
    if sft:
        if 'loss_mask' not in batch:
            raise ValueError("SFT requires a response loss_mask")
        selected = selected & batch['loss_mask'][:, 1:].bool()
    return selected


def causal_lm_loss(logits: Tensor, batch: dict[str, Tensor], *, sft: bool = False) -> tuple[Tensor, int]:
    """Return summed shifted CE and supervised-target count (not batch mean)."""
    selected = target_mask(batch, sft=sft)
    if logits.shape[:2] != batch['input_ids'].shape or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    labels = batch['input_ids'][:, 1:].long().masked_fill(~selected, -100)
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100, reduction='sum')
    return loss, int(selected.sum().item())


def lr_multiplier(index: int, *, total_steps: int, warmup_steps: int) -> float:
    """Zero-based update index: linear warmup, then cosine to 10% of peak LR."""
    if total_steps <= 0 or not 0 <= warmup_steps <= total_steps:
        raise ValueError("invalid schedule lengths")
    index = min(max(index, 0), total_steps - 1)
    if index < warmup_steps:
        return (index + 1) / warmup_steps
    progress = (index - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


class TrainingLogger:
    """CSV/TensorBoard writer; explicit resume rejects logs ahead of checkpoint."""

    fields = ['kind', 'stage', 'step', 'loss', 'lr', 'grad_norm', 'update_skipped',
              'processed_tokens', 'supervised_tokens', 'seconds', 'tokens_per_second']

    def __init__(self, directory: str | Path, *, resume_step: int | None = None) -> None:
        from torch.utils.tensorboard import SummaryWriter
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'metrics.csv'
        exists = path.exists()
        self._last_train_step = -1
        if resume_step is not None and resume_step < 0:
            raise ValueError("resume_step must be nonnegative")
        if exists and resume_step is not None:
            with path.open(newline='', encoding='utf-8') as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != self.fields:
                    raise ValueError("CSV schema does not match")
                for row in reader:
                    if int(row['step']) > resume_step:
                        raise ValueError("logs are ahead of checkpoint; choose a new log directory")
                    if row['kind'] == 'train':
                        self._last_train_step = max(self._last_train_step, int(row['step']))
        self.file = path.open('a' if resume_step is not None else 'x', newline='', encoding='utf-8')
        self.csv = csv.DictWriter(self.file, fieldnames=self.fields)
        if not exists:
            self.csv.writeheader()
        self.writer = SummaryWriter(str(directory / 'tensorboard'),
                                    purge_step=resume_step + 1 if resume_step is not None else None)

    def log(self, row: dict) -> None:
        if row['kind'] == 'train':
            if row['step'] <= self._last_train_step:
                raise ValueError("training log step must increase")
            self._last_train_step = row['step']
        self.csv.writerow(row)
        self.file.flush()
        for name in ('loss', 'lr', 'grad_norm', 'update_skipped', 'processed_tokens',
                     'supervised_tokens', 'tokens_per_second'):
            if name in row:
                self.writer.add_scalar(f"{row['stage']}/{row['kind']}/{name}", row[name], row['step'])
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class TrainState:
    optimizer_steps: int = 0
    attempted_steps: int = 0
    processed_tokens: int = 0
    supervised_tokens: int = 0


class Trainer:
    """One stage/optimizer per instance; CPU fp32 or CUDA fp32/fp16."""

    def __init__(self, model: TransformerLM, config: TrainConfig, *, stage: str,
                 total_steps: int, logger: TrainingLogger | None = None,
                 weight_decay: float = .1) -> None:
        if stage not in ('pretrain', 'sft') or total_steps <= 0:
            raise ValueError("stage must be pretrain/sft and total_steps positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.model, self.config, self.stage = model, config, stage
        self.total_steps, self.logger = total_steps, logger
        self.weight_decay = weight_decay
        self._at_update_boundary = True
        self._data_source = None
        self.device = next(model.parameters()).device
        if config.precision == 'fp16' and self.device.type != 'cuda':
            raise ValueError("FP16 training requires CUDA; use fp32 for CPU checks")
        self.amp_enabled = config.precision == 'fp16'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        lr = config.pretrain_lr if stage == 'pretrain' else config.sft_lr
        self.optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                                            {'params': no_decay, 'weight_decay': 0.0}], lr=lr)
        warmup = min(total_steps, math.ceil(config.warmup_fraction * total_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda index: lr_multiplier(index, total_steps=total_steps, warmup_steps=warmup))
        self.state = TrainState()

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.float16) if self.amp_enabled else nullcontext()

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def train_epoch(self, batches: Iterable[dict[str, Tensor]], *, max_updates: int | None = None,
                    on_update: Callable[['Trainer'], None] | None = None) -> list[dict]:
        """Process a finite iterable; flush a partial final accumulation window.

        Each window is normalized by its total supervised tokens, not the mean
        of microbatch means. Input batches should be CPU tensors (moved lazily).
        """
        if max_updates is not None and (not isinstance(max_updates, int) or max_updates <= 0):
            raise ValueError("max_updates must be a positive integer")
        if not self._at_update_boundary:
            raise ValueError("previous update failed; restore a checkpoint before continuing")
        self._data_source = batches
        self.model.train()
        iterator, rows = iter(batches), []
        while self.state.attempted_steps < self.total_steps:
            if max_updates is not None and len(rows) >= max_updates:
                break
            self._at_update_boundary = False
            window = list(islice(iterator, self.config.gradient_accumulation_steps))
            if not window:
                self._at_update_boundary = True
                break
            count = sum(int(target_mask(b, sft=self.stage == 'sft').sum()) for b in window)
            if count == 0:
                raise ValueError("accumulation window has no supervised targets")
            self._sync()
            start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            loss_total, tokens = 0.0, 0
            for batch in window:
                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                tokens += int(batch.get('attention_mask', torch.ones_like(batch['input_ids'])).sum())
                with self._autocast():
                    summed, _ = causal_lm_loss(self.model(device_batch['input_ids']), device_batch, sft=self.stage == 'sft')
                if not torch.isfinite(summed):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("nonfinite training loss")
                self.scaler.scale(summed / count).backward()
                loss_total += float(summed.detach())
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            if not self.amp_enabled and not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("nonfinite gradients")
            lr = self.optimizer.param_groups[0]['lr']
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped = self.scaler.get_scale() < scale_before
            if not skipped:
                self.scheduler.step()
                self.state.optimizer_steps += 1
            self.state.attempted_steps += 1
            self.state.processed_tokens += tokens
            self.state.supervised_tokens += count
            self.optimizer.zero_grad(set_to_none=True)
            self._sync()
            seconds = time.perf_counter() - start
            row = dict(kind='train', stage=self.stage, step=self.state.attempted_steps,
                       loss=loss_total/count, lr=lr, grad_norm=float(grad_norm), update_skipped=int(skipped),
                       processed_tokens=self.state.processed_tokens, supervised_tokens=self.state.supervised_tokens,
                       seconds=seconds, tokens_per_second=tokens/seconds)
            self._at_update_boundary = True
            if self.logger:
                self.logger.log(row)
            rows.append(row)
            if on_update:
                on_update(self)
        return rows

    @torch.no_grad()
    def evaluate(self, batches: Iterable[dict[str, Tensor]]) -> dict:
        flags = [(module, module.training) for module in self.model.modules()]
        summed_loss, targets = 0.0, 0
        try:
            self.model.eval()
            for batch in batches:
                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                with self._autocast():
                    summed, count = causal_lm_loss(self.model(device_batch['input_ids']), device_batch, sft=self.stage == 'sft')
                if not torch.isfinite(summed):
                    raise FloatingPointError("nonfinite validation loss")
                summed_loss += float(summed)
                targets += count
        finally:
            for module, training in flags:
                module.training = training
        if targets == 0:
            raise ValueError("validation has no supervised targets")
        row = dict(kind='validation', stage=self.stage, step=self.state.attempted_steps, loss=summed_loss/targets)
        if self.logger:
            self.logger.log(row)
        return row
