"""Prove continuous vs resumed training equivalence on tiny synthetic CPU data."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from llm_reasoning.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from llm_reasoning.config import ModelConfig, TokenizerConfig, TrainConfig
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer, TrainingLogger, collate_sft


def new_stream():
    generator = torch.Generator().manual_seed(113)
    records = [dict(input_ids=torch.randint(0, 40, (6,), generator=generator).tolist(),
                    loss_mask=[0, 0, 1, 1, 1, 1]) for _ in range(13)]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return ResumableBatchStream([collate_sft([row], 0) for row in records], fingerprint=fingerprint, seed=9)


def new_trainer(stage):
    random.seed(42)
    torch.manual_seed(42)
    geometry = ModelConfig(n_layers=1, d_model=8, n_heads=2, d_ff=16, max_context_tokens=16)
    model = TransformerLM(geometry, vocab_size=262)
    config = TrainConfig(precision='fp32', pretrain_seq_len=6, sft_seq_len=6,
                         micro_batch_size=1, gradient_accumulation_steps=2, warmup_fraction=.2)
    return Trainer(model, config, stage=stage, total_steps=10)


def finish(trainer, stream):
    rows = trainer.train_epoch(stream)
    if trainer.state.attempted_steps < trainer.total_steps:
        stream.next_epoch()
        rows += trainer.train_epoch(stream)
    return rows


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/task_2_7_smoke'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tokenizer = args.output/'tokenizer.json'
    ByteBPETokenizer(TokenizerConfig(vocab_size=262)).save(tokenizer)
    summary = {}
    for stage in ('pretrain', 'sft'):
        full, full_stream = new_trainer(stage), new_stream()
        with TrainingLogger(args.output/stage/'continuous') as logger:
            full.logger = logger
            full_rows = finish(full, full_stream)
        expected_rng = torch.get_rng_state()
        interrupted, stream = new_trainer(stage), new_stream()
        log_dir = args.output/stage/'resumed'
        path = args.output/stage/'latest.pt'
        with TrainingLogger(log_dir) as logger:
            interrupted.logger = logger
            rows = interrupted.train_epoch(stream, max_updates=3)
            save_checkpoint(path, interrupted, stream, tokenizer_path=tokenizer)
        torch.rand(100)
        random.random()
        stream = new_stream()
        with TrainingLogger(log_dir, resume_step=3) as logger:
            resumed = load_checkpoint(path, stream, tokenizer_path=tokenizer, expected_stage=stage, logger=logger)
            rows += finish(resumed, stream)
        checks = dict(
            model_exact=equal_nested(full.model.state_dict(), resumed.model.state_dict()),
            optimizer_exact=equal_nested(full.optimizer.state_dict(), resumed.optimizer.state_dict()),
            scheduler_exact=equal_nested(full.scheduler.state_dict(), resumed.scheduler.state_dict()),
            scaler_exact=equal_nested(full.scaler.state_dict(), resumed.scaler.state_dict()),
            data_position_exact=equal_nested(full_stream.state_dict(), stream.state_dict()),
            counters_exact=asdict(full.state) == asdict(resumed.state),
            loss_lr_exact=[(r['loss'], r['lr']) for r in full_rows] == [(r['loss'], r['lr']) for r in rows],
            torch_rng_exact=torch.equal(expected_rng, torch.get_rng_state()),
        )
        if not all(checks.values()):
            raise RuntimeError(f'Resume mismatch: {checks}')
        summary[stage] = dict(checks=checks, state=asdict(resumed.state), epoch=stream.epoch,
                              next_batch_cursor=stream.cursor, interruption_step=3,
                              max_parameter_difference=max(float((a.detach()-b.detach()).abs().max()) for a, b in
                                                           zip(full.model.parameters(), resumed.model.parameters())),
                              checkpoint_sha256=file_sha256(path), tokenizer_sha256=file_sha256(tokenizer),
                              model_config=asdict(resumed.model.config), train_config=asdict(resumed.config),
                              toy_data=True, device='cpu')
        # Export completed stage checkpoint too; latest.pt remains the interruption checkpoint.
        save_checkpoint(args.output/stage/'completed.pt', resumed, stream, tokenizer_path=tokenizer)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
