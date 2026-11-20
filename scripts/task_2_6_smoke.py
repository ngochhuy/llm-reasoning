"""CPU-only toy overfit check; never trains the project's production model."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from llm_reasoning.config import ModelConfig, TrainConfig
from llm_reasoning.model import TransformerLM
from llm_reasoning.training import Trainer, TrainingLogger, collate_sft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/task_2_6_smoke'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {}
    for stage in ('pretrain', 'sft'):
        torch.manual_seed(42)
        geometry = ModelConfig(n_layers=1, d_model=16, n_heads=4, d_ff=32, max_context_tokens=16)
        model = TransformerLM(geometry, vocab_size=17)
        config = TrainConfig(precision='fp32', gradient_accumulation_steps=2,
                             pretrain_seq_len=6, sft_seq_len=6, micro_batch_size=1,
                             pretrain_lr=.01, sft_lr=.01, gradient_clip_norm=1.0,
                             warmup_fraction=.05)
        records = [dict(input_ids=[1, 2, 3, 4, 5, 6], loss_mask=[0, 0, 1, 1, 1, 1])]
        batch = collate_sft(records, pad_token_id=0)
        with TrainingLogger(args.output/stage) as logger:
            trainer = Trainer(model, config, stage=stage, total_steps=60, logger=logger)
            before = trainer.evaluate([batch])['loss']
            trainer.train_epoch([batch]*120)
            after = trainer.evaluate([batch])['loss']
        results[stage] = dict(initial_loss=before, final_loss=after, state=asdict(trainer.state),
                              model_parameters=model.num_parameters(), model_config=asdict(geometry),
                              train_config=asdict(config), seed=42, device='cpu', toy_data=True)
    (args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    if not all(r['final_loss'] < r['initial_loss']*.15 for r in results.values()):
        raise RuntimeError('Toy overfit check failed')


if __name__ == '__main__':
    main()
