"""One-command core demo: load checkpoint, check forward, generate tokens."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

from llm_reasoning.checkpoint import file_sha256, load_model_artifact, save_checkpoint
from llm_reasoning.config import ModelConfig, TokenizerConfig, TrainConfig
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.generation import generate
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer


def create_smoke_artifact(directory: Path, seed: int, *, max_context_tokens: int = 256) -> tuple[Path, Path]:
    """Create a tiny random-weight checkpoint, save, then load via normal path."""
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=512))
    tokenizer.train(['Question: 2 + 3 = ?\n<think>\nAdd two and three. The answer is five.'] * 20)
    tokenizer_path = directory/'tokenizer.json'
    tokenizer.save(tokenizer_path)
    model = TransformerLM(ModelConfig(n_layers=2, d_model=32, n_heads=4, d_ff=128,
                                      max_context_tokens=max_context_tokens), tokenizer.vocab_size)
    config = TrainConfig(precision='fp32', pretrain_seq_len=16, sft_seq_len=16,
                         micro_batch_size=1, gradient_accumulation_steps=1)
    trainer = Trainer(model, config, stage='pretrain', total_steps=1)
    # This stream only supplies checkpoint metadata. No train_epoch is called.
    ids = tokenizer.encode('2 + 3 = 5')
    fingerprint = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    stream = ResumableBatchStream([{'input_ids': torch.tensor([ids])}], fingerprint=fingerprint)
    checkpoint_path = directory/'random_model.pt'
    save_checkpoint(checkpoint_path, trainer, stream, tokenizer_path=tokenizer_path, checkpoint_kind='smoke')
    return checkpoint_path, tokenizer_path


@torch.inference_mode()
def run_demo(checkpoint: Path, tokenizer_path: Path, *, prompt: str,
             prompt_format: str = 'question', device: str = 'cpu', max_new_tokens: int = 32,
             temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
             use_cache: bool = True) -> dict:
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    model, metadata = load_model_artifact(checkpoint, tokenizer_path=tokenizer_path, device=device)
    if prompt_format not in ('raw', 'question'):
        raise ValueError("prompt_format must be raw or question")
    if not prompt.strip():
        raise ValueError("prompt must be nonempty")
    encoded_prompt = f'Question: {prompt}\n<think>\n' if prompt_format == 'question' else prompt
    ids = tokenizer.encode(encoded_prompt, allowed_special=True)
    if not ids or len(ids) > model.config.max_context_tokens:
        raise ValueError("encoded prompt exceeds context or is empty; no truncation is performed")
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)

    def sync():
        if input_ids.device.type == 'cuda':
            torch.cuda.synchronize(input_ids.device)

    sync()
    start = time.perf_counter()
    logits = model(input_ids)
    sync()
    forward_seconds = time.perf_counter() - start
    finite = bool(torch.isfinite(logits).all())
    shape = list(logits.shape)
    if not finite:
        raise FloatingPointError('forward produced nonfinite logits')
    del logits
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    eos = tokenizer.special_to_id.get('<eos>')
    sync()
    start = time.perf_counter()
    output = generate(model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos,
                      temperature=temperature, top_p=top_p, generator=generator, use_cache=use_cache)
    sync()
    generation_seconds = time.perf_counter() - start
    new_ids = output[0, len(ids):].tolist()
    reason = ('eos' if new_ids and new_ids[-1] == eos else
              'context_limit' if output.shape[1] == model.config.max_context_tokens else 'max_new_tokens')
    kind = metadata['checkpoint_kind']
    notice = ('Checkpoint kiểm tra với trọng số ngẫu nhiên; chưa train chính thức. '
              'Nội dung sinh ra chỉ minh họa cơ chế.' if kind == 'smoke' else
              'Nguồn checkpoint chưa được phân loại; demo không xác nhận chất lượng mô hình.' if kind == 'unclassified' else
              'Checkpoint được người tạo đánh dấu training; kết quả demo không thay thế benchmark chất lượng.')
    return dict(notice=notice, checkpoint_kind=kind, checkpoint=str(checkpoint),
                checkpoint_sha256=file_sha256(checkpoint), tokenizer_sha256=metadata['tokenizer_sha256'],
                checkpoint_stage=metadata['stage'], saved_train_state=metadata['train_state'],
                model_config=metadata['model_config'], parameters=model.num_parameters(), vocab_size=model.vocab_size,
                device=str(input_ids.device), torch_version=str(torch.__version__),
                forward=dict(logits_shape=shape, all_finite=finite, seconds=forward_seconds),
                decoding=dict(temperature=temperature, top_p=top_p, seed=seed, use_cache=use_cache,
                              max_new_tokens=max_new_tokens, eos_token_id=eos),
                prompt=prompt, encoded_prompt=encoded_prompt, prompt_tokens=len(ids),
                generated_tokens=len(new_ids), generated_token_ids=new_ids,
                generated_text=tokenizer.decode(new_ids), stop_reason=reason,
                generation_seconds=generation_seconds,
                timing_note='Demo timings only: no benchmark repetitions; not the official T4 profiling report.')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke', action='store_true', help='Create and reload a small random checkpoint; does not train')
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--smoke-dir', type=Path, default=Path('artifacts/task_2_8_demo'))
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path, help='UTF-8 text file')
    parser.add_argument('--prompt-format', choices=['question', 'raw'], default='question')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1, help='CPU threads for this demo, not a training recommendation')
    parser.add_argument('--cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, help='Optional JSON report; must not exist yet')
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0 or args.threads <= 0:
        parser.error('max-new-tokens must be nonnegative and threads positive')
    if not math.isfinite(args.temperature) or args.temperature < 0 or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error('temperature must be finite >= 0 and top-p in (0, 1]')
    if args.smoke and args.tokenizer is not None:
        parser.error('--smoke creates its own tokenizer; omit --tokenizer')
    if not args.smoke and args.tokenizer is None:
        parser.error('--checkpoint requires --tokenizer')
    try:
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('demo supports CPU or CUDA')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; select --device cpu')
        torch.set_num_threads(args.threads)
        prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else args.prompt if args.prompt is not None else '2 + 3 = ?'
        if not prompt.strip():
            raise ValueError('prompt must be nonempty')
        checkpoint, tokenizer = (create_smoke_artifact(args.smoke_dir, args.seed) if args.smoke else
                                 (args.checkpoint, args.tokenizer))
        result = run_demo(checkpoint, tokenizer, prompt=prompt, prompt_format=args.prompt_format,
                          device=args.device, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          top_p=args.top_p, seed=args.seed, use_cache=args.cache)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as output:
                output.write(text + '\n')
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    print(text)


if __name__ == '__main__':
    main()
