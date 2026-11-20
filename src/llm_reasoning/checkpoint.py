"""Atomic full-state checkpoints at completed optimizer-update boundaries."""

import hashlib
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from llm_reasoning.config import ModelConfig, TrainConfig
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer, TrainingLogger, TrainState


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng() -> dict:
    result = dict(python=random.getstate(), torch_cpu=torch.get_rng_state(),
                  torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], numpy=None)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        result['numpy'] = (algorithm, keys.tolist(), position, has_gauss, cached_gaussian)
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from saved run")
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    if state['numpy'] is not None:
        import numpy as np
        algorithm, keys, position, has_gauss, cached_gaussian = state['numpy']
        np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def save_checkpoint(path: str | Path, trainer: Trainer, stream: ResumableBatchStream,
                    *, tokenizer_path: str | Path, checkpoint_kind: str = 'unclassified') -> None:
    """Synchronous temp-file + fsync + replace; preserve the last file on failure.

    Call after a completed accumulation window, e.g. from Trainer.on_update.
    stream must be the exact object passed to train_epoch, with no prefetch.
    """
    if checkpoint_kind not in ('unclassified', 'smoke', 'training'):
        raise ValueError("invalid checkpoint kind")
    if not trainer._at_update_boundary or any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError("checkpoint requires a completed update boundary with no pending gradients")
    if trainer._data_source is not None and trainer._data_source is not stream:
        raise ValueError("checkpoint stream must be the same object used by Trainer")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != trainer.model.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model")
    payload = dict(
        format_version=1, checkpoint_kind=checkpoint_kind, model_config=asdict(trainer.model.config),
        model_dtype=str(next(trainer.model.parameters()).dtype).removeprefix('torch.'),
        vocab_size=trainer.model.vocab_size, train_config=asdict(trainer.config),
        stage=trainer.stage, total_steps=trainer.total_steps, weight_decay=trainer.weight_decay,
        model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
        scheduler=trainer.scheduler.state_dict(), scaler=trainer.scaler.state_dict(),
        train_state=asdict(trainer.state), data_stream=stream.state_dict(),
        tokenizer_sha256=file_sha256(tokenizer_path),
        rng=capture_rng(), device_type=trainer.device.type,
        module_training=[module.training for module in trainer.model.modules()],
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
    )
    payload = _cpu_snapshot(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read(path: str | Path, tokenizer_path: str | Path) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError("unsupported checkpoint format")
    if payload['tokenizer_sha256'] != file_sha256(tokenizer_path):
        raise ValueError("tokenizer SHA256 mismatch")
    if ByteBPETokenizer.load(tokenizer_path).vocab_size != payload['vocab_size']:
        raise ValueError("tokenizer vocabulary mismatch")
    return payload


def _model_from_payload(payload: dict, device) -> TransformerLM:
    dtype = {'float32': torch.float32, 'float64': torch.float64,
             'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(payload['model_dtype'])
    if dtype is None:
        raise ValueError("unsupported model dtype")
    model = TransformerLM(ModelConfig(**payload['model_config']), payload['vocab_size']).to(device=device, dtype=dtype)
    model.load_state_dict(payload['model'], strict=True)
    return model


def load_checkpoint(path: str | Path, stream: ResumableBatchStream, *,
                    tokenizer_path: str | Path, device: str | torch.device = 'cpu',
                    expected_stage: str | None = None, logger: TrainingLogger | None = None) -> Trainer:
    """Rebuild trainer from saved config and restore RNG last, after initialization.

    Recreate the same immutable micro-batch sequence and fingerprint first. No
    override of stage/schedule is allowed: SFT transfer uses load_model_checkpoint.
    """
    payload = _read(path, tokenizer_path)
    stream.validate_state(payload['data_stream'])
    device = torch.device(device)
    if device.type != payload['device_type']:
        raise ValueError("exact resume requires the original device type")
    if expected_stage is not None and expected_stage != payload['stage']:
        raise ValueError("checkpoint stage mismatch; use model-only loading for stage transfer")
    model = _model_from_payload(payload, device)
    trainer = Trainer(model, TrainConfig(**payload['train_config']), stage=payload['stage'],
                      total_steps=payload['total_steps'], weight_decay=payload['weight_decay'], logger=logger)
    trainer.optimizer.load_state_dict(payload['optimizer'])
    trainer.scheduler.load_state_dict(payload['scheduler'])
    trainer.scaler.load_state_dict(payload['scaler'])
    trainer.state = TrainState(**payload['train_state'])
    if not 0 <= trainer.state.optimizer_steps <= trainer.state.attempted_steps <= trainer.total_steps:
        raise ValueError("invalid training step counters")
    for module, training in zip(model.modules(), payload['module_training'], strict=True):
        module.training = training
    stream.load_state_dict(payload['data_stream'])
    trainer._data_source = stream
    restore_rng(payload['rng'])
    return trainer


def load_model_checkpoint(path: str | Path, *, tokenizer_path: str | Path,
                          device: str | torch.device = 'cpu') -> TransformerLM:
    """Load weights/config only for inference or creating a new SFT optimizer."""
    return load_model_artifact(path, tokenizer_path=tokenizer_path, device=device)[0]


def load_model_artifact(path: str | Path, *, tokenizer_path: str | Path,
                        device: str | torch.device = 'cpu') -> tuple[TransformerLM, dict]:
    """Load a model and scalar provenance in one checkpoint read (no RNG restore)."""
    payload = _read(path, tokenizer_path)
    metadata = {key: payload[key] for key in ('model_config', 'vocab_size', 'stage',
                                             'total_steps', 'train_state', 'tokenizer_sha256',
                                             'environment')}
    metadata['checkpoint_kind'] = payload.get('checkpoint_kind', 'unclassified')
    return _model_from_payload(payload, device).eval(), metadata


class PeriodicCheckpoint:
    """Save at update boundaries when the wall-clock interval elapses."""

    def __init__(self, path: str | Path, stream: ResumableBatchStream, *,
                 tokenizer_path: str | Path, interval_minutes: float = 30) -> None:
        import time
        if not 0 < interval_minutes < float('inf'):
            raise ValueError("checkpoint interval must be positive and finite")
        self.path, self.stream, self.tokenizer_path = path, stream, tokenizer_path
        self.interval_seconds = interval_minutes * 60
        self.last_save = time.monotonic()

    def __call__(self, trainer: Trainer) -> None:
        import time
        if time.monotonic() - self.last_save >= self.interval_seconds:
            save_checkpoint(self.path, trainer, self.stream, tokenizer_path=self.tokenizer_path)
            self.last_save = time.monotonic()
