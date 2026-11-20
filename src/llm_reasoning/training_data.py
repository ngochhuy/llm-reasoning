"""Indexed, read-only training micro-batches for the prepared artifacts."""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from llm_reasoning.checkpoint import file_sha256
from llm_reasoning.training import collate_sft


def _fingerprint(paths: list[Path], settings: dict) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


class PretrainBatches:
    """Memory-map fixed-length uint16 blocks; batch boundaries are deterministic."""

    def __init__(self, root: str | Path, split: str, *, batch_size: int,
                 max_batches: int | None = None) -> None:
        import numpy as np

        if split not in ('train', 'validation') or batch_size <= 0:
            raise ValueError('invalid pretrain split or batch size')
        root = Path(root)
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['token_dtype'] != 'uint16' or manifest['byte_order'] != sys.byteorder:
            raise ValueError('pretrain shard dtype or byte order mismatch')
        self.block_size = manifest['block_size']
        if self.block_size < 2:
            raise ValueError('pretrain block size must be at least two')
        paths = []
        for source, info in manifest['sources'].items():
            for name in info['shards'][split]['files']:
                path = root / source / split / name
                if not path.is_file() or path.stat().st_size % (self.block_size * 2):
                    raise ValueError(f'missing or malformed shard: {path}')
                paths.append(path)
        if not paths:
            raise ValueError(f'no {split} pretrain shards')
        self.maps = [np.memmap(path, dtype=np.uint16, mode='r') for path in paths]
        self.blocks = [(map_, i) for map_ in self.maps for i in range(len(map_) // self.block_size)]
        full_batches = len(self.blocks) // batch_size
        self.count = min(full_batches, max_batches) if max_batches is not None else full_batches
        if self.count <= 0 or (max_batches is not None and max_batches <= 0):
            raise ValueError('no complete pretrain micro-batches')
        self.batch_size = batch_size
        self.fingerprint = _fingerprint([manifest_path, *paths],
                                        dict(kind='pretrain', split=split, batch_size=batch_size,
                                             block_size=self.block_size, max_batches=max_batches))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        rows = []
        for map_, block in self.blocks[index * self.batch_size:(index + 1) * self.batch_size]:
            # Copy from read-only memmap before constructing a writable tensor.
            start = block * self.block_size
            rows.append(torch.tensor(map_[start:start + self.block_size].copy(), dtype=torch.long))
        return {'input_ids': torch.stack(rows)}


class SFTBatches:
    """Index JSONL byte offsets without retaining all token lists in RAM."""

    def __init__(self, root: str | Path, split: str, *, batch_size: int,
                 pad_token_id: int, max_seq_len: int, max_batches: int | None = None) -> None:
        if split not in ('train', 'validation') or batch_size <= 0 or max_seq_len < 2:
            raise ValueError('invalid SFT split, batch size or sequence length')
        root = Path(root)
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('sft_sources') != ['gsm8k']:
            raise ValueError('expected GSM8K-only SFT manifest')
        self.path = root / ('gsm_train.jsonl' if split == 'train' else 'gsm_validation.jsonl')
        self.offsets = []
        with self.path.open('rb') as file:
            while file.readline():
                self.offsets.append(file.tell())
        self.offsets.insert(0, 0)
        self.offsets.pop()
        full_batches = math.ceil(len(self.offsets) / batch_size)
        self.count = min(full_batches, max_batches) if max_batches is not None else full_batches
        if self.count <= 0 or (max_batches is not None and max_batches <= 0):
            raise ValueError('no SFT examples')
        self.batch_size, self.pad_token_id, self.max_seq_len = batch_size, pad_token_id, max_seq_len
        self.fingerprint = _fingerprint([manifest_path, self.path],
                                        dict(kind='sft', split=split, batch_size=batch_size,
                                             max_seq_len=max_seq_len, pad_token_id=pad_token_id,
                                             max_batches=max_batches))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        selected = self.offsets[index * self.batch_size:(index + 1) * self.batch_size]
        rows = []
        with self.path.open('rb') as file:
            for offset in selected:
                file.seek(offset)
                row = json.loads(file.readline())
                if not 2 <= len(row['input_ids']) <= self.max_seq_len:
                    raise ValueError('SFT example exceeds configured length')
                rows.append(row)
        return collate_sft(rows, self.pad_token_id)
